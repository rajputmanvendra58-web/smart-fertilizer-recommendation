from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pickle
import numpy as np
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'smart_farm_secret_key'

# Load models safely
try:
    models = pickle.load(open('model.pkl', 'rb'))
    crop_model = models['crop_model']
    fert_model = models['fert_model']
except Exception as e:
    print(f"Error loading models: {e}")

def init_db():
    try:
        conn = sqlite3.connect('history.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp TEXT,
                city TEXT,
                nitrogen REAL,
                phosphorus REAL,
                potassium REAL,
                temperature REAL,
                humidity REAL,
                ph REAL,
                rainfall REAL,
                soil_type INTEGER,
                predicted_crop TEXT,
                prediction TEXT,
                alternative TEXT,
                quantity TEXT,
                confidence REAL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")

init_db()

fertilizer_guides = {
    'Urea': {
        'application': 'Apply in split doses (basal and top-dressing) to prevent nitrogen loss.',
        'irrigation': 'Irrigate immediately after application to incorporate nitrogen into the root zone.',
        'soil_tip': 'Continuous use can increase soil acidity; monitor pH regularly.',
        'quantity': '45-50 kg per acre'
    },
    'DAP': {
        'application': 'Best applied as a basal dose during sowing, placed close to the root zone.',
        'irrigation': 'Ensure adequate soil moisture at the time of sowing for phosphorus absorption.',
        'soil_tip': 'Safe for most soil types and provides essential early root strength.',
        'quantity': '50 kg per acre'
    },
    'MOP': {
        'application': 'Apply during land preparation or early vegetative growth stages.',
        'irrigation': 'Maintain moderate soil moisture; avoid excessive waterlogging.',
        'soil_tip': 'Enhances potassium levels, improving drought and disease resistance.',
        'quantity': '25-30 kg per acre'
    },
    '14-35-14': {
        'application': 'Use as a starter fertilizer during planting for rapid early development.',
        'irrigation': 'Provide light watering right after planting.',
        'soil_tip': 'Balanced early-stage booster for root and shoot establishment.',
        'quantity': '40 kg per acre'
    },
    '28-28': {
        'application': 'Ideal for boosting vegetative growth mid-season.',
        'irrigation': 'Ensure regular irrigation schedule is maintained.',
        'soil_tip': 'Promotes lush leaf growth and overall plant vigor.',
        'quantity': '35 kg per acre'
    }
}

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['password']
            
            conn = sqlite3.connect('history.db')
            cursor = conn.cursor()
            cursor.execute('SELECT id, password FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()
            conn.close()
            
            if user and check_password_hash(user[1], password):
                session['user_id'] = user[0]
                session['username'] = username
                return redirect(url_for('dashboard'))
            else:
                return render_template('login.html', error='Invalid username or password')
        except Exception as e:
            return render_template('login.html', error='An error occurred during login. Please try again.')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = generate_password_hash(request.form['password'])
            
            conn = sqlite3.connect('history.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error='Username already exists!')
        except Exception as e:
            return render_template('register.html', error='Registration failed. Please try again.')
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    try:
        conn = sqlite3.connect('history.db')
        cursor = conn.cursor()
        cursor.execute('SELECT timestamp, city, nitrogen, phosphorus, potassium, ph, predicted_crop, prediction, alternative, quantity, confidence FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 10', (session['user_id'],))
        history = cursor.fetchall()
        conn.close()
        return render_template('index.html', history=history, username=session['username'])
    except Exception as e:
        return render_template('index.html', history=[], username=session.get('username', 'User'), error='Could not load your history.')

@app.route('/get_weather', methods=['POST'])
def get_weather():
    city = request.form.get('city')
    if not city:
        return jsonify({'success': False, 'message': 'City name cannot be empty'})
    
    return jsonify({
        'success': True, 
        'temperature': 28.5, 
        'humidity': 65.0
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    try:
        city = request.form.get('city', 'Manual Input')
        n = float(request.form['nitrogen'])
        p = float(request.form['phosphorus'])
        k = float(request.form['potassium'])
        temp = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        soil = int(request.form['soil_type'])

        features = np.array([[n, p, k, temp, humidity, ph, rainfall, soil]])
        
        predicted_crop = crop_model.predict(features)[0]
        
        probs = fert_model.predict_proba(features)[0]
        classes = fert_model.classes_
        top_indices = np.argsort(probs)[::-1]
        
        prediction = classes[top_indices[0]]
        alternative = classes[top_indices[1]] if len(top_indices) > 1 else "None"
        confidence = round(probs[top_indices[0]] * 100, 2)

        explanation = f"For growing {predicted_crop.capitalize()}, based on your soil's NPK levels and pH ({ph}), {prediction} is recommended."
        
        guide = fertilizer_guides.get(prediction, {
            'application': 'Apply uniformly as per standard agricultural practices.',
            'irrigation': 'Maintain regular moisture levels.',
            'soil_tip': 'Test soil periodically to maintain nutrient balance.',
            'quantity': '40 kg per acre'
        })
        
        quantity = guide['quantity']

        conn = sqlite3.connect('history.db')
        cursor = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        cursor.execute('''
            INSERT INTO predictions (user_id, timestamp, city, nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, soil_type, predicted_crop, prediction, alternative, quantity, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session['user_id'], timestamp, city, n, p, k, temp, humidity, ph, rainfall, soil, predicted_crop, prediction, alternative, quantity, confidence))
        conn.commit()

        cursor.execute('SELECT timestamp, city, nitrogen, phosphorus, potassium, ph, predicted_crop, prediction, alternative, quantity, confidence FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 10', (session['user_id'],))
        history = cursor.fetchall()
        conn.close()

        return render_template('index.html', 
                               predicted_crop=predicted_crop,
                               prediction_text=prediction, 
                               alternative=alternative,
                               quantity=quantity,
                               confidence=confidence, 
                               explanation=explanation,
                               guide=guide,
                               history=history,
                               username=session['username'])
                               
    except ValueError:
        return render_template('index.html', history=[], username=session.get('username', 'User'), error='Invalid input format. Please enter valid numeric values for soil parameters.')
    except Exception as e:
        return render_template('index.html', history=[], username=session.get('username', 'User'), error='An unexpected error occurred during prediction. Please check your inputs.')

if __name__ == '__main__':
    app.run(debug=True)