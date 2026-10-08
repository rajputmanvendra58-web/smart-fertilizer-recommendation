import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_smart_farm_model():
    data = {
        'N': [85, 60, 20, 40, 26, 42, 50, 75, 12, 34, 65, 90, 15, 45, 55],
        'P': [18, 55, 35, 72, 35, 48, 20, 45, 10, 40, 50, 22, 12, 60, 30],
        'K': [37, 44, 15, 80, 18, 42, 30, 38, 12, 25, 45, 35, 10, 50, 20],
        'temperature': [20.8, 23.0, 22.6, 26.5, 21.5, 23.2, 24.1, 22.0, 19.5, 25.0, 21.0, 24.5, 18.2, 27.0, 23.5],
        'humidity': [82.0, 60.3, 52.1, 80.5, 45.2, 70.1, 65.0, 78.2, 50.0, 62.1, 75.0, 85.1, 40.5, 68.0, 55.4],
        'ph': [6.5, 7.0, 5.8, 6.2, 7.2, 6.8, 6.0, 6.7, 5.5, 7.1, 6.4, 6.6, 5.2, 6.3, 6.9],
        'rainfall': [202.9, 90.0, 65.0, 150.0, 75.5, 110.2, 85.0, 120.0, 50.2, 95.0, 130.5, 210.0, 45.0, 160.2, 100.5],
        'soil_type': [2, 1, 0, 3, 1, 2, 0, 1, 3, 0, 2, 1, 3, 2, 0],
        'crop': ['rice', 'maize', 'chickpea', 'cotton', 'mothbeans', 'rice', 'maize', 'rice', 'kidneybeans', 'mungbean', 'maize', 'rice', 'mothbeans', 'cotton', 'mungbean'],
        'fertilizer': ['Urea', 'DAP', '14-35-14', '28-28', 'MOP', 'Urea', 'DAP', 'Urea', '14-35-14', 'MOP', 'DAP', 'Urea', '28-28', 'DAP', 'MOP']
    }
    
    df = pd.DataFrame(data)
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'soil_type']]
    
    crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
    crop_model.fit(X, df['crop'])
    
    fert_model = RandomForestClassifier(n_estimators=100, random_state=42)
    fert_model.fit(X, df['fertilizer'])
    
    models = {'crop_model': crop_model, 'fert_model': fert_model}
    with open('model.pkl', 'wb') as f:
        pickle.dump(models, f)
        
    print("Models successfully trained and saved!")

if __name__ == '__main__':
    train_smart_farm_model()