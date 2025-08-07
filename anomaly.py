# === anomaly.py - Backend for Anomaly Detection Model ===
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

def build_autoencoder(seq_len, features):
    """Build and return an LSTM autoencoder model"""
    input_layer = Input(shape=(seq_len, features))
    x = LSTM(64, activation='relu')(input_layer)
    x = RepeatVector(seq_len)(x)
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    output_layer = TimeDistributed(Dense(features))(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def load_model(model_path, seq_len, features):
    """Load or rebuild the model"""
    try:
        # Try to rebuild the model and load weights
        model = build_autoencoder(seq_len, features)
        model.load_weights(model_path)
        return model, True
    except:
        # If loading fails, build a new model (without trained weights)
        model = build_autoencoder(seq_len, features)
        return model, False

def create_sequences(data, seq_len):
    """Create sequences for LSTM input"""
    if isinstance(data, pd.DataFrame):
        data = data.values
    return np.array([data[i:i+seq_len] for i in range(len(data) - seq_len)])

def detect_anomalies(model, sequences, threshold_factor=2.0):
    """Detect anomalies using the model"""
    # Generate reconstructions
    reconstructions = model.predict(sequences)
    
    # Calculate mean squared error
    mse = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
    
    # Calculate threshold
    threshold = np.mean(mse) + threshold_factor * np.std(mse)
    
    # Identify anomalies
    anomalies = mse > threshold
    
    return mse, threshold, anomalies, reconstructions

def preprocess_data(data):
    """Preprocess the data for anomaly detection"""
    # Store timestamp separately and remove it from the data to be normalized
    if 'timestamp' in data.columns:
        timestamps = data['timestamp'].copy()
        numeric_data = data.drop(columns=['timestamp'])
    else:
        timestamps = pd.Series(range(len(data)))
        numeric_data = data
    
    # Normalize the numeric data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    
    return scaled_data, numeric_data, timestamps, scaler

def generate_demo_data():
    """Generate demo data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2025-04-10', periods=1000, freq='5min')
    
    # Create demo dataframe
    demo_data = {
        'timestamp': dates,
        'heart_rate': np.random.normal(75, 10, 1000),
        'blood_pressure_systolic': np.random.normal(120, 10, 1000),
        'blood_pressure_diastolic': np.random.normal(80, 8, 1000),
        'temperature': np.random.normal(36.5, 0.5, 1000),
        'spo2': np.random.normal(98, 2, 1000),
        'respiratory_rate': np.random.normal(16, 2, 1000),
        'ecg_abnormality': np.random.choice([0, 1], 1000, p=[0.95, 0.05]),
        'consciousness_level': np.random.choice([15, 14, 13], 1000, p=[0.9, 0.07, 0.03]),
        'pain_score': np.random.normal(2, 1.5, 1000),
        'movement_activity': np.random.choice([1, 2, 3, 4], 1000),
        'fall_detected': np.random.choice([0, 1], 1000, p=[0.99, 0.01]),
        'medication_flag': np.random.choice([0, 1], 1000, p=[0.85, 0.15]),
        'nurse_call_button': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    }
    
    # Add a few anomalies to the demo data
    for i in range(5):
        idx = np.random.randint(100, 900)
        demo_data['heart_rate'][idx:idx+5] = np.random.normal(120, 15, 5)
        demo_data['blood_pressure_systolic'][idx:idx+5] = np.random.normal(160, 10, 5)
        demo_data['spo2'][idx:idx+5] = np.random.normal(85, 5, 5)
        demo_data['fall_detected'][idx:idx+3] = 1
    
    return pd.DataFrame(demo_data)

def calculate_feature_contributions(sequences, reconstructions, anomaly_indices, numeric_data):
    """Calculate feature contribution to anomalies"""
    feature_contribution = {}
    for idx in anomaly_indices:
        feature_errors = np.mean(np.square(sequences[idx] - reconstructions[idx]), axis=0)
        for i, err in enumerate(feature_errors):
            feature_name = numeric_data.columns[i]
            if feature_name in feature_contribution:
                feature_contribution[feature_name].append(err)
            else:
                feature_contribution[feature_name] = [err]
    
    avg_contribution = {feat: np.mean(errs) for feat, errs in feature_contribution.items()}
    sorted_features = sorted(avg_contribution.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_features, feature_contribution