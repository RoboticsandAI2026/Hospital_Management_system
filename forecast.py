# === forecast.py - Backend for Time Series Forecasting Model ===
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model
from tensorflow.keras.metrics import MeanSquaredError

def load_forecast_model(model_path):
    """Load the trained forecasting model with custom metric handling"""
    try:
        # Register custom metrics before loading
        custom_objects = {
            'mse': MeanSquaredError(name='mse'),
            'mean_squared_error': MeanSquaredError(name='mean_squared_error')
        }
        model = tf_load_model(model_path, custom_objects=custom_objects)
        return model
    except Exception as e:
        raise ValueError(f"Error loading model: {str(e)}")

def build_forecast_model(seq_len, features):
    """Build a new forecasting model architecture"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(seq_len, features)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

def create_sequences(data, seq_len=30, target_col_index=0):
    """Create sequences for LSTM input"""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, target_col_index])  # Predict future heart_rate
    return np.array(X), np.array(y)

def predict_future(model, last_sequence, n_steps=50):
    """Generate future predictions using the latest available sequence"""
    curr_seq = last_sequence.copy()
    future_preds = []
    
    for _ in range(n_steps):
        # Get prediction for next time step
        next_pred = model.predict(curr_seq.reshape(1, len(curr_seq), -1), verbose=0)[0][0]
        future_preds.append(next_pred)
        
        # Update sequence: remove first element and add prediction at the end
        # Keep all features from the last time step except heart rate
        new_step = curr_seq[-1].copy()
        new_step[0] = next_pred  # Update only heart rate
        
        # Roll sequence forward
        curr_seq = np.roll(curr_seq, -1, axis=0)
        curr_seq[-1] = new_step
        
    return np.array(future_preds)

def inverse_transform_hr(scaled_data, scaler):
    """Convert scaled heart rate values back to original scale"""
    # Create dummy array with same shape as original data
    dummy = np.zeros((len(scaled_data), scaler.n_features_in_))
    dummy[:, 0] = scaled_data.flatten()  # Only heart rate column
    
    # Inverse transform
    unscaled = scaler.inverse_transform(dummy)
    
    # Return only heart rate column
    return unscaled[:, 0].reshape(-1, 1)
