import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Import the backend modules
from anomaly import (
    load_model as load_anomaly_model, 
    create_sequences as create_anomaly_sequences, 
    detect_anomalies, 
    preprocess_data, 
    generate_demo_data,
    calculate_feature_contributions
)

from forecast import (
    load_forecast_model,
    create_sequences as create_forecast_sequences,
    predict_future,
    inverse_transform_hr
)

def run():
    # Add custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #4b778d;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #28b5b5;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .info-box {
            background-color: #f0f8ff;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 5px solid #4b778d;
        }
        .anomaly-high {
            color: #d62728;
            font-weight: bold;
        }
        .anomaly-medium {
            color: #ff7f0e;
            font-weight: bold;
        }
        .normal {
            color: #2ca02c;
        }
    </style>
    """, unsafe_allow_html=True)

    # App title and description
    st.markdown("<h1 class='main-header'>Patient Vitals Monitoring System</h1>", unsafe_allow_html=True)

    with st.expander("ℹ️ About this application", expanded=False):
        st.markdown("""
        <div class="info-box">
        <p>This application provides two main functionalities:</p>
        <ol>
            <li><b>Anomaly Detection:</b> Uses an LSTM Autoencoder model to detect anomalies in patient vital signs.</li>
            <li><b>Forecasting:</b> Uses a multivariate LSTM model to predict future heart rate values based on historical data.</li>
        </ol>
        
        <p><b>How to use:</b></p>
        <ol>
            <li>Upload patient vitals data in CSV format</li>
            <li>Select the analysis type (Anomaly Detection or Forecasting)</li>
            <li>Configure model parameters</li>
            <li>Run the analysis to get results</li>
            <li>Review the results with interactive visualizations</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar for configuration
    st.sidebar.title("Vitals Monitoring Configuration")

    # Cache model loading
    @st.cache_resource
    def get_cached_anomaly_model(model_path, seq_len, features):
        return load_anomaly_model(model_path, seq_len, features)

    @st.cache_resource
    def get_cached_forecast_model(model_path):
        return load_forecast_model(model_path)

    # Sidebar settings
    st.sidebar.markdown("<div class='sub-header'>1. Data Input</div>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Upload patient vitals CSV file", type=["csv"], key="vitals_file_uploader")

    # Demo data option
    use_demo_data = st.sidebar.checkbox("Use demo data", value=True if not uploaded_file else False, key="vitals_use_demo")

    # Analysis type selection
    analysis_type = st.sidebar.radio("Select Analysis Type", 
                                    ["Anomaly Detection", "Forecasting"],
                                    index=0, key="vitals_analysis_type")

    # Common parameters
    st.sidebar.markdown("<div class='sub-header'>2. Model Parameters</div>", unsafe_allow_html=True)
    seq_len = st.sidebar.slider("Sequence Length", min_value=5, max_value=30, value=10, 
                              help="Number of time steps in each sequence", key="vitals_seq_len")

    if analysis_type == "Anomaly Detection":
        threshold_factor = st.sidebar.slider("Anomaly Threshold Factor", min_value=1.0, max_value=5.0, value=2.0, step=0.1,
                                           help="Higher values will detect fewer but more significant anomalies", key="vitals_threshold")
        model_path = st.sidebar.text_input("Anomaly Model Path", 
                                          value="anomaly_detector_model_from_scratch (1).h5",
                                          help="Path to the trained anomaly detection model", key="vitals_anomaly_model_path")
    else:
        forecast_steps = st.sidebar.slider("Forecast Steps", min_value=1, max_value=100, value=30,
                                         help="Number of future time steps to predict", key="vitals_forecast_steps")
        model_path = st.sidebar.text_input("Forecast Model Path", 
                                          value="best_model.h5",
                                          help="Path to the trained forecasting model", key="vitals_forecast_model_path")

    # Main content
    if uploaded_file or use_demo_data:
        # Load data
        try:
            if use_demo_data:
                st.info("Using demo data (sample patient vitals)")
                data = generate_demo_data()
            else:
                data = pd.read_csv(uploaded_file)
            
            st.markdown("<div class='sub-header'>Data Preview</div>", unsafe_allow_html=True)
            st.dataframe(data.head())
            
            # Check if timestamp exists and convert to datetime
            if 'timestamp' in data.columns:
                if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            if analysis_type == "Anomaly Detection":
                # Data preprocessing for anomaly detection
                with st.spinner("Preprocessing data for anomaly detection..."):
                    scaled_data, numeric_data, timestamps, scaler = preprocess_data(data)
                    
                    # Create sequences
                    sequences = create_anomaly_sequences(scaled_data, seq_len)
                    
                    st.success(f"Created {len(sequences)} sequences for analysis")
                
                # Load anomaly detection model
                with st.spinner("Loading anomaly detection model..."):
                    model, model_loaded = get_cached_anomaly_model(model_path, seq_len, sequences.shape[2])
                    if model_loaded:
                        st.success("Anomaly detection model loaded successfully")
                
                # Run anomaly detection button
                if st.button("Run Anomaly Detection", key="vitals_run_anomaly"):
                    with st.spinner("Detecting anomalies..."):
                        start_time = time.time()
                        
                        # Detect anomalies
                        mse, threshold, anomalies, reconstructions = detect_anomalies(
                            model, sequences, threshold_factor)
                        
                        execution_time = time.time() - start_time
                        
                        # Get anomaly indices
                        anomaly_indices = np.where(anomalies)[0]
                        
                        # Map to original timestamps
                        anomaly_original_indices = [i + seq_len for i in anomaly_indices]
                        anomaly_timestamps = timestamps.iloc[anomaly_original_indices] if len(anomaly_original_indices) > 0 else []
                        
                        st.success(f"Analysis completed in {execution_time:.2f} seconds")
                    
                    # Display results
                    st.markdown("<div class='sub-header'>Anomaly Detection Results</div>", unsafe_allow_html=True)
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", len(data))
                    with col2:
                        st.metric("Sequences Analyzed", len(sequences))
                    with col3:
                        anomaly_percentage = len(anomaly_indices) / len(sequences) * 100 if len(sequences) > 0 else 0
                        st.metric("Anomalies Detected", f"{len(anomaly_indices)} ({anomaly_percentage:.2f}%)")
                    with col4:
                        st.metric("Anomaly Threshold", f"{threshold:.4f}")
                    
                    # Plot reconstruction error
                    st.markdown("<div class='sub-header'>Reconstruction Error Analysis</div>", unsafe_allow_html=True)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(len(mse))),
                        y=mse,
                        mode='lines',
                        name='Reconstruction Error',
                        line=dict(color='blue', width=1)
                    ))
                    fig.add_trace(go.Scatter(
                        x=list(range(len(mse))),
                        y=[threshold] * len(mse),
                        mode='lines',
                        name=f'Threshold ({threshold:.4f})',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    if len(anomaly_indices) > 0:
                        fig.add_trace(go.Scatter(
                            x=anomaly_indices,
                            y=mse[anomaly_indices],
                            mode='markers',
                            name='Anomalies',
                            marker=dict(color='red', size=8, symbol='circle')
                        ))
                    
                    fig.update_layout(
                        title='Reconstruction Error Over Time',
                        xaxis_title='Sequence Index',
                        yaxis_title='Mean Squared Error',
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Anomaly details
                    if len(anomaly_indices) > 0:
                        st.markdown("<div class='sub-header'>Anomaly Details</div>", unsafe_allow_html=True)
                        
                        # Create a dataframe for anomalies
                        anomaly_data = []
                        for i, idx in enumerate(anomaly_indices):
                            orig_idx = anomaly_original_indices[i]
                            if orig_idx < len(timestamps):
                                ts = timestamps.iloc[orig_idx]
                                error = mse[idx]
                                severity = "High" if error > threshold * 1.5 else "Medium"
                                
                                # Get the feature contributions for this anomaly
                                feature_errors = np.mean(np.square(sequences[idx] - reconstructions[idx]), axis=0)
                                feature_names = numeric_data.columns
                                top_features = sorted(zip(feature_names, feature_errors), key=lambda x: x[1], reverse=True)[:3]
                                top_feature_names = ", ".join([f"{name}" for name, _ in top_features])
                                
                                anomaly_data.append({
                                    "ID": i+1,
                                    "Timestamp": ts,
                                    "Original Index": orig_idx,
                                    "Error": error,
                                    "Severity": severity,
                                    "Top Contributing Features": top_feature_names
                                })
                        
                        anomaly_df = pd.DataFrame(anomaly_data)
                        
                        # Apply styling
                        def color_severity(val):
                            if val == "High":
                                return 'color: #d62728; font-weight: bold'
                            elif val == "Medium":
                                return 'color: #ff7f0e; font-weight: bold'
                            return ''
                        
                        # Show the styled dataframe
                        st.dataframe(anomaly_df.style.applymap(color_severity, subset=['Severity']))
                        
                        # Feature contribution analysis
                        st.markdown("<div class='sub-header'>Feature Contribution to Anomalies</div>", unsafe_allow_html=True)
                        
                        # Calculate average contribution per feature
                        sorted_features, feature_contribution = calculate_feature_contributions(
                            sequences, reconstructions, anomaly_indices, numeric_data)
                        
                        # Create a bar chart
                        contrib_df = pd.DataFrame(sorted_features, columns=['Feature', 'Contribution'])
                        fig = px.bar(contrib_df, x='Feature', y='Contribution',
                                     title='Average Feature Contribution to Anomalies',
                                     color='Contribution',
                                     color_continuous_scale=px.colors.sequential.Reds)
                        fig.update_layout(xaxis_tickangle=-45, height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.info("No anomalies detected with the current threshold. Try lowering the threshold factor.")
            
            else:  # Forecasting
                # Data preprocessing for forecasting
                with st.spinner("Preprocessing data for forecasting..."):
                    # Select relevant features
                    features = [
                        "heart_rate",
                        "blood_pressure_systolic",
                        "blood_pressure_diastolic",
                        "temperature",
                        "spo2"
                    ]
                    
                    # Keep only relevant features and drop any rows with missing values
                    forecast_data = data[features].dropna().copy()
                    
                    # Initialize a new MinMaxScaler for forecasting
                    forecast_scaler = MinMaxScaler()
                    scaled_data = forecast_scaler.fit_transform(forecast_data)
                    
                    # Create sequences
                    sequences, _ = create_forecast_sequences(scaled_data, seq_len=seq_len)
                    
                    st.success(f"Created {len(sequences)} sequences for forecasting")
                
                # Load forecasting model
                with st.spinner("Loading forecasting model..."):
                    model = get_cached_forecast_model(model_path)
                    st.success("Forecasting model loaded successfully")
                
                # Run forecasting button
                if st.button("Run Forecasting", key="vitals_run_forecast"):
                    with st.spinner("Generating forecasts..."):
                        start_time = time.time()
                        
                        # Get the most recent sequence
                        last_sequence = sequences[-1]
                        
                        # Predict future values
                        future_preds = predict_future(model, last_sequence, n_steps=forecast_steps)
                        
                        # Convert to original scale using the forecast scaler
                        future_hr = inverse_transform_hr(future_preds.reshape(-1, 1), forecast_scaler)
                        
                        execution_time = time.time() - start_time
                        st.success(f"Forecasting completed in {execution_time:.2f} seconds")
                    
                    # Display results
                    st.markdown("<div class='sub-header'>Heart Rate Forecasting Results</div>", unsafe_allow_html=True)
                    
                    # Create time index for future predictions
                    last_timestamp = data['timestamp'].iloc[-1]
                    freq = pd.infer_freq(data['timestamp'])
                    future_dates = pd.date_range(start=last_timestamp, periods=forecast_steps+1, freq=freq)[1:]
                    
                    # Create plot
                    fig = go.Figure()
                    
                    # Plot historical data
                    fig.add_trace(go.Scatter(
                        x=data['timestamp'],
                        y=data['heart_rate'],
                        mode='lines',
                        name='Historical Heart Rate',
                        line=dict(color='blue')
                    ))
                    
                    # Plot forecast
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_hr.flatten(),
                        mode='lines+markers',
                        name='Forecasted Heart Rate',
                        line=dict(color='red', dash='dot')
                    ))
                    
                    # Add vertical line at forecast start
                    fig.add_vline(x=last_timestamp, line_width=2, line_dash="dash", line_color="green")
                    
                    fig.update_layout(
                        title='Heart Rate Forecast',
                        xaxis_title='Time',
                        yaxis_title='Heart Rate (bpm)',
                        height=600,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show forecasted values in a table
                    st.markdown("<div class='sub-header'>Forecasted Values</div>", unsafe_allow_html=True)
                    forecast_df = pd.DataFrame({
                        'Timestamp': future_dates,
                        'Forecasted Heart Rate': future_hr.flatten()
                    })
                    st.dataframe(forecast_df)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

    else:
        # Initial state when no file is uploaded
        st.info("Please upload patient vitals data in CSV format or use the demo data to get started.")
        
        # Display sample data format
        st.markdown("<div class='sub-header'>Expected Data Format</div>", unsafe_allow_html=True)
        st.markdown("""
        Your CSV file should contain patient vitals with columns like:
        - timestamp
        - heart_rate
        - blood_pressure_systolic
        - blood_pressure_diastolic
        - temperature
        - spo2
        - respiratory_rate
        - etc.
        
        Check the "Use demo data" option to see an example of the expected format.
        """)

    # Add footer
    st.markdown("""
    ---
    <div style="text-align: center; color: #888;">
    Patient Vitals Monitoring System | Created with Streamlit
    </div>
    """, unsafe_allow_html=True)
