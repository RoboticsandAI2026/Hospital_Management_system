# Complete Nurse Assistance System for Person 2 Task
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime
import random

# Set page configuration
st.set_page_config(
    page_title="Nurse Assistance System",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state for persistent data
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.model_trained = False
    st.session_state.predicted_workload = None
    st.session_state.schedule = None
    st.session_state.tasks = None

# Title and description
st.title("🏥 Nurse Assistance System")
st.write("Predict workload and generate real-time nurse schedules using LSTM and Dynamic Scheduling")

# Functions for data generation
def generate_synthetic_data(days=100, noise_level=0.2):
    """Generate synthetic patient inflow and workload data"""
    date_range = pd.date_range(end=datetime.datetime.now(), periods=days)
    
    # Base patient inflow with weekly and random patterns
    base_inflow = np.sin(np.linspace(0, 4*np.pi, days)) * 10 + 20  # Seasonal pattern
    weekday_effect = np.array([2 if i % 7 < 5 else -3 for i in range(days)])  # Weekday vs weekend
    random_effect = np.random.normal(0, noise_level * 10, days)  # Random variation
    
    patient_inflow = base_inflow + weekday_effect + random_effect
    patient_inflow = np.maximum(patient_inflow, 5)  # Ensure at least 5 patients
    
    # Workload based on patient inflow plus some randomness
    workload = patient_inflow * 1.5 + np.random.normal(0, noise_level * 15, days)
    workload = np.maximum(workload, 10)  # Ensure at least 10 workload units
    
    # Staff on duty
    staff = np.round(patient_inflow / 4) + 3  # Base staffing ratio
    staff = np.maximum(staff, 3)  # Minimum 3 staff members
    
    # Convert to dataframe
    df = pd.DataFrame({
        'date': date_range,
        'patient_inflow': patient_inflow,
        'staff_on_duty': staff,
        'workload': workload
    })
    
    return df

# LSTM Model for workload prediction
def create_sequences(data, seq_length=7):
    """Create sequences for LSTM model"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def build_lstm_model(seq_length, n_features):
    """Build and compile LSTM model"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)  # Predict workload
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train LSTM model
def train_workload_model(data, seq_length=7, epochs=50):
    """Train LSTM model on historical data"""
    # Prepare data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['patient_inflow', 'staff_on_duty', 'workload']])
    
    # Create sequences
    X, y = create_sequences(scaled_data, seq_length)
    
    # Training-validation split
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size, 2], y[train_size:, 2]  # Index 2 corresponds to workload
    
    # Build and train model
    model = build_lstm_model(seq_length, scaled_data.shape[1])
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=16,
        validation_data=(X_val, y_val),
        verbose=0
    )
    
    return model, scaler, history

# Predict workload
def predict_workload(model, scaler, recent_data, seq_length=7, days_ahead=7):
    """Predict workload for upcoming days"""
    # Scale the data
    scaled_data = scaler.transform(recent_data[['patient_inflow', 'staff_on_duty', 'workload']])
    
    # Get the latest sequence
    latest_seq = scaled_data[-seq_length:]
    latest_seq = latest_seq.reshape(1, seq_length, scaled_data.shape[1])
    
    # Initial prediction
    predictions = []
    current_seq = latest_seq.copy()
    
    # Predict workload for next 'days_ahead' days
    for _ in range(days_ahead):
        # Predict next day
        pred = model.predict(current_seq, verbose=0)[0][0]
        
        # Create a new data point with predicted workload
        new_point = np.array([
            np.mean(scaled_data[-3:, 0]),  # patient_inflow
            np.mean(scaled_data[-3:, 1]),  # staff_on_duty
            pred  # predicted workload
        ]).reshape(1, 1, 3)
        
        # Save the prediction (need to rescale)
        dummy = np.zeros((1, 3))
        dummy[0, 2] = pred
        rescaled = scaler.inverse_transform(dummy)[0, 2]
        predictions.append(rescaled)
        
        # Update sequence for next prediction
        current_seq = np.concatenate([current_seq[:, 1:, :], new_point], axis=1)
    
    return predictions

# Dynamic Scheduling Algorithm
def dynamic_scheduling(predicted_workload, nurses, shifts_per_day=3, days=7):
    """Generate nurse schedule based on predicted workload"""
    num_nurses = len(nurses)
    
    # Create empty schedule matrix (nurses x days x shifts)
    schedule = np.zeros((num_nurses, days, shifts_per_day), dtype=int)
    
    # Calculate nurses needed per shift based on workload
    nurses_needed = []
    for workload in predicted_workload:
        # Simple formula: 1 nurse per 10 workload units, minimum 2 nurses
        day_nurses = []
        for shift in range(shifts_per_day):
            shift_weight = 1.0 if shift == 0 else (0.8 if shift == 1 else 0.6)  # Morning, afternoon, night
            needed = max(2, int(np.ceil(workload * shift_weight / 10)))
            day_nurses.append(needed)
        nurses_needed.append(day_nurses)
    
    # Calculate days since last shift for each nurse
    last_shift = [-3] * num_nurses  # Initialize with -3 to allow assignment on first day
    
    # Track shifts assigned to each nurse
    nurse_shifts = [0] * num_nurses
    
    # Assign shifts
    for day in range(days):
        for shift in range(shifts_per_day):
            # Sort nurses by days since last shift and total shifts assigned
            nurse_priority = [(i, day - last_shift[i], nurse_shifts[i]) 
                              for i in range(num_nurses)]
            nurse_priority.sort(key=lambda x: (-x[1], x[2]))  # Sort by rest time (desc) and shifts (asc)
            
            # Assign nurses until required number is reached
            assigned = 0
            for nurse_idx, rest_time, _ in nurse_priority:
                if rest_time >= 2:  # Ensure minimum rest period (at least 1 full day)
                    schedule[nurse_idx, day, shift] = 1
                    last_shift[nurse_idx] = day
                    nurse_shifts[nurse_idx] += 1
                    assigned += 1
                    
                    if assigned >= nurses_needed[day][shift]:
                        break
    
    # Convert to pandas DataFrame for better visualization
    schedule_df = pd.DataFrame()
    
    for day in range(days):
        for shift in range(shifts_per_day):
            shift_name = ['Morning', 'Afternoon', 'Night'][shift]
            col_name = f'Day {day+1} - {shift_name}'
            schedule_df[col_name] = [nurses[i] if schedule[i, day, shift] == 1 else "" for i in range(num_nurses)]
    
    schedule_df.index = [f"Slot {i+1}" for i in range(num_nurses)]
    
    return schedule_df, nurses_needed

# Task Prioritization Logic
def prioritize_tasks(predicted_workload, day_index=0):
    """Prioritize nursing tasks based on predicted workload"""
    # Base tasks that always need to be done
    base_tasks = [
        "Medication administration",
        "Patient assessment",
        "Documentation",
        "Care coordination",
        "Patient education"
    ]
    
    # Conditional tasks based on workload
    conditional_tasks = {
        "Critical care monitoring": 70,
        "Emergency response": 90,
        "Regular vital checks": 40,
        "Pain management": 50,
        "Wound care": 45,
        "Medical tests assistance": 55,
        "Patient mobility assistance": 35,
        "Family communication": 30,
        "Discharge planning": 25
    }
    
    # Calculate workload percentage (0-100 scale)
    workload_pct = min(100, max(0, predicted_workload[day_index] / 100 * 100))
    
    # Select tasks based on workload
    selected_tasks = base_tasks.copy()
    priority_scores = [95, 90, 85, 80, 75]  # Base task priorities
    
    for task, threshold in conditional_tasks.items():
        if workload_pct >= threshold:
            selected_tasks.append(task)
            # Higher workload tasks get higher priority
            priority_scores.append(threshold)
    
    # Combine tasks and priorities, then sort by priority
    tasks_with_priority = list(zip(selected_tasks, priority_scores))
    tasks_with_priority.sort(key=lambda x: -x[1])  # Sort by priority (descending)
    
    return tasks_with_priority

# UI Layout
with st.sidebar:
    st.header("📊 Settings")
    
    tab1, tab2 = st.tabs(["Data & Model", "Scheduling"])
    
    with tab1:
        st.subheader("Data Settings")
        data_option = st.radio("Data Source", ["Generate Synthetic Data", "Upload Data"])
        
        if data_option == "Generate Synthetic Data":
            days = st.slider("Days of Historical Data", 30, 365, 100)
            
            if st.button("Generate Data"):
                with st.spinner("Generating synthetic data..."):
                    data = generate_synthetic_data(days)
                    st.session_state.data = data
                    st.success(f"Generated {days} days of data!")
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                if all(col in data.columns for col in ['date', 'patient_inflow', 'staff_on_duty', 'workload']):
                    st.session_state.data = data
                    st.success("Data loaded successfully!")
                else:
                    st.error("CSV must contain required columns")
        
        st.subheader("Model Settings")
        seq_length = st.slider("Sequence Length (days)", 3, 14, 7)
        epochs = st.slider("Training Epochs", 10, 100, 50)
        
        if 'data' in st.session_state and st.button("Train Model"):
            with st.spinner("Training LSTM model..."):
                model, scaler, history = train_workload_model(st.session_state.data, seq_length, epochs)
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.history = history.history
                st.session_state.model_trained = True
                st.success("Model trained successfully!")
    
    with tab2:
        st.subheader("Scheduling Settings")
        forecast_days = st.slider("Forecast Days", 1, 14, 7)
        
        st.subheader("Nurse Information")
        num_nurses = st.slider("Number of Nurses", 5, 20, 10)
        
        # Generate nurse names if not already created
        if 'nurses' not in st.session_state or len(st.session_state.nurses) != num_nurses:
            first_names = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte", "Amelia", 
                          "Harper", "Evelyn", "James", "Robert", "John", "Michael", "William"]
            last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", 
                         "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris"]
            
            # Generate unique names
            st.session_state.nurses = []
            for i in range(num_nurses):
                first = random.choice(first_names)
                last = random.choice(last_names)
                st.session_state.nurses.append(f"{first} {last}")
        
        shifts_per_day = st.selectbox("Shifts per Day", [2, 3], 1)  # Default to 3 shifts
        
        if 'model_trained' in st.session_state and st.session_state.model_trained:
            if st.button("Generate Schedule"):
                with st.spinner("Generating schedule..."):
                    # Get recent data for prediction
                    recent_data = st.session_state.data.iloc[-seq_length:]
                    
                    # Predict workload
                    predicted_workload = predict_workload(
                        st.session_state.model, 
                        st.session_state.scaler, 
                        recent_data, 
                        seq_length, 
                        forecast_days
                    )
                    
                    # Generate schedule
                    schedule, nurses_needed = dynamic_scheduling(
                        predicted_workload, 
                        st.session_state.nurses, 
                        shifts_per_day, 
                        forecast_days
                    )
                    
                    # Prioritize tasks
                    tasks = prioritize_tasks(predicted_workload)
                    
                    # Save to session state
                    st.session_state.predicted_workload = predicted_workload
                    st.session_state.schedule = schedule
                    st.session_state.nurses_needed = nurses_needed
                    st.session_state.tasks = tasks
                    
                    st.success("Schedule generated successfully!")

# Main content area
if 'data' in st.session_state:
    st.subheader("📈 Historical Data")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Recent patient inflow and workload data")
        st.dataframe(st.session_state.data.tail(10))
    
    with col2:
        st.write("Patient Inflow and Workload Trends")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(st.session_state.data['patient_inflow'].values[-30:], label='Patient Inflow')
        ax.plot(st.session_state.data['workload'].values[-30:], label='Workload')
        ax.set_title('Last 30 Days Trends')
        ax.legend()
        st.pyplot(fig)

if 'predicted_workload' in st.session_state and st.session_state.predicted_workload is not None:
    st.subheader("🔮 Workload Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        # Show predicted workload
        days = [f"Day {i+1}" for i in range(len(st.session_state.predicted_workload))]
        workload_df = pd.DataFrame({
            'Day': days,
            'Predicted Workload': [round(w, 2) for w in st.session_state.predicted_workload]
        })
        st.dataframe(workload_df)
        
    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(days, st.session_state.predicted_workload)
        ax.set_title('Predicted Workload')
        ax.set_ylabel('Workload Units')
        st.pyplot(fig)
        
    # Nurses needed per shift
    if 'nurses_needed' in st.session_state:
        st.write("👥 Nurses Needed per Shift")
        
        # Prepare data for display
        shifts = ['Morning', 'Afternoon', 'Night'][:len(st.session_state.nurses_needed[0])]
        nurses_data = {}
        
        for i, day in enumerate(days):
            nurses_data[day] = st.session_state.nurses_needed[i]
        
        nurses_df = pd.DataFrame(nurses_data, index=shifts)
        st.dataframe(nurses_df)

if 'schedule' in st.session_state and st.session_state.schedule is not None:
    st.subheader("📅 Nurse Schedule")
    st.dataframe(st.session_state.schedule)
    
    # Show nurse workload distribution
    nurse_shifts = {}
    for nurse in st.session_state.nurses:
        nurse_shifts[nurse] = 0
    
    for col in st.session_state.schedule.columns:
        for nurse in st.session_state.schedule[col]:
            if nurse != "":
                nurse_shifts[nurse] += 1
    
    st.write("👩‍⚕️ Nurse Shift Distribution")
    shifts_df = pd.DataFrame({
        'Nurse': list(nurse_shifts.keys()),
        'Assigned Shifts': list(nurse_shifts.values())
    })
    shifts_df = shifts_df.sort_values('Assigned Shifts', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(shifts_df)
        
    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(shifts_df['Nurse'], shifts_df['Assigned Shifts'])
        ax.set_title('Shifts per Nurse')
        ax.set_ylabel('Number of Shifts')
        plt.xticks(rotation=45)
        fig.tight_layout()
        st.pyplot(fig)

if 'tasks' in st.session_state and st.session_state.tasks is not None:
    st.subheader("📋 Task Prioritization")
    
    # Let user select day to view
    if 'predicted_workload' in st.session_state:
        days = [f"Day {i+1}" for i in range(len(st.session_state.predicted_workload))]
        selected_day = st.selectbox("Select Day", days)
        day_index = int(selected_day.split()[1]) - 1
        
        # Get tasks for selected day
        tasks = prioritize_tasks(st.session_state.predicted_workload, day_index)
        
        # Show tasks
        task_df = pd.DataFrame(tasks, columns=['Task', 'Priority'])
        task_df['Priority'] = task_df['Priority'].apply(lambda x: f"{x}/100")
        
        st.write(f"Tasks prioritized for {selected_day}")
        st.dataframe(task_df)
