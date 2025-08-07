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

# Functions for data generation
def generate_synthetic_data(days=100, noise_level=0.2):
    """Generate synthetic patient inflow and workload data"""
    date_range = pd.date_range(end=datetime.datetime.now(), periods=days)
    
    base_inflow = np.sin(np.linspace(0, 4*np.pi, days)) * 10 + 20
    weekday_effect = np.array([2 if i % 7 < 5 else -3 for i in range(days)])
    random_effect = np.random.normal(0, noise_level * 10, days)
    
    patient_inflow = base_inflow + weekday_effect + random_effect
    patient_inflow = np.maximum(patient_inflow, 5)
    
    workload = patient_inflow * 1.5 + np.random.normal(0, noise_level * 15, days)
    workload = np.maximum(workload, 10)
    
    staff = np.round(patient_inflow / 4) + 3
    staff = np.maximum(staff, 3)
    
    df = pd.DataFrame({
        'date': date_range,
        'patient_inflow': patient_inflow,
        'staff_on_duty': staff,
        'workload': workload
    })
    
    return df

# LSTM Model functions
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
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_workload_model(data, seq_length=7, epochs=50):
    """Train LSTM model on historical data"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['patient_inflow', 'staff_on_duty', 'workload']])
    
    X, y = create_sequences(scaled_data, seq_length)
    
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size, 2], y[train_size:, 2]
    
    model = build_lstm_model(seq_length, scaled_data.shape[1])
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=16,
        validation_data=(X_val, y_val),
        verbose=0
    )
    
    return model, scaler, history

def predict_workload(model, scaler, recent_data, seq_length=7, days_ahead=7):
    """Predict workload for upcoming days"""
    scaled_data = scaler.transform(recent_data[['patient_inflow', 'staff_on_duty', 'workload']])
    
    latest_seq = scaled_data[-seq_length:]
    latest_seq = latest_seq.reshape(1, seq_length, scaled_data.shape[1])
    
    predictions = []
    current_seq = latest_seq.copy()
    
    for _ in range(days_ahead):
        pred = model.predict(current_seq, verbose=0)[0][0]
        
        new_point = np.array([
            np.mean(scaled_data[-3:, 0]),
            np.mean(scaled_data[-3:, 1]),
            pred
        ]).reshape(1, 1, 3)
        
        dummy = np.zeros((1, 3))
        dummy[0, 2] = pred
        rescaled = scaler.inverse_transform(dummy)[0, 2]
        predictions.append(rescaled)
        
        current_seq = np.concatenate([current_seq[:, 1:, :], new_point], axis=1)
    
    return predictions

# Dynamic Scheduling
def dynamic_scheduling(predicted_workload, nurses, shifts_per_day=3, days=7):
    """Generate nurse schedule based on predicted workload"""
    num_nurses = len(nurses)
    schedule = np.zeros((num_nurses, days, shifts_per_day), dtype=int)
    
    nurses_needed = []
    for workload in predicted_workload:
        day_nurses = []
        for shift in range(shifts_per_day):
            shift_weight = 1.0 if shift == 0 else (0.8 if shift == 1 else 0.6)
            needed = max(2, int(np.ceil(workload * shift_weight / 10)))
            day_nurses.append(needed)
        nurses_needed.append(day_nurses)
    
    last_shift = [-3] * num_nurses
    nurse_shifts = [0] * num_nurses
    
    for day in range(days):
        for shift in range(shifts_per_day):
            nurse_priority = [(i, day - last_shift[i], nurse_shifts[i]) 
                             for i in range(num_nurses)]
            nurse_priority.sort(key=lambda x: (-x[1], x[2]))
            
            assigned = 0
            for nurse_idx, rest_time, _ in nurse_priority:
                if rest_time >= 2:
                    schedule[nurse_idx, day, shift] = 1
                    last_shift[nurse_idx] = day
                    nurse_shifts[nurse_idx] += 1
                    assigned += 1
                    
                    if assigned >= nurses_needed[day][shift]:
                        break
    
    schedule_df = pd.DataFrame()
    for day in range(days):
        for shift in range(shifts_per_day):
            shift_name = ['Morning', 'Afternoon', 'Night'][shift]
            col_name = f'Day {day+1} - {shift_name}'
            schedule_df[col_name] = [nurses[i] if schedule[i, day, shift] == 1 else "" for i in range(num_nurses)]
    
    schedule_df.index = [f"Slot {i+1}" for i in range(num_nurses)]
    
    return schedule_df, nurses_needed

# Task Prioritization
def prioritize_tasks(predicted_workload, day_index=0):
    """Prioritize nursing tasks based on predicted workload"""
    base_tasks = [
        "Medication administration",
        "Patient assessment",
        "Documentation",
        "Care coordination",
        "Patient education"
    ]
    
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
    
    workload_pct = min(100, max(0, predicted_workload[day_index] / 100 * 100))
    
    selected_tasks = base_tasks.copy()
    priority_scores = [95, 90, 85, 80, 75]
    
    for task, threshold in conditional_tasks.items():
        if workload_pct >= threshold:
            selected_tasks.append(task)
            priority_scores.append(threshold)
    
    tasks_with_priority = list(zip(selected_tasks, priority_scores))
    tasks_with_priority.sort(key=lambda x: -x[1])
    
    return tasks_with_priority

# Main run function
def run():
    st.header("🏥 Nurse Assistance System")
    st.write("Predict workload and generate real-time nurse schedules using LSTM and Dynamic Scheduling")

    # Initialize session state with unique keys to avoid conflicts
    if 'nurse_initialized' not in st.session_state:
        st.session_state.nurse_initialized = True
        st.session_state.nurse_model_trained = False
        st.session_state.nurse_predicted_workload = None
        st.session_state.nurse_schedule = None
        st.session_state.nurse_tasks = None
        st.session_state.nurse_nurses = []

    # Sidebar for settings
    with st.sidebar:
        st.subheader("📊 Nurse System Settings")
        
        # Data Settings
        st.subheader("Data Settings")
        data_option = st.radio("Data Source", ["Generate Synthetic Data", "Upload Data"], key="nurse_data_option")
        
        if data_option == "Generate Synthetic Data":
            days = st.slider("Days of Historical Data", 30, 365, 100, key="nurse_days")
            if st.button("Generate Data", key="nurse_generate_data"):
                with st.spinner("Generating synthetic data..."):
                    data = generate_synthetic_data(days)
                    st.session_state.nurse_data = data
                    st.success(f"Generated {days} days of data!")
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="nurse_upload")
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                if all(col in data.columns for col in ['date', 'patient_inflow', 'staff_on_duty', 'workload']):
                    st.session_state.nurse_data = data
                    st.success("Data loaded successfully!")
                else:
                    st.error("CSV must contain required columns")
        
        # Model Settings
        st.subheader("Model Settings")
        seq_length = st.slider("Sequence Length (days)", 3, 14, 7, key="nurse_seq_length")
        epochs = st.slider("Training Epochs", 10, 100, 50, key="nurse_epochs")
        
        if 'nurse_data' in st.session_state and st.button("Train Model", key="nurse_train_model"):
            with st.spinner("Training LSTM model..."):
                model, scaler, history = train_workload_model(st.session_state.nurse_data, seq_length, epochs)
                st.session_state.nurse_model = model
                st.session_state.nurse_scaler = scaler
                st.session_state.nurse_history = history.history
                st.session_state.nurse_model_trained = True
                st.success("Model trained successfully!")
        
        # Scheduling Settings
        st.subheader("Scheduling Settings")
        forecast_days = st.slider("Forecast Days", 1, 14, 7, key="nurse_forecast_days")
        num_nurses = st.slider("Number of Nurses", 5, 20, 10, key="nurse_num_nurses")
        
        # Generate nurse names
        if len(st.session_state.nurse_nurses) != num_nurses:
            first_names = ["Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte", "Amelia", 
                          "Harper", "Evelyn", "James", "Robert", "John", "Michael", "William"]
            last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", 
                         "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris"]
            st.session_state.nurse_nurses = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(num_nurses)]
        
        shifts_per_day = st.selectbox("Shifts per Day", [2, 3], index=1, key="nurse_shifts_per_day")
        
        if st.session_state.nurse_model_trained and st.button("Generate Schedule", key="nurse_generate_schedule"):
            with st.spinner("Generating schedule..."):
                recent_data = st.session_state.nurse_data.iloc[-seq_length:]
                predicted_workload = predict_workload(
                    st.session_state.nurse_model, 
                    st.session_state.nurse_scaler, 
                    recent_data, 
                    seq_length, 
                    forecast_days
                )
                schedule, nurses_needed = dynamic_scheduling(
                    predicted_workload, 
                    st.session_state.nurse_nurses, 
                    shifts_per_day, 
                    forecast_days
                )
                tasks = prioritize_tasks(predicted_workload)
                
                st.session_state.nurse_predicted_workload = predicted_workload
                st.session_state.nurse_schedule = schedule
                st.session_state.nurse_nurses_needed = nurses_needed
                st.session_state.nurse_tasks = tasks
                
                st.success("Schedule generated successfully!")

    # Main content
    if 'nurse_data' in st.session_state:
        st.subheader("📈 Historical Data")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Recent patient inflow and workload data")
            st.dataframe(st.session_state.nurse_data.tail(10))
        
        with col2:
            st.write("Patient Inflow and Workload Trends")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(st.session_state.nurse_data['patient_inflow'].values[-30:], label='Patient Inflow')
            ax.plot(st.session_state.nurse_data['workload'].values[-30:], label='Workload')
            ax.set_title('Last 30 Days Trends')
            ax.legend()
            st.pyplot(fig)

    if 'nurse_predicted_workload' in st.session_state and st.session_state.nurse_predicted_workload is not None:
        st.subheader("🔮 Workload Prediction")
        col1, col2 = st.columns(2)
        
        with col1:
            days = [f"Day {i+1}" for i in range(len(st.session_state.nurse_predicted_workload))]
            workload_df = pd.DataFrame({
                'Day': days,
                'Predicted Workload': [round(w, 2) for w in st.session_state.nurse_predicted_workload]
            })
            st.dataframe(workload_df)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(days, st.session_state.nurse_predicted_workload)
            ax.set_title('Predicted Workload')
            ax.set_ylabel('Workload Units')
            st.pyplot(fig)
        
        if 'nurse_nurses_needed' in st.session_state:
            st.write("👥 Nurses Needed per Shift")
            shifts = ['Morning', 'Afternoon', 'Night'][:len(st.session_state.nurse_nurses_needed[0])]
            nurses_data = {}
            for i, day in enumerate(days):
                nurses_data[day] = st.session_state.nurse_nurses_needed[i]
            nurses_df = pd.DataFrame(nurses_data, index=shifts)
            st.dataframe(nurses_df)

    if 'nurse_schedule' in st.session_state and st.session_state.nurse_schedule is not None:
        st.subheader("📅 Nurse Schedule")
        st.dataframe(st.session_state.nurse_schedule)
        
        nurse_shifts = {nurse: 0 for nurse in st.session_state.nurse_nurses}
        for col in st.session_state.nurse_schedule.columns:
            for nurse in st.session_state.nurse_schedule[col]:
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

    if 'nurse_tasks' in st.session_state and st.session_state.nurse_tasks is not None:
        st.subheader("📋 Task Prioritization")
        days = [f"Day {i+1}" for i in range(len(st.session_state.nurse_predicted_workload))]
        selected_day = st.selectbox("Select Day", days, key="nurse_day_select")
        day_index = int(selected_day.split()[1]) - 1
        
        tasks = prioritize_tasks(st.session_state.nurse_predicted_workload, day_index)
        task_df = pd.DataFrame(tasks, columns=['Task', 'Priority'])
        task_df['Priority'] = task_df['Priority'].apply(lambda x: f"{x}/100")
        
        st.write(f"Tasks prioritized for {selected_day}")
        st.dataframe(task_df)

# No need for if __name__ == "__main__" since this will be imported
