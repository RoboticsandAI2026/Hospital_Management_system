
import streamlit as st
import app_karthi
import app_puji2
import app_sus

st.set_page_config(page_title="My Dashboard", layout="wide")

# Sidebar for navigation with logo
st.sidebar.image("SASTRA Hospitals.jpg", use_container_width=True)  # Updated parameter
st.sidebar.title("Application")
app_choice = st.sidebar.radio("Go to", ["Management System", "Prescription Generator", "Patient Monitoring", "Nurse Assistance"])

# Run the selected app
if app_choice == "Patient Monitoring":
    app_karthi.run()
elif app_choice == "Management System":
    app_sus.run()
elif app_choice == "Nurse Assistance":
    app_puji2.run()

