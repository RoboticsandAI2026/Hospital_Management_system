import streamlit as st
import pandas as pd
from pathlib import Path
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import datetime
import re
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Configuration ---
FEEDBACK_MODEL_PATH = "tiny_gpt_feedback.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DB_PATH = "data/hospital.db"
LOGO_PATH = "logo.jpg"  # Placeholder path for hospital logo

# --- Hyperparameters ---
feedback_block_size = 64
feedback_embedding_dim = 128
feedback_n_heads = 4
feedback_n_layers = 2
feedback_dropout = 0.1

# Vocabulary
feedback_vocab = [
    ' ', "'", ',', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    ':', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

def get_vocab(model_type):
    if model_type != "feedback":
        raise ValueError("Only feedback model is supported")
    vocab = feedback_vocab
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    encode = lambda s: [stoi[c] for c in s.lower() if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])
    return vocab, encode, decode

def init_db():
    Path("data").mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("PRAGMA foreign_keys = ON")
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            dob TEXT NOT NULL,
            gender TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            address TEXT,
            created_at TEXT NOT NULL
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            department TEXT NOT NULL,
            doctor TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            items TEXT NOT NULL,
            total REAL NOT NULL,
            covered REAL NOT NULL,
            payable REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

# --- Database Operations ---
def add_patient(first_name, last_name, dob, gender, email, phone, address):
    patient_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO patients (patient_id, first_name, last_name, dob, gender, email, phone, address, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (patient_id, first_name, last_name, dob, gender, email, phone, address, created_at))
    conn.commit()
    conn.close()
    return patient_id

def get_patient(patient_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,))
    patient = c.fetchone()
    conn.close()
    if patient:
        return {
            "patient_id": patient[0],
            "first_name": patient[1],
            "last_name": patient[2],
            "dob": patient[3],
            "gender": patient[4],
            "email": patient[5],
            "phone": patient[6],
            "address": patient[7],
            "created_at": patient[8]
        }
    return None

def search_patients(search_term):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    search_term = f"%{search_term}%"
    c.execute('''
        SELECT patient_id, first_name, last_name, dob, gender
        FROM patients
        WHERE first_name LIKE ? OR last_name LIKE ? OR patient_id LIKE ?
    ''', (search_term, search_term, search_term))
    patients = c.fetchall()
    conn.close()
    return [
        {
            "patient_id": p[0],
            "first_name": p[1],
            "last_name": p[2],
            "dob": p[3],
            "gender": p[4]
        } for p in patients
    ]

def get_patient_records(patient_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get appointments
    c.execute('SELECT * FROM appointments WHERE patient_id = ?', (patient_id,))
    appointments = c.fetchall()
    
    # Get invoices
    c.execute('SELECT * FROM invoices WHERE patient_id = ?', (patient_id,))
    invoices = c.fetchall()
    
    conn.close()
    
    return {
        "appointments": [
            {
                "id": a[0],
                "patient_id": a[1],
                "department": a[2],
                "doctor": a[3],
                "date": a[4],
                "time": a[5]
            } for a in appointments
        ],
        "invoices": [
            {
                "id": i[0],
                "patient_id": i[1],
                "items": json.loads(i[2]),
                "total": i[3],
                "covered": i[4],
                "payable": i[5],
                "created_at": i[6]
            } for i in invoices
        ]
    }

# --- Model Components ---
class Head(nn.Module):
    def __init__(self, embedding_dim, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(torch.tril(torch.ones(T, T, device=DEVICE)) == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return self.dropout(wei @ v)

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList([Head(embedding_dim, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.ffwd = FeedForward(embedding_dim, dropout)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TinyGPTFeedback(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, feedback_embedding_dim)
        self.position_embedding = nn.Embedding(feedback_block_size, feedback_embedding_dim)
        self.blocks = nn.Sequential(*[Block(feedback_embedding_dim, feedback_n_heads, feedback_dropout) 
                                    for _ in range(feedback_n_layers)])
        self.ln_final = nn.LayerNorm(feedback_embedding_dim)
        self.fc_out = nn.Linear(feedback_embedding_dim, vocab_size)
        self.dropout = nn.Dropout(feedback_dropout)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(self.dropout(x))
        logits = self.fc_out(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens=300, temperature=0.7):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -feedback_block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

@st.cache_resource
def load_model(model_path):
    if not Path(model_path).exists():
        st.error(f"Model file not found at {model_path}. Please ensure the file exists.")
        return None, None, None

    try:
        vocab, encode, decode = get_vocab("feedback")
        model = TinyGPTFeedback(len(vocab)).to(DEVICE)
        state_dict = torch.load(model_path, map_location=DEVICE)
        
        # Check for state dictionary compatibility
        model_state_dict = model.state_dict()
        missing_keys = set(model_state_dict.keys()) - set(state_dict.keys())
        unexpected_keys = set(state_dict.keys()) - set(model_state_dict.keys())
        
        if missing_keys or unexpected_keys:
            st.warning(f"State dictionary mismatch detected.\nMissing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}\nAttempting to load with strict=False.")
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        st.success(f"✅ Loaded model from {model_path}")
        return model, encode, decode
    except Exception as e:
        st.error(f"❌ Model load failed: {str(e)}. Please verify the model file is not corrupted and matches the expected architecture.")
        return None, None, None

# --- Text Processing ---
def format_feedback(text):
    feedback_points = []
    current_category = "General Feedback"
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        if line.endswith("Feedback:") and len(line.split()) <= 3:
            current_category = line.replace(":", "")
            continue
        
        if line.startswith("-"):
            feedback_points.append(f"**{current_category}**\n{line}")
        elif ":" in line:
            question, answer = [part.strip() for part in line.split(":", 1)]
            feedback_points.append(f"**{current_category}**\n- {question}: {answer.split('.')[0]}")

    return "\n\n".join(feedback_points[:5])

# --- Invoice PDF Generator ---
def generate_invoice_pdf(patient_name, selected_costs, total, covered, payable):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(name='Header', fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name='SubHeader', fontSize=12, leading=14, alignment=TA_LEFT, spaceAfter=8))
    styles.add(ParagraphStyle(name='Item', fontSize=10, leading=12, alignment=TA_LEFT, spaceAfter=4))
    styles.add(ParagraphStyle(name='Total', fontSize=10, leading=12, alignment=TA_LEFT, spaceAfter=4, fontName='Helvetica-Bold'))
    
    # Content
    content = []
    
    # Add logo if it exists
    if Path(LOGO_PATH).exists():
        logo = Image(LOGO_PATH, width=10*cm, height=6*cm)
        logo.hAlign = 'CENTER'
        content.append(logo)
        content.append(Spacer(1, 0.5*cm))
    else:
        content.append(Paragraph("Hospital Logo Placeholder", styles['Header']))
        content.append(Spacer(1, 0.5*cm))
        st.warning(f"Logo file not found at {LOGO_PATH}. Please place your hospital logo (PNG/JPG) at this path.")
    
    # Header
    content.append(Paragraph("Hospital Management System", styles['Header']))
    content.append(Paragraph("Invoice", styles['Header']))
    content.append(Spacer(1, 0.5*cm))
    
    # Patient Info
    content.append(Paragraph(f"Patient: {patient_name}", styles['SubHeader']))
    content.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", styles['SubHeader']))
    content.append(Spacer(1, 0.5*cm))
    
    # Itemized List
    content.append(Paragraph("Itemized Charges", styles['SubHeader']))
    content.append(Spacer(1, 0.2*cm))
    
    # Table-like structure for items
    for item, cost in selected_costs.items():
        content.append(Paragraph(f"{item}: INR {cost:.2f}", styles['Item']))
    
    content.append(Spacer(1, 0.5*cm))
    
    # Totals
    content.append(Paragraph(f"Total Charges: INR {total:.2f}", styles['Total']))
    content.append(Paragraph(f"Insurance Covered: -INR {covered:.2f}", styles['Total']))
    content.append(Paragraph(f"Amount Payable: INR {payable:.2f}", styles['Total']))
    
    # Build PDF
    doc.build(content)
    buffer.seek(0)
    return buffer

# --- Patient Management ---
def patient_management():
    st.title("👤 Patient Management")
    st.caption("Register, search, and view patient records")

    tabs = st.tabs(["Register Patient", "Search Patients", "View Patient Record"])

    # Enhanced Patient Registration Page
    with tabs[0]:
        st.subheader("Register New Patient")
        with st.form("patient_registration_form"):
            col1, col2 = st.columns(2)
            with col1:
                first_name = st.text_input("First Name *", placeholder="John")
                dob = st.date_input("Date of Birth *", min_value=datetime(1900, 1, 1), max_value=datetime.today())
                email = st.text_input("Email", placeholder="john.doe@example.com")
            with col2:
                last_name = st.text_input("Last Name *", placeholder="Doe")
                gender = st.selectbox("Gender *", ["Male", "Female", "Other"])
                phone = st.text_input("Phone", placeholder="+91 1234567890")
            
            address = st.text_area("Address", placeholder="123 Main St, City, Country")
            submit = st.form_submit_button("📝 Register Patient", type="primary")

            if submit:
                if not (first_name and last_name and dob and gender):
                    st.warning("Please fill all required fields (*)")
                elif not re.match(r"^[a-zA-Z\s]+$", first_name) or not re.match(r"^[a-zA-Z\s]+$", last_name):
                    st.warning("Names should only contain letters and spaces")
                elif email and not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                    st.warning("Invalid email format")
                elif phone and not re.match(r"^\+?\d{10,12}$", phone):
                    st.warning("Invalid phone number format")
                else:
                    try:
                        patient_id = add_patient(
                            first_name=first_name,
                            last_name=last_name,
                            dob=dob.strftime("%Y-%m-%d"),
                            gender=gender,
                            email=email,
                            phone=phone,
                            address=address
                        )
                        st.success(f"✅ Patient {first_name} {last_name} registered with ID: {patient_id}")
                        st.session_state['selected_patient_id'] = patient_id
                    except Exception as e:
                        st.error(f"❌ Failed to register patient: {str(e)}")

    # Patient Lookup Component
    with tabs[1]:
        st.subheader("Search Patients")
        search_term = st.text_input("Search by Name or ID", placeholder="Enter name or patient ID")
        if search_term:
            patients = search_patients(search_term)
            if patients:
                df = pd.DataFrame(patients, columns=["patient_id", "first_name", "last_name", "dob", "gender"])
                st.dataframe(df, use_container_width=True)
                selected_index = st.number_input("Select patient index", min_value=0, max_value=len(patients)-1, step=1)
                if st.button("View Record"):
                    st.session_state['selected_patient_id'] = patients[selected_index]["patient_id"]
                    st.rerun()
            else:
                st.info("No patients found matching the search term.")

    # Comprehensive Patient Record View
    with tabs[2]:
        st.subheader("Patient Record")
        patient_id = st.text_input("Enter Patient ID", value=st.session_state.get('selected_patient_id', ''))
        if patient_id:
            patient = get_patient(patient_id)
            if patient:
                st.markdown(f"**Name**: {patient['first_name']} {patient['last_name']}")
                st.markdown(f"**ID**: {patient['patient_id']}")
                st.markdown(f"**DOB**: {patient['dob']}")
                st.markdown(f"**Gender**: {patient['gender']}")
                st.markdown(f"**Email**: {patient['email'] or 'N/A'}")
                st.markdown(f"**Phone**: {patient['phone'] or 'N/A'}")
                st.markdown(f"**Address**: {patient['address'] or 'N/A'}")
                st.markdown(f"**Registered**: {patient['created_at']}")

                records = get_patient_records(patient_id)
                
                with st.expander("Appointments"):
                    if records["appointments"]:
                        df = pd.DataFrame(records["appointments"])
                        st.dataframe(df.drop(columns=["patient_id"]), use_container_width=True)
                    else:
                        st.info("No appointments found.")

                with st.expander("Invoices"):
                    if records["invoices"]:
                        for i in records["invoices"]:
                            st.markdown(f"**Invoice #{i['id']}** (Created: {i['created_at']})")
                            for item, cost in i["items"].items():
                                st.markdown(f"- {item}: ₹{cost}")
                            st.markdown(f"- **Total**: ₹{i['total']}")
                            st.markdown(f"- **Covered**: ₹{i['covered']}")
                            st.markdown(f"- **Payable**: ₹{i['payable']}")
                            st.markdown("---")
                    else:
                        st.info("No invoices found.")
            else:
                st.error("Patient ID not found.")

# --- Feedback Generator ---
def feedback_generator(model, encode, decode):
    st.title("🏥 Patient Feedback Generator")
    st.markdown("Generate patient feedback using AI based on your hospital's HIS data")
    
    with st.sidebar.expander("⚙️ Feedback Settings", expanded=True):
        temperature = st.slider("Creativity", 0.1, 1.0, 0.7)
        max_length = st.slider("Max Length", 100, 500, 300, step=50)
        
        if st.button("Clear Cache"):
            st.cache_resource.clear()
            st.success("Cache cleared!")
    
    prompt_templates = [
        "Patient feedback about:",
        "Summary of patient experience:",
        "The patient reported that:",
        "Key feedback points:",
        "Cleanliness feedback:",
        "Staff behavior feedback:",
        "Doctor communication:",
        "Billing experience:",
        "Discharge process:"
    ]
    
    selected_template = st.selectbox("Choose a template:", prompt_templates)
    custom_prompt = st.text_input("Or enter your own prompt:", value=selected_template)
    
    if st.button("Generate Feedback", type="primary"):
        if not custom_prompt:
            st.warning("Please enter a prompt")
            return
            
        with st.spinner("Generating feedback..."):
            try:
                encoded = encode(custom_prompt)
                if not encoded:
                    st.error("Prompt contains no valid characters")
                    return
                    
                input_tensor = torch.tensor(encoded, dtype=torch.long, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    output = model.generate(input_tensor, max_new_tokens=max_length, temperature=temperature)
                
                generated_text = decode(output[0].tolist())
                formatted_feedback = format_feedback(generated_text)
                
                st.subheader("Generated Feedback")
                st.markdown(formatted_feedback)
                
                with st.expander("View raw output"):
                    st.text(generated_text)
                    
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")

# --- Appointment Scheduling ---
def appointment_scheduling():
    st.title("📅 Appointment Scheduling")
    
    with st.sidebar.expander("⚙️ Appointment Settings", expanded=True):
        st.info("Configure appointment settings here")
    
    st.header("Schedule New Appointment")
    patient_id = st.text_input("Patient ID *", placeholder="Enter patient ID")
    patient = get_patient(patient_id) if patient_id else None
    
    if patient_id and not patient:
        st.warning("Patient ID not found. Please register the patient first.")
        return
    
    # Initialize session state for department and doctor
    if 'selected_department' not in st.session_state:
        st.session_state.selected_department = "General Medicine"
    if 'selected_doctor' not in st.session_state:
        st.session_state.selected_doctor = None
    
    # Department selection
    department = st.selectbox(
        "Department:",
        ["General Medicine", "Cardiology", "Orthopedics", "Pediatrics", "Neurology"],
        key="department_select",
        on_change=lambda: st.session_state.update({"selected_doctor": None, "selected_time": None})
    )
    
    # Update session state when department changes
    if department != st.session_state.selected_department:
        st.session_state.selected_department = department
        st.session_state.selected_doctor = None
        st.session_state.selected_time = None
    
    # Doctor list based on department
    doctors_by_department = {
        "General Medicine": ["Dr. Smith (9 AM - 12 PM)", "Dr. Arjun (2 PM - 5 PM)", "Dr. Kiran (10 AM - 1 PM)"],
        "Cardiology": ["Dr. Rakesh (10 AM - 1 PM)", "Dr. Meera (1 PM - 4 PM)", "Dr. Arvind (3 PM - 6 PM)"],
        "Orthopedics": ["Dr. Leena (11 AM - 2 PM)", "Dr. Rajiv (3 PM - 6 PM)", "Dr. Divya (9 AM - 12 PM)"],
        "Pediatrics": ["Dr. Fatima (9 AM - 11 AM)", "Dr. Sunil (1 PM - 3 PM)", "Dr. Neeraj (3 PM - 5 PM)"],
        "Neurology": ["Dr. Nisha (1 PM - 4 PM)", "Dr. Vikram (10 AM - 12 PM)", "Dr. Sneha (2 PM - 5 PM)"]
    }
    
    # Doctor selection
    doctor = st.selectbox(
        "Preferred Doctor:",
        doctors_by_department[department],
        key="doctor_select",
        index=0 if not st.session_state.get("selected_doctor") else doctors_by_department[department].index(st.session_state.selected_doctor) if st.session_state.selected_doctor in doctors_by_department[department] else 0,
        on_change=lambda: st.session_state.update({"selected_time": None})
    )
    
    # Update selected doctor in session state
    if doctor != st.session_state.selected_doctor:
        st.session_state.selected_doctor = doctor
        st.session_state.selected_time = None
    
    # Generate time slots based on doctor's availability
    def generate_time_slots(doctor):
        match = re.search(r"\((\d+\s*(?:AM|PM))\s*-\s*(\d+\s*(?:AM|PM))\)", doctor)
        if not match:
            return []
        
        start_str, end_str = match.groups()
        
        try:
            start_time = datetime.strptime(start_str, "%I %p")
            end_time = datetime.strptime(end_str, "%I %p")
            
            if end_time.hour < start_time.hour:
                end_time = end_time.replace(hour=end_time.hour + 12)
            
            time_slots = []
            current_time = start_time
            while current_time < end_time:
                time_slots.append(current_time.strftime("%I:%M %p"))
                current_time += timedelta(minutes=30)
            
            return time_slots
        except ValueError:
            return []
    
    # Time selection
    time_slots = generate_time_slots(doctor)
    if not time_slots:
        st.warning("No valid time slots available for this doctor. Please select another doctor.")
        return
    
    time = st.selectbox(
        "Select Time:",
        time_slots,
        key="time_select",
        index=0 if not st.session_state.get("selected_time") else time_slots.index(st.session_state.selected_time) if st.session_state.selected_time in time_slots else 0
    )
    
    # Update selected time in session state
    st.session_state.selected_time = time
    
    with st.form("appointment_form"):
        st.markdown(f"**Patient**: {patient['first_name']} {patient['last_name']}" if patient else "**Patient**: Not selected")
        date = st.date_input("Select Appointment Date:", min_value=datetime.today())
        
        submitted = st.form_submit_button("📆 Confirm Appointment")
        if submitted:
            if not patient_id:
                st.warning("Please enter a valid Patient ID")
            else:
                try:
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute('''
                        INSERT INTO appointments (patient_id, department, doctor, date, time)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (patient_id, department, doctor, date.strftime("%Y-%m-%d"), time))
                    conn.commit()
                    conn.close()
                    st.success(f"✅ Appointment confirmed for {patient['first_name']} {patient['last_name']} with {doctor} on {date} at {time}.")
                except Exception as e:
                    st.error(f"❌ Failed to save appointment: {str(e)}")

    st.header("📋 Manage Appointments")
    if patient_id and patient:
        records = get_patient_records(patient_id)
        if records["appointments"]:
            df = pd.DataFrame(records["appointments"])
            df["DateTime"] = pd.to_datetime(df["date"] + " " + df["time"])
            now = datetime.now()
            df["Status"] = df["DateTime"].apply(lambda dt: "🟢 Upcoming" if dt > now else "🔴 Expired")
            st.dataframe(df.drop(columns=["patient_id", "DateTime"]), use_container_width=True)
            
            selected_index = st.number_input(
                "Enter the index of the appointment to manage:",
                min_value=0,
                max_value=len(df) - 1,
                step=1
            )
            
            with st.form("manage_form"):
                new_date = st.date_input("New date:", value=datetime.strptime(df.iloc[selected_index]["date"], "%Y-%m-%d"))
                new_time = st.selectbox(
                    "New time:",
                    time_slots,
                    index=time_slots.index(df.iloc[selected_index]["time"]) if df.iloc[selected_index]["time"] in time_slots else 0
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    submit_reschedule = st.form_submit_button("🔁 Reschedule")
                with col2:
                    submit_delete = st.form_submit_button("❌ Delete")
                
                if submit_reschedule:
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute('''
                            UPDATE appointments
                            SET date = ?, time = ?
                            WHERE id = ?
                        ''', (new_date.strftime("%Y-%m-%d"), new_time, df.iloc[selected_index]["id"]))
                        conn.commit()
                        conn.close()
                        st.success("✅ Appointment rescheduled successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to reschedule appointment: {str(e)}")
                
                elif submit_delete:
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute('DELETE FROM appointments WHERE id = ?', (df.iloc[selected_index]["id"],))
                        conn.commit()
                        conn.close()
                        st.warning("🗑️ Appointment deleted.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to delete appointment: {str(e)}")
        else:
            st.info("No appointments found for this patient.")

# --- Billing System ---
def billing_system():
    st.title("🧾 Billing System")
    
    with st.sidebar.expander("⚙️ Billing Settings", expanded=True):
        st.info("Configure billing settings here")
    
    st.header("Generate Patient Invoice")
    patient_id = st.text_input("Patient ID *", placeholder="Enter patient ID")
    patient = get_patient(patient_id) if patient_id else None
    
    if patient_id and not patient:
        st.warning("Patient ID not found. Please register the patient first.")
        return
    
    patient_name = f"{patient['first_name']} {patient['last_name']}" if patient else ""
    
    service_catalog = {
        "Services": {
            "Consultation": 500,
            "X-Ray": 1000,
            "Blood Test": 750,
            "ECG": 400,
            "MRI Scan": 3500,
            "Ultrasound": 1200,
            "Physiotherapy": 800,
            "Surgery Charges": 15000,
            "ICU Charges": 5000
        },
        "Medications": {
            "Amoxicillin (500mg tid)": 150,
            "Atorvastatin (10mg od)": 100,
            "Azithromycin (500mg od)": 200,
            "Cetirizine (10mg od)": 50,
            "Diclofenac (50mg bid)": 80,
            "Doxycycline (100mg bid)": 120,
            "Hydrocortisone cream (bid)": 90,
            "Ibuprofen (400mg bid)": 60,
            "Iron supplements (od)": 70,
            "Levocetirizine (5mg/10mg od)": 110,
            "Loratadine (10mg od)": 55,
            "Metronidazole (400mg tid)": 130,
            "Multivitamin (daily)": 40,
            "Ondansetron (4mg tid)": 140,
            "Oral rehydration salts": 30,
            "Pantoprazole (40mg od)": 95,
            "Paracetamol (500mg tid)": 45,
            "Salbutamol inhaler (prn)": 250
        }
    }

    # Combine services and medications for multiselect
    all_items = (
        [f"Service: {key}" for key in service_catalog["Services"].keys()] +
        [f"Medication: {key}" for key in service_catalog["Medications"].keys()]
    )

    selected_items = st.multiselect(
        "Select services and medications:",
        options=all_items,
        help="Select all services and medications the patient received"
    )
    
    insurance_coverage = st.slider("Insurance Coverage (%)", 0, 100, 70)

    if selected_items:
        selected_costs = {}
        for item in selected_items:
            if item.startswith("Service: "):
                service_name = item.replace("Service: ", "")
                selected_costs[service_name] = service_catalog["Services"][service_name]
            elif item.startswith("Medication: "):
                med_name = item.replace("Medication: ", "")
                selected_costs[med_name] = service_catalog["Medications"][med_name]
        
        total = sum(selected_costs.values())
        covered = int(total * insurance_coverage / 100)
        payable = total - covered

        df = pd.DataFrame({
            "Item": [item for item in selected_costs.keys()],
            "Type": ["Service" if item in service_catalog["Services"] else "Medication" for item in selected_costs.keys()],
            "Cost (INR)": list(selected_costs.values())
        })
        
        st.subheader("Invoice Breakdown")
        st.table(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Charges", f"INR {total}")
            st.metric("Insurance Covered", f"-INR {covered}")
        with col2:
            st.metric("Amount Payable", f"INR {payable}", delta_color="inverse")

        if patient_id and patient_name:
            pdf_buffer = generate_invoice_pdf(patient_name, selected_costs, total, covered, payable)
            st.download_button(
                label="📥 Download Invoice as PDF",
                data=pdf_buffer,
                file_name=f"invoice_{patient_name.replace(' ', '_')}.pdf",
                mime="application/pdf",
                type="primary"
            )
            
            if st.button("💾 Save Invoice"):
                try:
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute('''
                        INSERT INTO invoices (patient_id, items, total, covered, payable, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        patient_id,
                        json.dumps(selected_costs),
                        total,
                        covered,
                        payable,
                        datetime.now().isoformat()
                    ))
                    conn.commit()
                    conn.close()
                    st.success("✅ Invoice saved to patient record.")
                except Exception as e:
                    st.error(f"❌ Failed to save invoice: {str(e)}")
    else:
        st.info("Please select at least one service or medication to generate the bill.")

# --- FHIR Export ---
def fhir_export_page():
    st.title("📤 Export to FHIR-Compatible EHR")
    st.markdown("Generate a FHIR-compliant Patient JSON for external systems.")

    if 'fhir_data' not in st.session_state:
        st.session_state.fhir_data = None

    patient_id = st.text_input("Patient ID *", placeholder="Enter patient ID")
    patient = get_patient(patient_id) if patient_id else None
    
    if patient_id and not patient:
        st.warning("Patient ID not found. Please register the patient first.")
        return

    with st.form("fhir_export_form"):
        first_name = st.text_input("First Name", value=patient['first_name'] if patient else "John", disabled=bool(patient))
        last_name = st.text_input("Last Name", value=patient['last_name'] if patient else "Doe", disabled=bool(patient))
        dob = st.date_input("Date of Birth", value=datetime.strptime(patient['dob'], "%Y-%m-%d") if patient else datetime.today())
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(patient['gender']) if patient else 0, disabled=bool(patient))

        if st.form_submit_button("📤 Generate FHIR JSON"):
            try:
                patient_data = {
                    "patient_id": patient_id,
                    "first_name": first_name,
                    "last_name": last_name,
                    "dob": dob.strftime("%Y-%m-%d"),
                    "gender": gender
                }

                fhir_patient = {
                    "resourceType": "Patient",
                    "id": patient_data["patient_id"],
                    "name": [{
                        "use": "official",
                        "family": patient_data["last_name"],
                        "given": [patient_data["first_name"]]
                    }],
                    "gender": patient_data["gender"].lower(),
                    "birthDate": patient_data["dob"]
                }

                st.session_state.fhir_data = json.dumps(fhir_patient, indent=2)
                st.success("FHIR patient data generated successfully! Click the download button below.")
                
            except Exception as e:
                st.error(f"Error generating FHIR data: {str(e)}")

    if st.session_state.fhir_data:
        st.download_button(
            label="📥 Download FHIR Patient File",
            data=st.session_state.fhir_data,
            file_name=f"fhir_patient_{patient_id}.json",
            mime="application/json"
        )

def database_visualization():
    st.title("📊 Database Visualization and Editing")
    st.caption("Explore, visualize, and edit patient data stored in the hospital database")

    # Connect to the database
    conn = sqlite3.connect(DB_PATH)
    
    # Define table queries with joins for patient names
    table_queries = {
        "patients": {
            "query": "SELECT patient_id, first_name, last_name, dob, gender, email, phone, address, created_at FROM patients",
            "display_columns": ["patient_id", "first_name", "last_name", "dob", "gender", "email", "phone", "address", "created_at"],
            "editable_columns": ["first_name", "last_name", "dob", "gender", "email", "phone", "address"],
            "required_columns": ["first_name", "last_name", "dob", "gender"],
            "primary_key": "patient_id"
        },
        "appointments": {
            "query": """
                SELECT a.id, a.patient_id, p.first_name || ' ' || p.last_name AS patient_name, 
                       a.department, a.doctor, a.date, a.time
                FROM appointments a
                LEFT JOIN patients p ON a.patient_id = p.patient_id
            """,
            "display_columns": ["id", "patient_id", "patient_name", "department", "doctor", "date", "time"],
            "editable_columns": ["patient_id", "department", "doctor", "date", "time"],
            "required_columns": ["patient_id", "department", "doctor", "date", "time"],
            "primary_key": "id"
        },
        "invoices": {
            "query": """
                SELECT i.id, i.patient_id, p.first_name || ' ' || p.last_name AS patient_name, 
                       i.items, i.total, i.covered, i.payable, i.created_at
                FROM invoices i
                LEFT JOIN patients p ON i.patient_id = p.patient_id
            """,
            "display_columns": ["id", "patient_id", "patient_name", "items", "total", "covered", "payable", "created_at"],
            "editable_columns": ["patient_id", "items", "total", "covered", "payable"],
            "required_columns": ["patient_id", "items", "total", "covered", "payable"],
            "primary_key": "id"
        }
    }

    # Tabs for visualization and editing
    tabs = st.tabs(["View Database", "Edit Database"])

    # Visualization Tab
    with tabs[0]:
        for table, config in table_queries.items():
            with st.expander(f"Table: {table.capitalize()}"):
                # Fetch data
                df = pd.read_sql_query(config["query"], conn)
                
                # Display statistics
                st.markdown(f"**Records**: {len(df)}")
                
                # Display DataFrame with selected columns
                st.subheader(f"{table.capitalize()} Data")
                st.dataframe(df[config["display_columns"]], use_container_width=True)
                
                # Download as CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download {table}.csv",
                    data=csv,
                    file_name=f"{table}_data.csv",
                    mime="text/csv",
                    key=f"db_viz_download_{table}_{uuid.uuid4()}"
                )

    # Editing Tab
    with tabs[1]:
        st.subheader("Edit Database Records")
        unique_form_id = str(uuid.uuid4())
        table_to_edit = st.selectbox("Select Table to Edit", list(table_queries.keys()), key=f"db_viz_table_select_{unique_form_id}")
        
        config = table_queries[table_to_edit]
        primary_key = config["primary_key"]
        
        # Fetch primary keys for selection
        df = pd.read_sql_query(config["query"], conn)
        if df.empty:
            st.warning(f"No records found in {table_to_edit} table.")
            conn.close()
            return
        
        record_ids = df[primary_key].astype(str).tolist()
        selected_record_id = st.selectbox(f"Select Record ({primary_key})", record_ids, key=f"db_viz_edit_{table_to_edit}_id_{unique_form_id}")
        
        # Fetch the selected record
        query = f"SELECT * FROM {table_to_edit} WHERE {primary_key} = ?"
        record = pd.read_sql_query(query, conn, params=(selected_record_id,)).iloc[0]
        
        # Editing form
        with st.form(f"db_viz_edit_{table_to_edit}_form_{unique_form_id}"):
            st.markdown(f"**Editing {table_to_edit.capitalize()} Record ({primary_key}: {selected_record_id})**")
            form_inputs = {}
            patients_available = True
            
            for col in config["editable_columns"]:
                if col == "patient_id":
                    # Check for patients in the database
                    patients_df = pd.read_sql_query("SELECT patient_id, first_name || ' ' || last_name AS patient_name FROM patients", conn)
                    if patients_df.empty:
                        st.warning("No patients found in the database. Please add patients via the 'FHIR Export' section or another patient creation feature before editing appointments or invoices.")
                        patients_available = False
                        form_inputs[col] = record[col]
                        st.text_input("Patient ID (cannot be edited until patients are added)", value=record[col], disabled=True, key=f"db_viz_{table_to_edit}_{col}_{unique_form_id}")
                    else:
                        patient_options = {row["patient_name"]: row["patient_id"] for _, row in patients_df.iterrows()}
                        current_patient_id = record[col]
                        default_index = 0
                        if current_patient_id in patient_options.values():
                            default_index = list(patient_options.values()).index(current_patient_id)
                        selected_name = st.selectbox(
                            "Patient",
                            list(patient_options.keys()),
                            index=default_index,
                            key=f"db_viz_{table_to_edit}_{col}_{unique_form_id}"
                        )
                        form_inputs[col] = patient_options.get(selected_name, list(patient_options.values())[0])
                elif col == "gender":
                    form_inputs[col] = st.selectbox(col.capitalize(), ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(record[col]) if record[col] in ["Male", "Female", "Other"] else 0, key=f"db_viz_{table_to_edit}_{col}_{unique_form_id}")
                elif col == "dob":
                    form_inputs[col] = st.date_input(col.capitalize(), value=datetime.strptime(record[col], "%Y-%m-%d"), min_value=datetime(1900, 1, 1), max_value=datetime.today(), key=f"db_viz_{table_to_edit}_{col}_{unique_form_id}")
                elif col == "date":
                    form_inputs[col] = st.date_input(col.capitalize(), value=datetime.strptime(record[col], "%Y-%m-%d"), min_value=datetime.today(), key=f"db_viz_{table_to_edit}_{col}_{unique_form_id}")
                elif col == "time":
                    if table_to_edit == "appointments":
                        doctor = record["doctor"]
                        def generate_time_slots(doctor):
                            match = re.search(r"\((\d+\s*(?:AM|PM))\s*-\s*(\d+\s*(?:AM|PM))\)", doctor)
                            if not match:
                                return []
                            start_str, end_str = match.groups()
                            try:
                                start_time = datetime.strptime(start_str, "%I %p")
                                end_time = datetime.strptime(end_str, "%I %p")
                                if end_time.hour < start_time.hour:
                                    end_time = end_time.replace(hour=end_time.hour + 12)
                                time_slots = []
                                current_time = start_time
                                while current_time < end_time:
                                    time_slots.append(current_time.strftime("%I:%M %p"))
                                    current_time += timedelta(minutes=30)
                                return time_slots
                            except ValueError:
                                return []
                        time_slots = generate_time_slots(doctor)
                        if not time_slots:
                            st.error("Invalid doctor availability. Please update doctor first.")
                            conn.close()
                            return
                        form_inputs[col] = st.selectbox(col.capitalize(), time_slots, index=time_slots.index(record[col]) if record[col] in time_slots else 0, key=f"db_viz_{table_to_edit}_{col}_{unique_form_id}")
                    else:
                        form_inputs[col] = st.text_input(col.capitalize(), value=record[col], key=f"db_viz_{table_to_edit}_{col}_{unique_form_id}")
                elif col == "department":
                    form_inputs[col] = st.selectbox(col.capitalize(), ["General Medicine", "Cardiology", "Orthopedics", "Pediatrics", "Neurology"], index=["General Medicine", "Cardiology", "Orthopedics", "Pediatrics", "Neurology"].index(record[col]) if record[col] in ["General Medicine", "Cardiology", "Orthopedics", "Pediatrics", "Neurology"] else 0, key=f"db_viz_{table_to_edit}_{col}_{unique_form_id}")
                elif col == "doctor" and table_to_edit == "appointments":
                    department = record["department"]
                    doctors_by_department = {
                        "General Medicine": ["Dr. Smith (9 AM - 12 PM)", "Dr. Arjun (2 PM - 5 PM)", "Dr. Kiran (10 AM - 1 PM)"],
                        "Cardiology": ["Dr. Rakesh (10 AM - 1 PM)", "Dr. Meera (1 PM - 4 PM)", "Dr. Arvind (3 PM - 6 PM)"],
                        "Orthopedics": ["Dr. Leena (11 AM - 2 PM)", "Dr. Rajiv (3 PM - 6 PM)", "Dr. Divya (9 AM - 12 PM)"],
                        "Pediatrics": ["Dr. Fatima (9 AM - 11 AM)", "Dr. Sunil (1 PM - 3 PM)", "Dr. Neeraj (3 PM - 5 PM)"],
                        "Neurology": ["Dr. Nisha (1 PM - 4 PM)", "Dr. Vikram (10 AM - 12 PM)", "Dr. Sneha (2 PM - 5 PM)"]
                    }
                    form_inputs[col] = st.selectbox(col.capitalize(), doctors_by_department.get(department, []), index=doctors_by_department[department].index(record[col]) if record[col] in doctors_by_department[department] else 0, key=f"db_viz_{table_to_edit}_{col}_{unique_form_id}")
                elif col == "items":
                    form_inputs[col] = st.text_area(col.capitalize(), value=json.dumps(json.loads(record[col]), indent=2), key=f"db_viz_{table_to_edit}_{col}_{unique_form_id}")
                elif col in ["total", "covered", "payable"]:
                    form_inputs[col] = st.number_input(col.capitalize(), min_value=0.0, value=float(record[col]), step=0.01, key=f"db_viz_{table_to_edit}_{col}_{unique_form_id}")
                else:
                    form_inputs[col] = st.text_area(col.capitalize(), value=record[col], key=f"db_viz_{table_to_edit}_{col}_{unique_form_id}")
            
            col1, col2 = st.columns(2)
            with col1:
                submit_update = st.form_submit_button("💾 Update Record", type="primary")
            with col2:
                submit_delete = st.form_submit_button("🗑️ Delete Record", type="secondary")
            
            if submit_update:
                if not patients_available and table_to_edit in ["appointments", "invoices"]:
                    st.error(f"Cannot update {table_to_edit} record: No patients are available in the database. Please add patients first.")
                    conn.close()
                    return
                
                # Validate required fields
                for col in config["required_columns"]:
                    if not form_inputs.get(col):
                        st.error(f"{col.capitalize()} is required.")
                        conn.close()
                        return
                
                # Validate patient_id
                if "patient_id" in form_inputs:
                    c = conn.cursor()
                    c.execute("SELECT patient_id FROM patients WHERE patient_id = ?", (form_inputs["patient_id"],))
                    if not c.fetchone():
                        st.error("Invalid patient_id. Patient does not exist.")
                        conn.close()
                        return
                
                # Validate items JSON for invoices
                if table_to_edit == "invoices" and "items" in form_inputs:
                    try:
                        json.loads(form_inputs["items"])
                    except json.JSONDecodeError:
                        st.error("Items must be valid JSON.")
                        conn.close()
                        return
                
                # Validate date format for dob and date
                if "dob" in form_inputs:
                    form_inputs["dob"] = form_inputs["dob"].strftime("%Y-%m-%d")
                if "date" in form_inputs:
                    form_inputs["date"] = form_inputs["date"].strftime("%Y-%m-%d")
                
                # Update record
                try:
                    c = conn.cursor()
                    update_cols = [f"{col} = ?" for col in config["editable_columns"]]
                    update_query = f"UPDATE {table_to_edit} SET {', '.join(update_cols)} WHERE {primary_key} = ?"
                    update_values = [form_inputs[col] for col in config["editable_columns"]] + [selected_record_id]
                    c.execute(update_query, update_values)
                    conn.commit()
                    st.success(f"✅ Record updated successfully in {table_to_edit}.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to update record: {str(e)}")
            
            if submit_delete:
                if table_to_edit == "patients":
                    st.warning(f"Deleting patient {selected_record_id} will also delete all related appointments and invoices. This action cannot be undone.")
                    confirm_delete = st.checkbox("Confirm deletion of patient and all related data", key=f"db_viz_confirm_delete_{selected_record_id}_{unique_form_id}")
                    if not confirm_delete:
                        st.info("Please confirm deletion to proceed.")
                        conn.close()
                        return
                
                try:
                    c = conn.cursor()
                    c.execute(f"DELETE FROM {table_to_edit} WHERE {primary_key} = ?", (selected_record_id,))
                    conn.commit()
                    if table_to_edit == "patients":
                        st.warning(f"🗑️ Patient {selected_record_id} and all related data deleted.")
                    else:
                        st.warning(f"🗑️ Record deleted from {table_to_edit}.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to delete record: {str(e)}")
    
    conn.close()

# --- Main Run Function ---
def run():
    init_db()  # Initialize database
    
    st.sidebar.title("🏥 Hospital Management")
    app_mode = st.sidebar.radio("Navigation", [
        "Patient Management",
        "Feedback Generator",
        "Appointment Scheduling",
        "Billing System",
        "FHIR Export",
        "Database Visualization"
    ])
    
    if app_mode == "Patient Management":
        patient_management()
    elif app_mode == "Feedback Generator":
        model, encode, decode = load_model(FEEDBACK_MODEL_PATH)
        if model:
            feedback_generator(model, encode, decode)
    elif app_mode == "Appointment Scheduling":
        appointment_scheduling()
    elif app_mode == "Billing System":
        billing_system()
    elif app_mode == "FHIR Export":
        fhir_export_page()
    elif app_mode == "Database Visualization":
        database_visualization()

if __name__ == "__main__":
    run()
