import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import pandas as pd
from pathlib import Path
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from datetime import datetime, timedelta
import re
import json
import sqlite3
import uuid

# --- Configuration ---
FEEDBACK_MODEL_PATH = "tiny_gpt_feedback.pt"
PRESCRIPTION_MODEL_PATH = "tiny_gpt_prescription_optimized.pt"
DOC_MODEL_PATH = "tiny_gpt_documentation.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DB_PATH = "data/hospital.db"

# --- Hyperparameters ---
feedback_block_size = 64
feedback_embedding_dim = 128
feedback_n_heads = 4
feedback_n_layers = 2
feedback_dropout = 0.1

prescription_block_size = 128
prescription_embedding_dim = 128
prescription_n_heads = 4
prescription_n_layers = 3
prescription_dropout = 0.1

# Vocabulary
feedback_vocab = [
    ' ', "'", ',', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    ':', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

prescription_vocab = sorted(list(set("abcdefghijklmnopqrstuvwxyz0123456789:,.!?()-=+ \n")))

def get_vocab(model_type):
    vocab = feedback_vocab if model_type == "feedback" else prescription_vocab
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    encode = lambda s: [stoi[c] for c in s.lower() if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])
    return vocab, encode, decode

# --- Database Setup ---
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
        CREATE TABLE IF NOT EXISTS prescriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            doctor TEXT NOT NULL,
            diagnosis TEXT NOT NULL,
            medications TEXT NOT NULL,
            instructions TEXT NOT NULL,
            created_at TEXT NOT NULL,
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
    
    c.execute('SELECT * FROM appointments WHERE patient_id = ?', (patient_id,))
    appointments = c.fetchall()
    
    c.execute('SELECT * FROM prescriptions WHERE patient_id = ?', (patient_id,))
    prescriptions = c.fetchall()
    
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
        "prescriptions": [
            {
                "id": p[0],
                "patient_id": p[1],
                "doctor": p[2],
                "diagnosis": p[3],
                "medications": p[4],
                "instructions": p[5],
                "created_at": p[6]
            } for p in prescriptions
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

class TinyGPTPrescription(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, prescription_embedding_dim)
        self.position_embedding = nn.Embedding(prescription_block_size, prescription_embedding_dim)
        self.blocks = nn.Sequential(*[Block(prescription_embedding_dim, prescription_n_heads, prescription_dropout) 
                                    for _ in range(prescription_n_layers)])
        self.ln_final = nn.LayerNorm(prescription_embedding_dim)
        self.fc_out = nn.Linear(prescription_embedding_dim, vocab_size)
        self.dropout = nn.Dropout(prescription_dropout)

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

    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -_block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
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
        if "feedback" in model_path.lower():
            vocab, encode, decode = get_vocab("feedback")
            model = TinyGPTFeedback(len(vocab)).to(DEVICE)
        elif "prescription" in model_path.lower():
            vocab, encode, decode = get_vocab("prescription")
            model = TinyGPTPrescription(len(vocab)).to(DEVICE)
        else:
            st.error("Unknown model type. Please specify 'feedback' or 'prescription' in the model path.")
            return None, None, None

        model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
        model.eval()
        st.success(f"✅ Loaded model from {model_path}")
        return model, encode, decode
    except Exception as e:
        st.error(f"❌ Model load failed: {str(e)}. Please check the model file and try again.")
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
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica", 12)
    y = height - 60

    c.drawString(50, y, f"Hospital Invoice for: {patient_name}")
    y -= 30
    for item, cost in selected_costs.items():
        c.drawString(60, y, f"{item}: ₹{cost}")
        y -= 20
    y -= 10
    c.drawString(60, y, f"Total: ₹{total}")
    y -= 20
    c.drawString(60, y, f"Insurance Covered: -₹{covered}")
    y -= 20
    c.drawString(60, y, f"Amount Payable: ₹{payable}")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def generate_prescription_pdf(doctor, patient, age, gender, diagnosis, medications, instructions):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height-50, "MEDICAL PRESCRIPTION")
    c.setFont("Helvetica", 10)
    c.drawRightString(width-50, height-50, datetime.now().strftime('%Y-%m-%d'))
    c.line(50, height-60, width-50, height-60)
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height-90, "Doctor:")
    c.setFont("Helvetica", 12)
    c.drawString(150, height-90, doctor)
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height-115, "Patient:")
    c.setFont("Helvetica", 12)
    c.drawString(150, height-115, f"{patient}, {age}y ({gender})")
    
    y = height - 155
    section_height = 30
    
    def draw_section(title, content):
        nonlocal y
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, title)
        c.setFont("Helvetica", 11)
        y -= 20
        for line in content.split('\n'):
            if line.strip():
                if line.startswith('-'):
                    c.drawString(55, y, '• ' + line[1:].strip())
                else:
                    c.drawString(55, y, line.strip())
                y -= 15
        y -= section_height
    
    draw_section("Diagnosis:", diagnosis)
    draw_section("Medications:", medications)
    draw_section("Instructions:", instructions)
    
    c.setFont("Helvetica-Oblique", 9)
    c.drawCentredString(width/2, 30, "Electronic prescription - Valid without signature")
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# --- Patient Management ---
def patient_management(prefix):
    st.header("👤 Patient Management")
    st.caption("Register, search, and view patient records")

    tabs = st.tabs(["Register Patient", "Search Patients", "View Patient Record"])

    with tabs[0]:
        st.subheader("Register New Patient")
        with st.form(f"{prefix}_patient_registration_form"):
            col1, col2 = st.columns(2)
            with col1:
                first_name = st.text_input("First Name *", placeholder="John", key=f"{prefix}_first_name")
                dob = st.date_input("Date of Birth *", min_value=datetime(1900, 1, 1), max_value=datetime.today(), key=f"{prefix}_dob")
                email = st.text_input("Email", placeholder="john.doe@example.com", key=f"{prefix}_email")
            with col2:
                last_name = st.text_input("Last Name *", placeholder="Doe", key=f"{prefix}_last_name")
                gender = st.selectbox("Gender *", ["Male", "Female", "Other"], key=f"{prefix}_gender")
                phone = st.text_input("Phone", placeholder="+91 1234567890", key=f"{prefix}_phone")
            
            address = st.text_area("Address", placeholder="123 Main St, City, Country", key=f"{prefix}_address")
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
                        st.session_state[f"{prefix}_selected_patient_id"] = patient_id
                    except Exception as e:
                        st.error(f"❌ Failed to register patient: {str(e)}")

    with tabs[1]:
        st.subheader("Search Patients")
        search_term = st.text_input("Search by Name or ID", placeholder="Enter name or patient ID", key=f"{prefix}_search_term")
        if search_term:
            patients = search_patients(search_term)
            if patients:
                df = pd.DataFrame(patients)
                st.dataframe(df, use_container_width=True)
                selected_index = st.number_input("Select patient index", min_value=0, max_value=len(patients)-1, step=1, key=f"{prefix}_selected_index")
                if st.button("View Record", key=f"{prefix}_view_record"):
                    st.session_state[f"{prefix}_selected_patient_id"] = patients[selected_index]["patient_id"]
                    st.rerun()
            else:
                st.info("No patients found matching the search term.")

    with tabs[2]:
        st.subheader("Patient Record")
        patient_id = st.text_input("Enter Patient ID", value=st.session_state.get(f"{prefix}_selected_patient_id", ""), key=f"{prefix}_patient_id_record")
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

                with st.expander("Prescriptions"):
                    if records["prescriptions"]:
                        for p in records["prescriptions"]:
                            st.markdown(f"**Prescription #{p['id']}** (Created: {p['created_at']})")
                            st.markdown(f"- **Doctor**: {p['doctor']}")
                            st.markdown(f"- **Diagnosis**: {p['diagnosis']}")
                            st.markdown(f"- **Medications**: {p['medications']}")
                            st.markdown(f"- **Instructions**: {p['instructions']}")
                            st.markdown("---")
                    else:
                        st.info("No prescriptions found.")

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
def feedback_generator(model, encode, decode, prefix):
    st.header("🏥 Patient Feedback Generator")
    st.markdown("Generate patient feedback using AI based on your hospital's HIS data")
    
    with st.sidebar.expander("⚙️ Feedback Settings", expanded=True):
        temperature = st.slider("Creativity", 0.1, 1.0, 0.7, key=f"{prefix}_feedback_temperature")
        max_length = st.slider("Max Length", 100, 500, 300, step=50, key=f"{prefix}_feedback_max_length")
        
        if st.button("Clear Cache", key=f"{prefix}_clear_cache"):
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
    
    selected_template = st.selectbox("Choose a template:", prompt_templates, key=f"{prefix}_feedback_template")
    custom_prompt = st.text_input("Or enter your own prompt:", value=selected_template, key=f"{prefix}_feedback_prompt")
    
    if st.button("Generate Feedback", type="primary", key=f"{prefix}_generate_feedback"):
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
def appointment_scheduling(prefix):
    st.header("📅 Appointment Scheduling")
    
    with st.sidebar.expander("⚙️ Appointment Settings", expanded=True):
        st.info("Configure appointment settings here")
    
    st.subheader("Schedule New Appointment")
    patient_id = st.text_input("Patient ID *", placeholder="Enter patient ID", key=f"{prefix}_appt_patient_id")
    patient = get_patient(patient_id) if patient_id else None
    
    if patient_id and not patient:
        st.warning("Patient ID not found. Please register the patient first.")
        return
    
    if f"{prefix}_selected_department" not in st.session_state:
        st.session_state[f"{prefix}_selected_department"] = "General Medicine"
    if f"{prefix}_selected_doctor" not in st.session_state:
        st.session_state[f"{prefix}_selected_doctor"] = None
    
    department = st.selectbox(
        "Department:",
        ["General Medicine", "Cardiology", "Orthopedics", "Pediatrics", "Neurology"],
        key=f"{prefix}_department_select",
        on_change=lambda: st.session_state.update({f"{prefix}_selected_doctor": None, f"{prefix}_selected_time": None})
    )
    
    if department != st.session_state[f"{prefix}_selected_department"]:
        st.session_state[f"{prefix}_selected_department"] = department
        st.session_state[f"{prefix}_selected_doctor"] = None
        st.session_state[f"{prefix}_selected_time"] = None
    
    doctors_by_department = {
        "General Medicine": ["Dr. Smith (9 AM - 12 PM)", "Dr. Arjun (2 PM - 5 PM)", "Dr. Kiran (10 AM - 1 PM)"],
        "Cardiology": ["Dr. Rakesh (10 AM - 1 PM)", "Dr. Meera (1 PM - 4 PM)", "Dr. Arvind (3 PM - 6 PM)"],
        "Orthopedics": ["Dr. Leena (11 AM - 2 PM)", "Dr. Rajiv (3 PM - 6 PM)", "Dr. Divya (9 AM - 12 PM)"],
        "Pediatrics": ["Dr. Fatima (9 AM - 11 AM)", "Dr. Sunil (1 PM - 3 PM)", "Dr. Neeraj (3 PM - 5 PM)"],
        "Neurology": ["Dr. Nisha (1 PM - 4 PM)", "Dr. Vikram (10 AM - 12 PM)", "Dr. Sneha (2 PM - 5 PM)"]
    }
    
    doctor = st.selectbox(
        "Preferred Doctor:",
        doctors_by_department[department],
        key=f"{prefix}_doctor_select",
        index=0 if not st.session_state.get(f"{prefix}_selected_doctor") else doctors_by_department[department].index(st.session_state[f"{prefix}_selected_doctor"]) if st.session_state[f"{prefix}_selected_doctor"] in doctors_by_department[department] else 0,
        on_change=lambda: st.session_state.update({f"{prefix}_selected_time": None})
    )
    
    if doctor != st.session_state[f"{prefix}_selected_doctor"]:
        st.session_state[f"{prefix}_selected_doctor"] = doctor
        st.session_state[f"{prefix}_selected_time"] = None
    
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
        st.warning("No valid time slots available for this doctor. Please select another doctor.")
        return
    
    time = st.selectbox(
        "Select Time:",
        time_slots,
        key=f"{prefix}_time_select",
        index=0 if not st.session_state.get(f"{prefix}_selected_time") else time_slots.index(st.session_state[f"{prefix}_selected_time"]) if st.session_state[f"{prefix}_selected_time"] in time_slots else 0
    )
    
    st.session_state[f"{prefix}_selected_time"] = time
    
    with st.form(f"{prefix}_appointment_form"):
        st.markdown(f"**Patient**: {patient['first_name']} {patient['last_name']}" if patient else "**Patient**: Not selected")
        date = st.date_input("Select Appointment Date:", min_value=datetime.today(), key=f"{prefix}_appt_date")
        
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

    st.subheader("📋 Manage Appointments")
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
                step=1,
                key=f"{prefix}_appt_manage_index"
            )
            
            with st.form(f"{prefix}_manage_form"):
                new_date = st.date_input("New date:", value=datetime.strptime(df.iloc[selected_index]["date"], "%Y-%m-%d"), key=f"{prefix}_appt_new_date")
                new_time = st.selectbox(
                    "New time:",
                    time_slots,
                    index=time_slots.index(df.iloc[selected_index]["time"]) if df.iloc[selected_index]["time"] in time_slots else 0,
                    key=f"{prefix}_appt_new_time"
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
def billing_system(prefix):
    st.header("🧾 Billing System")
    
    with st.sidebar.expander("⚙️ Billing Settings", expanded=True):
        st.info("Configure billing settings here")
    
    st.subheader("Generate Patient Invoice")
    patient_id = st.text_input("Patient ID *", placeholder="Enter patient ID", key=f"{prefix}_billing_patient_id")
    patient = get_patient(patient_id) if patient_id else None
    
    if patient_id and not patient:
        st.warning("Patient ID not found. Please register the patient first.")
        return
    
    patient_name = f"{patient['first_name']} {patient['last_name']}" if patient else ""
    
    medication_prices = {
        "paracetamol": 50,
        "ibuprofen": 80,
        "amoxicillin": 120,
        "omeprazole": 95,
        "cetirizine": 60,
        "atorvastatin": 150,
        "metformin": 70,
        "losartan": 110,
        "aspirin": 40,
        "diazepam": 90,
        "simvastatin": 130,
        "amlodipine": 85,
        "levothyroxine": 75,
        "prednisone": 65,
        "warfarin": 110,
        "furosemide": 55,
        "metoprolol": 80,
        "hydrochlorothiazide": 60,
        "sertraline": 95,
        "fluoxetine": 85
    }
    
    service_catalog = {
        "Consultation": 500,
        "X-Ray": 1000,
        "Blood Test": 750,
        "ECG": 400,
        "MRI Scan": 3500,
        "Ultrasound": 1200,
        "Physiotherapy": 800,
        "Surgery Charges": 15000,
        "ICU Charges": 5000
    }

    def extract_medications_from_prescription(prescription_text):
        if not prescription_text:
            return []
            
        normalized_text = prescription_text.lower()
        found_meds = []
        
        for med in medication_prices.keys():
            if med.lower() in normalized_text:
                found_meds.append(med)
                
        med_patterns = [
            r"(\w+) \d+mg",
            r"(\w+) \d+\s?x\s?\d+",
            r"(\w+) \d+\.\d+",
            r"(\w+) \d+/\d+"
        ]
        
        for pattern in med_patterns:
            matches = re.findall(pattern, normalized_text)
            for match in matches:
                if match.lower() in medication_prices and match.lower() not in found_meds:
                    found_meds.append(match.lower())
        
        return list(set(found_meds))

    medications_from_prescription = []
    if patient_id:
        records = get_patient_records(patient_id)
        latest_prescription = records["prescriptions"][-1] if records["prescriptions"] else None
        if latest_prescription:
            medications_from_prescription = extract_medications_from_prescription(latest_prescription["medications"])
    
    selected_services = st.multiselect(
        "Select services:",
        options=list(service_catalog.keys()),
        help="Select all services the patient received",
        key=f"{prefix}_billing_services"
    )
    
    if medications_from_prescription:
        st.subheader("Medications from Latest Prescription")
        med_display = "\n".join([f"- {med.capitalize()} (₹{medication_prices[med]})" 
                               for med in medications_from_prescription])
        st.markdown(med_display)
        
        confirm_meds = st.checkbox(
            "Include all detected medications in bill",
            value=True,
            help="Uncheck to remove medications from billing",
            key=f"{prefix}_billing_confirm_meds"
        )
    else:
        confirm_meds = False
        st.info("No medications detected in prescription. Add manually below if needed.")
    
    additional_meds = st.multiselect(
        "Add additional medications:",
        options=[m for m in medication_prices.keys() 
                if m not in medications_from_prescription],
        help="Add medications not automatically detected",
        key=f"{prefix}_billing_additional_meds"
    )
    
    insurance_coverage = st.slider("Insurance Coverage (%)", 0, 100, 70, key=f"{prefix}_billing_insurance")

    if selected_services or (confirm_meds and medications_from_prescription) or additional_meds:
        service_costs = {k: service_catalog[k] for k in selected_services}
        medication_costs = {k: medication_prices[k] for k in (medications_from_prescription if confirm_meds else []) + additional_meds}
        selected_costs = {**service_costs, **medication_costs}
        
        total = sum(selected_costs.values())
        covered = int(total * insurance_coverage / 100)
        payable = total - covered

        df = pd.DataFrame({
            "Item": [item.capitalize() for item in selected_costs.keys()],
            "Type": ["Service" if k in service_catalog else "Medication" 
                     for k in selected_costs.keys()],
            "Cost (₹)": list(selected_costs.values())
        })
        
        st.subheader("Invoice Breakdown")
        st.table(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Charges", f"₹{total}")
            st.metric("Insurance Covered", f"-₹{covered}")
        with col2:
            st.metric("Amount Payable", f"₹{payable}", delta_color="inverse")

        if patient_id and patient_name:
            pdf_buffer = generate_invoice_pdf(patient_name, selected_costs, total, covered, payable)
            st.download_button(
                label="📥 Download Invoice as PDF",
                data=pdf_buffer,
                file_name=f"invoice_{patient_name.replace(' ', '_')}.pdf",
                mime="application/pdf",
                type="primary",
                key=f"{prefix}_billing_download_pdf"
            )
            
            if st.button("💾 Save Invoice", key=f"{prefix}_billing_save"):
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
def fhir_export_page(prefix):
    st.header("📤 Export to FHIR-Compatible EHR")
    st.markdown("Generate a FHIR-compliant Patient JSON for external systems.")

    if f"{prefix}_fhir_data" not in st.session_state:
        st.session_state[f"{prefix}_fhir_data"] = None

    patient_id = st.text_input("Patient ID *", placeholder="Enter patient ID", key=f"{prefix}_fhir_patient_id")
    patient = get_patient(patient_id) if patient_id else None
    
    if patient_id and not patient:
        st.warning("Patient ID not found. Please register the patient first.")
        return

    with st.form(f"{prefix}_fhir_export_form"):
        first_name = st.text_input("First Name", value=patient['first_name'] if patient else "John", disabled=bool(patient), key=f"{prefix}_fhir_first_name")
        last_name = st.text_input("Last Name", value=patient['last_name'] if patient else "Doe", disabled=bool(patient), key=f"{prefix}_fhir_last_name")
        dob = st.date_input("Date of Birth", value=datetime.strptime(patient['dob'], "%Y-%m-%d") if patient else datetime.today(), key=f"{prefix}_fhir_dob")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(patient['gender']) if patient else 0, disabled=bool(patient), key=f"{prefix}_fhir_gender")

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

                st.session_state[f"{prefix}_fhir_data"] = json.dumps(fhir_patient, indent=2)
                st.success("FHIR patient data generated successfully! Click the download button below.")
                
            except Exception as e:
                st.error(f"Error generating FHIR data: {str(e)}")

    if st.session_state[f"{prefix}_fhir_data"]:
        st.download_button(
            label="📥 Download FHIR Patient File",
            data=st.session_state[f"{prefix}_fhir_data"],
            file_name=f"fhir_patient_{patient_id}.json",
            mime="application/json",
            key=f"{prefix}_fhir_download_json"
        )

# --- Database Visualization ---
def database_visualization(prefix):
    st.header("📊 Database Visualization and Editing")
    st.caption("Explore, visualize, and edit patient data stored in the hospital database")

    conn = sqlite3.connect(DB_PATH)
    
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
        "prescriptions": {
            "query": """
                SELECT pr.id, pr.patient_id, p.first_name || ' ' || p.last_name AS patient_name, 
                       pr.doctor, pr.diagnosis, pr.medications, pr.instructions, pr.created_at
                FROM prescriptions pr
                LEFT JOIN patients p ON pr.patient_id = p.patient_id
            """,
            "display_columns": ["id", "patient_id", "patient_name", "doctor", "diagnosis", "medications", "instructions", "created_at"],
            "editable_columns": ["patient_id", "doctor", "diagnosis", "medications", "instructions"],
            "required_columns": ["patient_id", "doctor", "diagnosis", "medications"],
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

    tabs = st.tabs(["View Database", "Edit Database"])

    with tabs[0]:
        for table, config in table_queries.items():
            with st.expander(f"Table: {table.capitalize()}"):
                df = pd.read_sql_query(config["query"], conn)
                
                st.markdown(f"**Records**: {len(df)}")
                
                st.subheader(f"{table.capitalize()} Data")
                st.dataframe(df[config["display_columns"]], use_container_width=True)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download {table}.csv",
                    data=csv,
                    file_name=f"{table}_data.csv",
                    mime="text/csv",
                    key=f"{prefix}_download_{table}"
                )

    with tabs[1]:
        st.subheader("Edit Database Records")
        table_to_edit = st.selectbox("Select Table to Edit", list(table_queries.keys()), key=f"{prefix}_edit_table_select")
        
        config = table_queries[table_to_edit]
        primary_key = config["primary_key"]
        
        df = pd.read_sql_query(config["query"], conn)
        if df.empty:
            st.warning(f"No records found in {table_to_edit} table.")
            conn.close()
            return
        
        record_ids = df[primary_key].astype(str).tolist()
        selected_record_id = st.selectbox(f"Select Record ({primary_key})", record_ids, key=f"{prefix}_edit_{table_to_edit}_id")
        
        query = f"SELECT * FROM {table_to_edit} WHERE {primary_key} = ?"
        record = pd.read_sql_query(query, conn, params=(selected_record_id,)).iloc[0]
        
        with st.form(f"{prefix}_edit_{table_to_edit}_form"):
            st.markdown(f"**Editing {table_to_edit.capitalize()} Record ({primary_key}: {selected_record_id})**")
            form_inputs = {}
            
            for col in config["editable_columns"]:
                if col == "patient_id":
                    patients_df = pd.read_sql_query("SELECT patient_id, first_name || ' ' || last_name AS patient_name FROM patients", conn)
                    patient_options = {row["patient_name"]: row["patient_id"] for _, row in patients_df.iterrows()}
                    selected_name = st.selectbox(
                        "Patient",
                        list(patient_options.keys()),
                        index=list(patient_options.values()).index(record[col]) if record[col] in patient_options.values() else 0,
                        key=f"{prefix}_{table_to_edit}_{col}"
                    )
                    form_inputs[col] = patient_options[selected_name]
                elif col == "gender":
                    form_inputs[col] = st.selectbox(col.capitalize(), ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(record[col]) if record[col] in ["Male", "Female", "Other"] else 0, key=f"{prefix}_{table_to_edit}_{col}")
                elif col == "dob":
                    form_inputs[col] = st.date_input(col.capitalize(), value=datetime.strptime(record[col], "%Y-%m-%d"), min_value=datetime(1900, 1, 1), max_value=datetime.today(), key=f"{prefix}_{table_to_edit}_{col}")
                elif col == "date":
                    form_inputs[col] = st.date_input(col.capitalize(), value=datetime.strptime(record[col], "%Y-%m-%d"), min_value=datetime.today(), key=f"{prefix}_{table_to_edit}_{col}")
                elif col == "time":
                    if table_to_edit == "appointments":
                        doctor = record["doctor"]
                        time_slots = generate_time_slots(doctor)
                        if not time_slots:
                            st.error("Invalid doctor availability. Please update doctor first.")
                            conn.close()
                            return
                        form_inputs[col] = st.selectbox(col.capitalize(), time_slots, index=time_slots.index(record[col]) if record[col] in time_slots else 0, key=f"{prefix}_{table_to_edit}_{col}")
                    else:
                        form_inputs[col] = st.text_input(col.capitalize(), value=record[col], key=f"{prefix}_{table_to_edit}_{col}")
                elif col == "department":
                    form_inputs[col] = st.selectbox(col.capitalize(), ["General Medicine", "Cardiology", "Orthopedics", "Pediatrics", "Neurology"], index=["General Medicine", "Cardiology", "Orthopedics", "Pediatrics", "Neurology"].index(record[col]) if record[col] in ["General Medicine", "Cardiology", "Orthopedics", "Pediatrics", "Neurology"] else 0, key=f"{prefix}_{table_to_edit}_{col}")
                elif col == "doctor" and table_to_edit == "appointments":
                    department = record["department"]
                    doctors_by_department = {
                        "General Medicine": ["Dr. Smith (9 AM - 12 PM)", "Dr. Arjun (2 PM - 5 PM)", "Dr. Kiran (10 AM - 1 PM)"],
                        "Cardiology": ["Dr. Rakesh (10 AM - 1 PM)", "Dr. Meera (1 PM - 4 PM)", "Dr. Arvind (3 PM - 6 PM)"],
                        "Orthopedics": ["Dr. Leena (11 AM - 2 PM)", "Dr. Rajiv (3 PM - 6 PM)", "Dr. Divya (9 AM - 12 PM)"],
                        "Pediatrics": ["Dr. Fatima (9 AM - 11 AM)", "Dr. Sunil (1 PM - 3 PM)", "Dr. Neeraj (3 PM - 5 PM)"],
                        "Neurology": ["Dr. Nisha (1 PM - 4 PM)", "Dr. Vikram (10 AM - 12 PM)", "Dr. Sneha (2 PM - 5 PM)"]
                    }
                    form_inputs[col] = st.selectbox(col.capitalize(), doctors_by_department.get(department, []), index=doctors_by_department[department].index(record[col]) if record[col] in doctors_by_department[department] else 0, key=f"{prefix}_{table_to_edit}_{col}")
                elif col == "items":
                    form_inputs[col] = st.text_area(col.capitalize(), value=json.dumps(json.loads(record[col]), indent=2), key=f"{prefix}_{table_to_edit}_{col}")
                elif col in ["total", "covered", "payable"]:
                    form_inputs[col] = st.number_input(col.capitalize(), min_value=0.0, value=float(record[col]), step=0.01, key=f"{prefix}_{table_to_edit}_{col}")
                else:
                    form_inputs[col] = st.text_area(col.capitalize(), value=record[col], key=f"{prefix}_{table_to_edit}_{col}")
            
            col1, col2 = st.columns(2)
            with col1:
                submit_update = st.form_submit_button("💾 Update Record", type="primary")
            with col2:
                submit_delete = st.form_submit_button("🗑️ Delete Record", type="secondary")
            
            if submit_update:
                for col in config["required_columns"]:
                    if not form_inputs.get(col):
                        st.error(f"{col.capitalize()} is required.")
                        conn.close()
                        return
                
                if "patient_id" in form_inputs:
                    c = conn.cursor()
                    c.execute("SELECT patient_id FROM patients WHERE patient_id = ?", (form_inputs["patient_id"],))
                    if not c.fetchone():
                        st.error("Invalid patient_id. Patient does not exist.")
                        conn.close()
                        return
                
                if table_to_edit == "invoices" and "items" in form_inputs:
                    try:
                        json.loads(form_inputs["items"])
                    except json.JSONDecodeError:
                        st.error("Items must be valid JSON.")
                        conn.close()
                        return
                
                if "dob" in form_inputs:
                    form_inputs["dob"] = form_inputs["dob"].strftime("%Y-%m-%d")
                if "date" in form_inputs:
                    form_inputs["date"] = form_inputs["date"].strftime("%Y-%m-%d")
                
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
                    st.warning(f"Deleting patient {selected_record_id} will also delete all related appointments, prescriptions, and invoices. This action cannot be undone.")
                    confirm_delete = st.checkbox("Confirm deletion of patient and all related data", key=f"{prefix}_confirm_delete_{selected_record_id}")
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
    init_db()
    
    st.sidebar.title("🏥 Hospital Management")
    app_mode = st.sidebar.radio("Navigation", [
        "Patient Management",
        "Feedback Generator",
        "Appointment Scheduling",
        "Billing System",
        "FHIR Export",
        "Database Visualization"
    ], key="hospital_app_mode")
    
    prefix = "hospital"
    
    if app_mode == "Patient Management":
        patient_management(prefix)
    elif app_mode == "Feedback Generator":
        model, encode, decode = load_model(FEEDBACK_MODEL_PATH)
        if model:
            feedback_generator(model, encode, decode, prefix)
    elif app_mode == "Appointment Scheduling":
        appointment_scheduling(prefix)
    elif app_mode == "Billing System":
        billing_system(prefix)
    elif app_mode == "Prescription Generator":
        prescription_generator(prefix)
    elif app_mode == "Documentation Generator":
        documentation_generator(prefix)
    elif app_mode == "FHIR Export":
        fhir_export_page(prefix)
    elif app_mode == "Database Visualization":
        database_visualization(prefix)
