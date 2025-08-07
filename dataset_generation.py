import pandas as pd
import random

# Symptoms pool
symptoms_pool = [
    "fever", "cough", "sore throat", "headache", "vomiting", "diarrhea",
    "chest pain", "shortness of breath", "abdominal pain", "back pain",
    "skin rash", "itching", "joint pain", "muscle pain", "ear pain",
    "runny nose", "sneezing", "eye redness", "dizziness", "fatigue",
    "weight loss", "loss of appetite", "frequent urination", "burning urination",
    "palpitations", "swelling", "difficulty breathing", "productive cough",
    "body ache", "night sweats", "loss of smell", "loss of taste"
]

# Medicines pool
medicines_pool = [
    "paracetamol 500mg tid", "azithromycin 500mg od", "amoxicillin 500mg tid",
    "ibuprofen 400mg bid", "ondansetron 4mg tid", "oral rehydration salts",
    "diclofenac 50mg bid", "cetirizine 10mg od", "salbutamol inhaler prn",
    "metronidazole 400mg tid", "hydrocortisone cream bid", "loratadine 10mg od",
    "pantoprazole 40mg od", "atorvastatin 10mg od", "multivitamin daily",
    "iron supplements od", "levocetirizine 5mg od", "doxycycline 100mg bid"
]

# Investigations pool
investigations_pool = [
    "cbc blood test", "ecg test", "x-ray chest", "urine analysis",
    "blood sugar test", "throat swab culture", "stool test", "spirometry",
    "serum electrolytes", "liver function test", "kidney function test",
    "dengue test", "malaria parasite test"
]

# Generate data
symptoms_column = []
prescriptions_column = []

for _ in range(1000):
    symptoms = random.sample(symptoms_pool, random.randint(2, 4))
    meds = random.sample(medicines_pool, random.randint(2, 3))
    tests = random.sample(investigations_pool, random.randint(1, 2))
    
    symptoms_column.append(", ".join(symptoms))
    prescriptions_column.append(", ".join(meds + tests))

# Save CSV
df = pd.DataFrame({
    "symptoms": symptoms_column,
    "prescription": prescriptions_column
})

df.to_csv("symptoms_prescriptions.csv", index=False)

print("✅ Dataset generated successfully!")
