import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

MAX_INPUT_LEN = 20
MAX_OUTPUT_LEN = 30
BEAM_WIDTH = 3

@st.cache_resource
def load_resources():
    model = load_model('seq2seq_model.h5')

    def load_tokenizer(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    symptom_tokenizer = load_tokenizer('symptom_tokenizer.pkl')
    prescription_tokenizer = load_tokenizer('prescription_tokenizer.pkl')

    reverse_prescription_index = {idx: word for word, idx in prescription_tokenizer.word_index.items()}
    reverse_prescription_index[prescription_tokenizer.word_index.get('<start>', 1)] = '<start>'
    reverse_prescription_index[prescription_tokenizer.word_index.get('<end>', 2)] = '<end>'

    return model, symptom_tokenizer, prescription_tokenizer, reverse_prescription_index

def clean_symptoms(text):
    text = text.lower()
    text = re.sub(r'\d{1,3}\s*-\s*year\s*-\s*old', '', text)
    text = re.sub(r'\d{1,3}\s*year\s*old', '', text)
    text = re.sub(r'\b(male|female|man|woman|boy|girl|other)\b', '', text)
    text = re.sub(r'with', '', text)
    text = re.sub(r'[^a-zA-Z, ]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def beam_search_decode(model, symptom_seq_padded, start_token_id, end_token_id, beam_width=3):
    sequences = [[list(), 0.0]]
    for _ in range(MAX_OUTPUT_LEN):
        all_candidates = []
        for seq, score in sequences:
            decoder_input = np.zeros((1, MAX_OUTPUT_LEN))
            if len(seq) > 0:
                decoder_input[0, :len(seq)] = seq
                decoder_input[0, len(seq)] = start_token_id
            else:
                decoder_input[0, 0] = start_token_id

            preds = model.predict([symptom_seq_padded, decoder_input], verbose=0)[0, len(seq), :]
            top_tokens = np.argsort(preds)[-beam_width:]

            for token_id in top_tokens:
                candidate = [seq + [token_id], score - np.log(preds[token_id] + 1e-10)]
                all_candidates.append(candidate)

        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_width]

        completed_sequences = [seq for seq, score in sequences if seq[-1] == end_token_id]
        if completed_sequences:
            return completed_sequences[0]

    return sequences[0][0]

def generate_prescription(model, symptom_tokenizer, prescription_tokenizer, reverse_prescription_index, age, gender, symptoms):
    cleaned_symptoms = clean_symptoms(symptoms)
    symptom_seq = symptom_tokenizer.texts_to_sequences([cleaned_symptoms])
    if not symptom_seq or not symptom_seq[0]:
        return "No valid symptoms detected. Please try different wording."

    symptom_seq_padded = pad_sequences(symptom_seq, maxlen=MAX_INPUT_LEN, padding='post')
    start_token_id = prescription_tokenizer.word_index.get('<start>', 1)
    end_token_id = prescription_tokenizer.word_index.get('<end>', 2)

    generated_sequence = beam_search_decode(model, symptom_seq_padded, start_token_id, end_token_id)

    prescription_words = [
        reverse_prescription_index.get(token_id, '')
        for token_id in generated_sequence
        if token_id not in [start_token_id, end_token_id]
    ]

    if not prescription_words:
        return "Could not generate valid prescription. Please try different symptoms."

    return (
        f"Patient: {age}-year-old {gender}\n"
        f"Diagnosis: {cleaned_symptoms.capitalize()}\n\n"
        f"Prescription:\n{' '.join(prescription_words).replace(' ,', ',').replace(' .', '.').capitalize()}\n\n"
        "Directions: Take as directed\nDuration: 5-7 days"
    )

def run():
    st.title("🏥 AI Prescription Generator")

    model, symptom_tokenizer, prescription_tokenizer, reverse_prescription_index = load_resources()

    with st.form("prescription_form"):
        st.subheader("Patient Information")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=40)
        with col2:
            gender = st.selectbox("Gender", ["male", "female", "other"])

        symptoms = st.text_area("Symptoms", "sore throat, fever, cough")
        st.info("🔹 Enter symptoms separated by commas (e.g., 'fever, cough'). No full sentences needed.")
        submitted = st.form_submit_button("Generate Prescription")

    if submitted:
        if not symptoms.strip():
            st.warning("Please enter symptoms")
        else:
            with st.spinner("Generating prescription..."):
                try:
                    prescription = generate_prescription(model, symptom_tokenizer, prescription_tokenizer, reverse_prescription_index, age, gender, symptoms)
                    st.success("✅ Generated Prescription")
                    st.text_area("Prescription", prescription, height=250)

                    st.download_button(
                        label="💾 Download Prescription as TXT",
                        data=prescription.encode('utf-8'),
                        file_name="prescription.txt",
                        mime="text/plain"
                    )

                    if st.button("➕ New Patient"):
                        st.experimental_rerun()

                    st.warning("""
                    **Disclaimer**: This AI-generated prescription is for educational purposes only.
                    Always consult with a licensed healthcare provider for medical advice.
                    """)

                except Exception as e:
                    st.error(f"Generation failed: {str(e)}")
                    st.info("Tips: Use common symptoms. Avoid rare medical terms.")
