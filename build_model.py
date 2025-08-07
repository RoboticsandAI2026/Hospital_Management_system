import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
data = pd.read_csv('symptoms_prescriptions.csv')

symptoms = data['symptoms'].astype(str).tolist()
prescriptions = ['<start> ' + text + ' <end>' for text in data['prescription'].astype(str).tolist()]

# Tokenize
symptom_tokenizer = Tokenizer()
symptom_tokenizer.fit_on_texts(symptoms)
symptom_sequences = symptom_tokenizer.texts_to_sequences(symptoms)
symptom_sequences = pad_sequences(symptom_sequences, maxlen=20, padding='post')

prescription_tokenizer = Tokenizer(filters='')
prescription_tokenizer.fit_on_texts(prescriptions)
prescription_sequences = prescription_tokenizer.texts_to_sequences(prescriptions)
prescription_sequences = pad_sequences(prescription_sequences, maxlen=30, padding='post')

# Save tokenizers
with open('symptom_tokenizer.pkl', 'wb') as f:
    pickle.dump(symptom_tokenizer, f)

with open('prescription_tokenizer.pkl', 'wb') as f:
    pickle.dump(prescription_tokenizer, f)

# Prepare data
X = symptom_sequences
y = prescription_sequences

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
vocab_size_symptoms = len(symptom_tokenizer.word_index) + 1
vocab_size_prescriptions = len(prescription_tokenizer.word_index) + 1
embedding_dim = 128
latent_dim = 256

encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(vocab_size_symptoms, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(vocab_size_prescriptions, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = TimeDistributed(Dense(vocab_size_prescriptions, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy')

earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('seq2seq_model.h5', monitor='val_loss', save_best_only=True)

model.fit(
    [X_train, y_train[:, :-1]],
    y_train[:, 1:],
    epochs=30,
    batch_size=64,
    validation_data=([X_val, y_val[:, :-1]], y_val[:, 1:]),
    callbacks=[earlystop, checkpoint]
)

# Save final model
model.save('seq2seq_model.h5')
print("✅ Model trained and saved!")
