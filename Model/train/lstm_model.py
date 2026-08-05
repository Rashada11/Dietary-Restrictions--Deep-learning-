import os
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Visualization
import matplotlib.pyplot as plt

# Import preprocessing functions
from preprocessing import clean_text, build_ingredients_text

warnings.filterwarnings('ignore')

# ==============================
# 1. LOAD DATASET
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(PROJECT_ROOT / "Model" / "dataset" / "food_ingredients_and_allergens.csv")

# ==============================
# 2. CLEAN DATA
# ==============================
df = df.replace('None', '')
df = df.fillna('')
df = df[df['Prediction'].str.strip().ne('')].copy()
df = build_ingredients_text(df)
df['cleaned_text'] = df['ingredients_text'].apply(clean_text)

# ==============================
# 3. LABEL ENCODING
# ==============================
encoder = LabelEncoder()
df['encoded_label'] = encoder.fit_transform(df['Prediction'])
num_classes = len(encoder.classes_)
y = to_categorical(df['encoded_label'])

# ==============================
# 4. TOKENIZATION + PADDING
# ==============================
max_words = 5000
max_length = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['cleaned_text'])
sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
X = pad_sequences(sequences, maxlen=max_length, padding='post')

# ==============================
# 5. TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# 6. MODEL BUILDERS
# ==============================
def build_lstm_model(max_words, max_length, num_classes):
    model = Sequential([
        Embedding(max_words, 128, input_length=max_length),
        LSTM(64),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def build_cnn_model(max_words, max_length, num_classes):
    model = Sequential([
        Embedding(max_words, 128, input_length=max_length),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def build_dnn_model(max_words, max_length, num_classes):
    model = Sequential([
        Embedding(max_words, 128, input_length=max_length),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# ==============================
# 7. TRAIN & EVALUATE ALL MODELS
# ==============================
models = {
    "LSTM": build_lstm_model(max_words, max_length, num_classes),
    "CNN": build_cnn_model(max_words, max_length, num_classes),
    "DNN": build_dnn_model(max_words, max_length, num_classes)
}

histories = {}
results = {}

for name, model in models.items():
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"\nTraining {name} model...\n")
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    histories[name] = history
    results[name] = {"accuracy": acc, "loss": loss}
    print(f"{name} Accuracy: {acc:.4f}")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(f"\nCLASSIFICATION REPORT for {name}:\n")
    print(classification_report(y_true, y_pred, target_names=encoder.classes_))

# ==============================
# 8. PICK BEST MODEL
# ==============================

best_model_name = "DNN"
best_model = models[best_model_name]
print(f"\nBest model is {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f} and loss {results[best_model_name]['loss']:.4f}")

# ==============================
# 9. SAVE MODELS AND PICK FINAL BEST
# ==============================
os.makedirs(PROJECT_ROOT / "Model" / "models", exist_ok=True)
for name, model in models.items():
    model.save(PROJECT_ROOT / "Model" / "models" / f"{name.lower()}_model.keras")

best_model.save(PROJECT_ROOT / "Model" / "models" / "best_model.keras")
print("\nBEST MODEL SAVED SUCCESSFULLY AS best_model.keras!")

# ==============================
# 10. PLOT COMPARISON
# ==============================
os.makedirs(PROJECT_ROOT / "Model" / "plots", exist_ok=True)
best_history = histories[best_model_name]

plt.figure(figsize=(10, 6))
plt.plot(best_history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(best_history.history['val_accuracy'], label='Val Accuracy', marker='o', linestyle='--')
plt.title(f"{best_model_name} Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "Model" / "plots" / "deep_learning_accuracy.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(best_history.history['loss'], label='Train Loss', marker='o')
plt.plot(best_history.history['val_loss'], label='Val Loss', marker='o', linestyle='--')
plt.title(f"{best_model_name} Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "Model" / "plots" / "deep_learning_curves.png", dpi=150)
plt.close()
