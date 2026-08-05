"""Train Logistic Regression and Random Forest models from the project CSV."""
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import build_ingredients_text, clean_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "Model" / "dataset" / "food_ingredients_and_allergens.csv"
MODELS_DIR = PROJECT_ROOT / "Model" / "models"


def make_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("classifier", classifier),
    ])


def load_training_data():
    if not DATASET.exists():
        raise FileNotFoundError(f"Training CSV is missing: {DATASET}")
    frame = pd.read_csv(DATASET).replace("None", "").fillna("")
    required = {"Prediction", "Main Ingredient", "Sweetener", "Fat/Oil", "Seasoning", "Allergens"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    frame = frame[frame["Prediction"].astype(str).str.strip().ne("")].copy()
    frame = build_ingredients_text(frame)
    return frame["ingredients_text"].map(clean_text), frame["Prediction"].str.strip()


def main():
    texts, labels = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    models = {
        "logistic_regression": make_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        ),
        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
            )
        ),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(f"\n{name} accuracy: {accuracy_score(y_test, predictions):.3f}")
        print(classification_report(y_test, predictions, zero_division=0))
        output = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, output)
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
