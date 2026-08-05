
from __future__ import annotations

import json
import string
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "Model" / "models"
STOP_WORDS = {"and", "or", "of", "the", "with", "contains", "may", "be", "from"}
RISK_NAMES = {
    "lactose": "Lactose intolerance",
    "gluten": "Gluten / wheat allergy",
    "diabetes": "Diabetes / sugar monitoring",
    "vegan": "Vegan",
    "nuts": "Nut allergy",
    "yeast": "Yeast intolerance",
    "salicylate": "Salicylate sensitivity",
    "nickel": "Nickel sensitivity",
}


def clean_text(text):
    text = (text or "").lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(word for word in text.split() if word not in STOP_WORDS and not word.isdigit())


@lru_cache(maxsize=1)
def _assets():
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
    except ImportError as exc:
        raise ImportError("TensorFlow is required for the final DNN model.") from exc
    metadata = json.loads((MODELS_DIR / "deep_metadata.json").read_text(encoding="utf-8"))
    tokenizer = tokenizer_from_json((MODELS_DIR / "deep_tokenizer.json").read_text(encoding="utf-8"))
    return metadata, tokenizer, pad_sequences, tf.keras.models.load_model(MODELS_DIR / "best_model.keras", compile=False)


def predict_all(text):
 
    analysis, errors = predict_restrictions(text, [])
    if analysis is None:
        return {"main": {"is_main": False, "status": "ERROR", "warnings": []}}, errors
    return {
        "main": {
            "is_main": True,
            "status": analysis["status"],
            "warnings": analysis["warnings"],
        }
    }, errors


def _fallback_restrictions(text, selected):
    from .restrictions import analyze_text

    analysis = analyze_text(text, selected)
    return {
        "status": analysis["status"],
        "risk_score": analysis["risk_score"],
        "warnings": analysis["warnings"],
        "unsupported": [],
        "recommendation": analysis["recommendation"],
    }, {}


def predict_restrictions(text, selected):
    
    try:
        metadata, tokenizer, pad_sequences, model = _assets()
        sequence = pad_sequences(tokenizer.texts_to_sequences([clean_text(text)]), maxlen=int(metadata["max_length"]), padding="post")
        probabilities = model.predict(sequence, verbose=0)[0]
        threshold = float(metadata.get("threshold", 0.5))
        scores = dict(zip(metadata["risk_ids"], probabilities))
    except (FileNotFoundError, ImportError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return None, {"final_model": str(exc)}

    warnings = [
        {"restriction": risk_id, "name": RISK_NAMES[risk_id], "confidence": round(float(scores[risk_id]) * 100, 2)}
        for risk_id in selected
        if risk_id in scores and scores[risk_id] >= threshold
    ]

    if warnings:
        status, recommendation = "AVOID", "The trained DNN predicts a dietary risk. Check the package's official allergen and nutrition information."
    elif selected:
        fallback_result, fallback_errors = _fallback_restrictions(text, selected)
        if fallback_result["warnings"]:
            return fallback_result, fallback_errors
        status, recommendation = "NO_MODEL_RISK", "The trained DNN did not predict a selected dietary risk. Verify the package label before consuming it."
    else:
        status, recommendation = "NEEDS_PROFILE", "Choose one or more dietary profiles for a DNN risk prediction."

    unsupported = [risk_id for risk_id in selected if risk_id not in scores]
    return {
        "status": status,
        "risk_score": min(100, round(sum(item["confidence"] for item in warnings))),
        "warnings": warnings,
        "unsupported": unsupported,
        "recommendation": recommendation,
    }, {}
