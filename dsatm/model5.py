"""
Model 5: Sentiment / Severity Detection
Model : distilbert-base-uncased-finetuned-sst-2-english
Task  : Classify medical report as Normal / Needs Attention
Compat: transformers v5+ (AutoTokenizer + AutoModelForSequenceClassification)
"""

import os
import logging
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
HF_TOKEN   = os.getenv("HF_TOKEN")  # ✅ don't hardcode

# ── Globals ───────────────────────────────────────────────────────────────────
tokenizer = None
model     = None
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model Loader ──────────────────────────────────────────────────────────────
def load_model():
    """
    Load model once at startup.
    """
    global tokenizer, model

    if tokenizer is not None and model is not None:
        logger.info("Model 5 already loaded, skipping.")
        return

    logger.info(f"Loading {MODEL_NAME} ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, token=HF_TOKEN)

    model.to(device)
    model.eval()

    logger.info(f"Model 5 (Severity) loaded successfully ✓ on {device}")


# ── Core Inference ────────────────────────────────────────────────────────────
MAX_INPUT_TOKENS = 512

def _predict(text: str):
    """
    Run classification and return logits + probabilities.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    logits = outputs.logits
    probs  = F.softmax(logits, dim=1)

    predicted_class_id = torch.argmax(probs, dim=1).item()
    confidence         = probs[0][predicted_class_id].item()

    label = model.config.id2label[predicted_class_id]

    return label, confidence


# ── Severity Mapping (Important Layer) ─────────────────────────────────────────
def map_to_medical_severity(label: str):
    """
    Convert generic sentiment → medical severity
    """
    if label == "POSITIVE":
        return "Normal"
    elif label == "NEGATIVE":
        return "Needs Attention"
    else:
        return "Unknown"


# ── Public Function ───────────────────────────────────────────────────────────
def detect_severity(text: str):
    """
    Main function for severity detection.
    """
    if not text or not text.strip():
        return {"error": "Input text is empty."}

    if tokenizer is None or model is None:
        return {"error": "Model not loaded. Call load_model() first."}

    label, confidence = _predict(text)
    severity          = map_to_medical_severity(label)

    return {
        "severity": severity,
        "raw_label": label,
        "confidence": round(confidence, 4)
    }


# ── Flask Blueprint ───────────────────────────────────────────────────────────
model5_bp = Blueprint("model5", __name__)

@model5_bp.route("/detect-severity", methods=["POST"])
def detect_severity_endpoint():
    """
    POST /detect-severity

    Request:
        { "text": "Patient shows severe chest pain and abnormal ECG." }

    Response:
        {
          "severity": "Needs Attention",
          "raw_label": "NEGATIVE",
          "confidence": 0.97
        }
    """
    data = request.get_json(force=True, silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "Missing required field: 'text'"}), 400

    try:
        result = detect_severity(data["text"])

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result), 200

    except Exception as e:
        logger.exception("Severity detection failed")
        return jsonify({"error": f"Inference error: {str(e)}"}), 500


# ── Standalone Test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_model()

    sample_1 = "The patient is stable with normal vitals and no major concerns."
    sample_2 = "Patient shows severe chest pain, high troponin levels, and abnormal ECG."

    print("\n── Input 1 ─────────────────────────────────")
    print(sample_1)
    print("Output:", detect_severity(sample_1))

    print("\n── Input 2 ─────────────────────────────────")
    print(sample_2)
    print("Output:", detect_severity(sample_2))