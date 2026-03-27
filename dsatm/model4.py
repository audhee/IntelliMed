"""
Model 4: Medical Named Entity Recognition (NER)
Model : d4data/biomedical-ner-all
Task  : Extract diseases, drugs, tests, anatomy from medical text
Compat: transformers v5+ (AutoTokenizer + AutoModelForTokenClassification)
"""

import os
import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

# ── HuggingFace Config ────────────────────────────────────────────────────────
MODEL_NAME = "d4data/biomedical-ner-all"
HF_TOKEN   = os.getenv("HF_TOKEN")  # ✅ NEVER hardcode

# ── Globals ───────────────────────────────────────────────────────────────────
tokenizer = None
model     = None
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model Loader ──────────────────────────────────────────────────────────────
def load_model():
    """
    Load NER model once at startup.
    """
    global tokenizer, model

    if tokenizer is not None and model is not None:
        logger.info("Model 4 already loaded, skipping.")
        return

    logger.info(f"Loading {MODEL_NAME} ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model     = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, token=HF_TOKEN)

    model.to(device)
    model.eval()

    logger.info(f"Model 4 (NER) loaded successfully ✓ on {device}")


# ── Core Inference ────────────────────────────────────────────────────────────
MAX_INPUT_TOKENS = 512

def _predict_entities(text: str):
    """
    Run token classification and return raw predictions.
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
    predictions = torch.argmax(logits, dim=2)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [model.config.id2label[p.item()] for p in predictions[0]]

    return tokens, labels


# ── Entity Aggregation (VERY IMPORTANT) ───────────────────────────────────────
def aggregate_entities(tokens, labels):
    """
    Convert token-level predictions into readable entity spans.
    Handles B- / I- tagging properly.
    """
    entities = []
    current_entity = []
    current_label = None

    for token, label in zip(tokens, labels):

        if label.startswith("B-"):
            # Save previous entity
            if current_entity:
                entities.append({
                    "entity": current_label,
                    "text": tokenizer.convert_tokens_to_string(current_entity)
                })
                current_entity = []

            current_label = label[2:]
            current_entity.append(token)

        elif label.startswith("I-") and current_entity:
            current_entity.append(token)

        else:
            if current_entity:
                entities.append({
                    "entity": current_label,
                    "text": tokenizer.convert_tokens_to_string(current_entity)
                })
                current_entity = []
                current_label = None

    # अंतिम entity
    if current_entity:
        entities.append({
            "entity": current_label,
            "text": tokenizer.convert_tokens_to_string(current_entity)
        })

    return entities


# ── Clean Output Formatting ───────────────────────────────────────────────────
def format_entities(entities):
    """
    Group entities by type (DISEASE, DRUG, TEST, etc.)
    """
    grouped = {}

    for ent in entities:
        label = ent["entity"]
        text  = ent["text"].strip()

        if not text:
            continue

        if label not in grouped:
            grouped[label] = []

        if text not in grouped[label]:
            grouped[label].append(text)

    return grouped


# ── Public Function ───────────────────────────────────────────────────────────
def extract_entities(text: str):
    """
    Main function to extract structured entities.
    """
    if not text or not text.strip():
        return {"error": "Input text is empty."}

    if tokenizer is None or model is None:
        return {"error": "Model not loaded. Call load_model() first."}

    tokens, labels = _predict_entities(text)
    raw_entities   = aggregate_entities(tokens, labels)
    formatted      = format_entities(raw_entities)

    return {
        "entities": formatted,
        "total_entities": sum(len(v) for v in formatted.values())
    }


# ── Flask Blueprint ───────────────────────────────────────────────────────────
model4_bp = Blueprint("model4", __name__)

@model4_bp.route("/extract-entities", methods=["POST"])
def extract_entities_endpoint():
    """
    POST /extract-entities

    Request:
        { "text": "Patient has diabetes and is taking metformin..." }

    Response:
        {
          "entities": {
              "DISEASE": ["diabetes"],
              "DRUG": ["metformin"]
          },
          "total_entities": 2
        }
    """
    data = request.get_json(force=True, silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "Missing required field: 'text'"}), 400

    try:
        result = extract_entities(data["text"])

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result), 200

    except Exception as e:
        logger.exception("NER inference failed")
        return jsonify({"error": f"Inference error: {str(e)}"}), 500


# ── Standalone Test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_model()

    sample = (
        "The patient has diabetes and hypertension. "
        "He is prescribed metformin and lisinopril. "
        "Blood glucose test shows elevated levels."
    )

    print("\n── Input ─────────────────────────────────────")
    print(sample)

    print("\n── Extracted Entities ────────────────────────")
    print(extract_entities(sample))