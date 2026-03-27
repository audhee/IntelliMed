"""
Model 1: Medical Report Simplifier
Model : facebook/bart-large-cnn
Task  : Summarization → plain English summary of medical reports
Compat: transformers v5+ (uses AutoTokenizer + AutoModelForSeq2SeqLM directly)
"""

import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

# ── HuggingFace Auth ─────────────────────────────────────────────────────────
HF_TOKEN   = os.getenv("HF_TOKEN", "REPLACE_WITH_YOUR_TOKEN")
MODEL_NAME = "facebook/bart-large-cnn"

if HF_TOKEN and HF_TOKEN != "REPLACE_WITH_YOUR_TOKEN":
    os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN

# ── Globals (loaded once) ─────────────────────────────────────────────────────
tokenizer = None
model     = None

# ── Model Loader ─────────────────────────────────────────────────────────────
def load_model():
    """
    Downloads (first run) and loads the BART model into memory.
    Call this ONCE at app startup — NOT on every request.
    """
    global tokenizer, model

    if tokenizer is not None and model is not None:
        logger.info("Model 1 already loaded, skipping.")
        return

    logger.info(f"Loading {MODEL_NAME} ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model.eval()
    logger.info("Model 1 (BART) loaded successfully ✓")


# ── Core Inference ────────────────────────────────────────────────────────────
MAX_INPUT_CHARS  = 3000
MAX_INPUT_TOKENS = 1024

def _summarize_chunk(text: str, max_new_tokens: int, min_new_tokens: int) -> str:
    """Summarize a single chunk that fits within BART's token limit."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_INPUT_TOKENS,
        truncation=True
    )

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def simplify_report(text: str, max_new_tokens: int = 130, min_new_tokens: int = 30) -> dict:
    """
    Simplify / summarize a medical report.

    Args:
        text           : Raw medical report text
        max_new_tokens : Max tokens to generate in output
        min_new_tokens : Min tokens to generate in output

    Returns:
        { "summary": "...", "chunks_processed": 1 }
    """
    if not text or not text.strip():
        return {"error": "Input text is empty."}

    if tokenizer is None or model is None:
        return {"error": "Model not loaded. Call load_model() first."}

    if len(text) > MAX_INPUT_CHARS:
        chunks    = [text[i:i + MAX_INPUT_CHARS] for i in range(0, len(text), MAX_INPUT_CHARS)]
        summaries = [_summarize_chunk(c, max_new_tokens, min_new_tokens) for c in chunks]
        return {"summary": " ".join(summaries), "chunks_processed": len(chunks)}

    return {"summary": _summarize_chunk(text, max_new_tokens, min_new_tokens), "chunks_processed": 1}


# ── Flask Blueprint ───────────────────────────────────────────────────────────
model1_bp = Blueprint("model1", __name__)

@model1_bp.route("/simplify-report", methods=["POST"])
def simplify_report_endpoint():
    """
    POST /simplify-report

    Request JSON:
        { "text": "...", "max_new_tokens": 130, "min_new_tokens": 30 }

    Response JSON:
        { "summary": "Plain English summary...", "chunks_processed": 1 }
    """
    data = request.get_json(force=True, silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "Missing required field: 'text'"}), 400

    text           = data["text"]
    max_new_tokens = int(data.get("max_new_tokens", 130))
    min_new_tokens = int(data.get("min_new_tokens", 30))

    if min_new_tokens >= max_new_tokens:
        return jsonify({"error": "'min_new_tokens' must be less than 'max_new_tokens'"}), 400

    try:
        result = simplify_report(text, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)
        return jsonify(result), 400 if "error" in result else 200
    except Exception as e:
        logger.exception("Model 1 inference failed")
        return jsonify({"error": f"Inference error: {str(e)}"}), 500


# ── Quick standalone test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_model()

    sample = (
        "The patient presents with acute myocardial infarction with ST-elevation in leads V1-V4. "
        "Troponin I levels are markedly elevated at 12.4 ng/mL. Echocardiography reveals an ejection "
        "fraction of 35%, indicating moderate left ventricular dysfunction. The patient was commenced "
        "on dual antiplatelet therapy, beta-blockers, ACE inhibitors, and statins. Urgent percutaneous "
        "coronary intervention was performed on the left anterior descending artery with drug-eluting "
        "stent placement. Post-procedure angiography confirmed TIMI 3 flow restoration."
    )

    print("\n── Input ──────────────────────────────────────────────────────")
    print(sample)
    print("\n── Output (Simplified) ────────────────────────────────────────")
    print(simplify_report(sample))