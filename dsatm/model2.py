"""
Model 2: Medical Text → Plain English
Model : google/flan-t5-base
Task  : Text-to-text generation — converts complex medical language into simple terms
Compat: transformers v5+ (AutoTokenizer + AutoModelForSeq2SeqLM)
"""

import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

# ── HuggingFace Auth ──────────────────────────────────────────────────────────
HF_TOKEN   = os.getenv("HF_TOKEN", "REPLACE_WITH_YOUR_TOKEN")
MODEL_NAME = "google/flan-t5-base"

if HF_TOKEN and HF_TOKEN != "REPLACE_WITH_YOUR_TOKEN":
    os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN

# ── Globals (loaded once) ─────────────────────────────────────────────────────
tokenizer = None
model     = None

# ── Prompt Templates ──────────────────────────────────────────────────────────
# Flan-T5 is instruction-tuned — quality depends heavily on the prompt.
# Multiple templates handle different use cases.
PROMPT_TEMPLATES = {
    "default": (
        "Explain this medical report in simple English that a patient can understand: {text}"
    ),
    "diagnosis": (
        "A doctor wrote this diagnosis. Explain it in very simple terms for the patient: {text}"
    ),
    "prescription": (
        "Explain this medical prescription in plain simple language: {text}"
    ),
    "lab_report": (
        "Explain these lab test results in simple words that a non-doctor can understand: {text}"
    ),
    "short": (
        "Summarize this medical text in one simple sentence: {text}"
    ),
}

# ── Model Loader ──────────────────────────────────────────────────────────────
def load_model():
    """
    Downloads (first run) and loads Flan-T5-base into memory.
    Call this ONCE at app startup — NOT on every request.
    """
    global tokenizer, model

    if tokenizer is not None and model is not None:
        logger.info("Model 2 already loaded, skipping.")
        return

    logger.info(f"Loading {MODEL_NAME} ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model.eval()
    logger.info("Model 2 (Flan-T5-base) loaded successfully ✓")


# ── Core Inference ────────────────────────────────────────────────────────────
MAX_INPUT_CHARS  = 1500   # Flan-T5-base encoder limit is 512 tokens (~1500 chars safe)
MAX_INPUT_TOKENS = 512

def _generate_plain_english(prompt: str, max_new_tokens: int, num_beams: int) -> str:
    """Run a single prompt through Flan-T5 and return decoded output."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_INPUT_TOKENS,
        truncation=True
    )

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.5,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def convert_to_plain_english(
    text: str,
    mode: str = "default",
    max_new_tokens: int = 150,
    num_beams: int = 4
) -> dict:
    """
    Convert complex medical text into plain English using Flan-T5.

    Args:
        text           : Medical text to simplify
        mode           : Prompt style — one of:
                         'default', 'diagnosis', 'prescription',
                         'lab_report', 'short'
        max_new_tokens : Max tokens to generate
        num_beams      : Beam search width (higher = better quality, slower)

    Returns:
        {
            "plain_english" : "Easy to understand explanation...",
            "mode_used"     : "default",
            "model"         : "google/flan-t5-base"
        }
    """
    if not text or not text.strip():
        return {"error": "Input text is empty."}

    if tokenizer is None or model is None:
        return {"error": "Model not loaded. Call load_model() first."}

    # ── Select prompt template ────────────────────────────────────────────────
    template = PROMPT_TEMPLATES.get(mode, PROMPT_TEMPLATES["default"])

    # ── Chunk if input is too long ────────────────────────────────────────────
    if len(text) > MAX_INPUT_CHARS:
        chunks  = [text[i:i + MAX_INPUT_CHARS] for i in range(0, len(text), MAX_INPUT_CHARS)]
        outputs = []
        for chunk in chunks:
            prompt = template.format(text=chunk)
            outputs.append(_generate_plain_english(prompt, max_new_tokens, num_beams))
        plain_text = " ".join(outputs)
    else:
        prompt     = template.format(text=text)
        plain_text = _generate_plain_english(prompt, max_new_tokens, num_beams)

    return {
        "plain_english" : plain_text,
        "mode_used"     : mode,
        "model"         : MODEL_NAME
    }


# ── Flask Blueprint ───────────────────────────────────────────────────────────
model2_bp = Blueprint("model2", __name__)

@model2_bp.route("/plain-english", methods=["POST"])
def plain_english_endpoint():
    """
    POST /plain-english

    Request JSON:
    {
        "text"           : "<complex medical text>",   ← required
        "mode"           : "default",                  ← optional
                           choices: default | diagnosis |
                                    prescription | lab_report | short
        "max_new_tokens" : 150,                        ← optional
        "num_beams"      : 4                           ← optional
    }

    Response JSON (success):
    {
        "plain_english" : "This means the patient had a heart attack...",
        "mode_used"     : "default",
        "model"         : "google/flan-t5-base"
    }

    Response JSON (error):
    {
        "error": "reason..."
    }
    """
    data = request.get_json(force=True, silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "Missing required field: 'text'"}), 400

    text           = data["text"]
    mode           = data.get("mode", "default")
    max_new_tokens = int(data.get("max_new_tokens", 150))
    num_beams      = int(data.get("num_beams", 4))

    if mode not in PROMPT_TEMPLATES:
        return jsonify({
            "error": f"Invalid mode '{mode}'. Choose from: {list(PROMPT_TEMPLATES.keys())}"
        }), 400

    try:
        result = convert_to_plain_english(
            text,
            mode=mode,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams
        )
        return jsonify(result), 400 if "error" in result else 200

    except Exception as e:
        logger.exception("Model 2 inference failed")
        return jsonify({"error": f"Inference error: {str(e)}"}), 500


@model2_bp.route("/plain-english/modes", methods=["GET"])
def list_modes():
    """
    GET /plain-english/modes
    Returns available prompt modes and their descriptions.
    """
    return jsonify({
        "available_modes": {
            "default"      : "General medical report simplification",
            "diagnosis"    : "Simplify a doctor's diagnosis",
            "prescription" : "Explain a medical prescription",
            "lab_report"   : "Explain lab/blood test results",
            "short"        : "One-sentence summary of medical text"
        }
    }), 200


# ── Quick standalone test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_model()

    tests = [
        {
            "mode": "default",
            "text": (
                "Patient exhibits bilateral lower lobe consolidation on chest radiograph "
                "consistent with community-acquired pneumonia. Commenced on IV amoxicillin-"
                "clavulanate and azithromycin. SpO2 92% on room air, supplemental oxygen "
                "initiated via nasal cannula at 2L/min."
            )
        },
        {
            "mode": "lab_report",
            "text": (
                "HbA1c: 9.2%, Fasting plasma glucose: 11.4 mmol/L, "
                "eGFR: 58 mL/min/1.73m², Microalbuminuria: 45 mg/24hr"
            )
        },
        {
            "mode": "short",
            "text": (
                "Echocardiogram demonstrates severe mitral regurgitation with left "
                "ventricular ejection fraction of 30%."
            )
        },
    ]

    for t in tests:
        print(f"\n── Mode: {t['mode']} ──────────────────────────────────────────")
        print(f"Input : {t['text']}")
        result = convert_to_plain_english(t["text"], mode=t["mode"])
        print(f"Output: {result['plain_english']}")