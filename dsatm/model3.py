"""
Model 3: Disease / Risk Prediction
Model : emilyalsentzer/Bio_ClinicalBERT
Task  : Predict possible conditions from symptom text using:
          1. BERT embeddings + cosine similarity (semantic matching)
          2. Keyword-based rule layer (fast fallback + confidence boost)
Compat: transformers v5+ (AutoTokenizer + AutoModel — encoder only, no generation)

WHY THIS APPROACH:
  Bio_ClinicalBERT is a masked-language model (fill-mask), NOT a classifier.
  It has no built-in label head for disease prediction.
  Strategy: encode symptom text → compute cosine similarity against
  pre-encoded disease symptom profiles → rank and return top matches.
"""

import os
import logging
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

# ── HuggingFace Auth ──────────────────────────────────────────────────────────
HF_TOKEN   = os.getenv("HF_TOKEN", "REPLACE_WITH_YOUR_TOKEN")
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

if HF_TOKEN and HF_TOKEN != "REPLACE_WITH_YOUR_TOKEN":
    os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN

# ── Globals ───────────────────────────────────────────────────────────────────
tokenizer          = None
model              = None
disease_embeddings = {}   # { disease_name: tensor }  — built once after load

# ═════════════════════════════════════════════════════════════════════════════
# DISEASE KNOWLEDGE BASE
# Each entry: display name → {
#     "profile"  : symptom sentence fed to BERT for embedding,
#     "keywords" : fast-match keywords for confidence boosting,
#     "advice"   : plain-English next-step advice
# }
# ═════════════════════════════════════════════════════════════════════════════
DISEASE_KB = {
    "Type 2 Diabetes": {
        "profile"  : "high blood sugar frequent urination excessive thirst fatigue blurred vision slow healing wounds",
        "keywords" : ["sugar", "glucose", "urination", "thirst", "diabetic", "hba1c", "insulin", "polyuria", "polydipsia"],
        "advice"   : "Consult an endocrinologist. Fasting blood glucose and HbA1c tests recommended.",
        "severity" : "moderate"
    },
    "Hypertension": {
        "profile"  : "high blood pressure headache dizziness shortness of breath chest pain nosebleed",
        "keywords" : ["blood pressure", "hypertension", "bp", "headache", "dizziness", "palpitation"],
        "advice"   : "Monitor BP regularly. Reduce salt intake. Consult a cardiologist.",
        "severity" : "moderate"
    },
    "Pneumonia": {
        "profile"  : "cough fever chest pain shortness of breath difficulty breathing chills sweating",
        "keywords" : ["cough", "fever", "chest", "breathing", "pneumonia", "consolidation", "sputum", "chills"],
        "advice"   : "Seek immediate medical attention. Chest X-ray and blood culture recommended.",
        "severity" : "high"
    },
    "Anemia": {
        "profile"  : "fatigue weakness pale skin shortness of breath cold hands feet dizziness low hemoglobin",
        "keywords" : ["fatigue", "pale", "hemoglobin", "anemia", "weakness", "iron", "dizzy", "breathless"],
        "advice"   : "Complete blood count (CBC) test recommended. Consult a hematologist.",
        "severity" : "moderate"
    },
    "Hypothyroidism": {
        "profile"  : "fatigue weight gain cold sensitivity constipation dry skin hair loss slow heart rate depression",
        "keywords" : ["thyroid", "tsh", "weight gain", "cold", "fatigue", "constipation", "hair loss", "hypothyroid"],
        "advice"   : "TSH and T4 blood tests recommended. Consult an endocrinologist.",
        "severity" : "low"
    },
    "Urinary Tract Infection": {
        "profile"  : "burning urination frequent urination cloudy urine pelvic pain strong odor urine fever",
        "keywords" : ["burning", "urination", "urine", "uti", "bladder", "pelvic", "cloudy", "frequency"],
        "advice"   : "Urine culture test recommended. Stay hydrated. Consult a physician.",
        "severity" : "low"
    },
    "Acute Myocardial Infarction": {
        "profile"  : "severe chest pain radiating arm jaw sweating nausea shortness of breath heart attack",
        "keywords" : ["chest pain", "heart attack", "myocardial", "troponin", "ecg", "radiating", "jaw pain", "sweating"],
        "advice"   : "EMERGENCY — call ambulance immediately. ECG and troponin levels needed.",
        "severity" : "critical"
    },
    "GERD / Acid Reflux": {
        "profile"  : "heartburn acid reflux regurgitation chest discomfort sour taste throat burning after eating",
        "keywords" : ["heartburn", "acid", "reflux", "gerd", "regurgitation", "throat", "burning", "stomach"],
        "advice"   : "Avoid spicy/fatty foods. Endoscopy may be advised. Consult a gastroenterologist.",
        "severity" : "low"
    },
    "Asthma": {
        "profile"  : "wheezing shortness of breath chest tightness coughing night breathlessness inhaler",
        "keywords" : ["wheeze", "asthma", "inhaler", "breathless", "tight chest", "cough", "spirometry"],
        "advice"   : "Pulmonary function test recommended. Avoid triggers. Consult a pulmonologist.",
        "severity" : "moderate"
    },
    "Migraine": {
        "profile"  : "severe headache throbbing pain nausea vomiting light sensitivity sound sensitivity aura",
        "keywords" : ["headache", "migraine", "throbbing", "nausea", "light", "aura", "vomiting", "sensitivity"],
        "advice"   : "Keep a headache diary. Avoid triggers. Consult a neurologist.",
        "severity" : "moderate"
    },
    "Depression / Anxiety": {
        "profile"  : "persistent sadness hopelessness loss of interest fatigue sleep problems anxiety worry panic",
        "keywords" : ["depression", "anxiety", "sad", "hopeless", "panic", "worry", "sleep", "fatigue", "mood"],
        "advice"   : "Consult a psychiatrist or psychologist. Do not self-medicate.",
        "severity" : "moderate"
    },
    "COVID-19 / Viral Infection": {
        "profile"  : "fever cough loss of taste smell fatigue body ache sore throat runny nose covid",
        "keywords" : ["covid", "coronavirus", "loss of taste", "loss of smell", "fever", "cough", "body ache", "viral"],
        "advice"   : "RT-PCR test recommended. Isolate and consult a physician.",
        "severity" : "moderate"
    },
}

SEVERITY_COLOR = {
    "low"      : "🟢",
    "moderate" : "🟡",
    "high"     : "🔴",
    "critical" : "🚨"
}


# ── Model Loader ──────────────────────────────────────────────────────────────
def load_model():
    """
    Load Bio_ClinicalBERT and pre-compute disease profile embeddings.
    Call ONCE at app startup.
    """
    global tokenizer, model, disease_embeddings

    if tokenizer is not None and model is not None:
        logger.info("Model 3 already loaded, skipping.")
        return

    logger.info(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model     = AutoModel.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model.eval()
    logger.info("Model 3 (Bio_ClinicalBERT) loaded ✓")

    logger.info("Pre-computing disease profile embeddings ...")
    for disease, info in DISEASE_KB.items():
        disease_embeddings[disease] = _embed(info["profile"])
    logger.info(f"Embeddings ready for {len(disease_embeddings)} diseases ✓")


# ── Embedding Helper ──────────────────────────────────────────────────────────
def _embed(text: str) -> torch.Tensor:
    """
    Encode text using Bio_ClinicalBERT CLS token → normalised vector.
    Uses [CLS] token representation (standard for sentence similarity).
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs    = model(**inputs)
        cls_vector = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        normalised = F.normalize(cls_vector, p=2, dim=1)  # unit vector
    return normalised.squeeze(0)


# ── Keyword Boost ─────────────────────────────────────────────────────────────
def _keyword_boost(text_lower: str, disease: str) -> float:
    """
    Count matching keywords → return a 0.0–0.15 confidence boost.
    Keeps semantic score as primary signal; keywords just tip close calls.
    """
    keywords = DISEASE_KB[disease]["keywords"]
    hits     = sum(1 for kw in keywords if kw in text_lower)
    return min(hits * 0.03, 0.15)   # max +15% boost


# ── Core Inference ────────────────────────────────────────────────────────────
def predict_risk(text: str, top_k: int = 3) -> dict:
    """
    Predict possible diseases/risks from symptom text.

    Args:
        text  : Patient symptom description (free text)
        top_k : Number of top predictions to return (default 3)

    Returns:
        {
            "predictions": [
                {
                    "rank"       : 1,
                    "disease"    : "Type 2 Diabetes",
                    "confidence" : "78.4%",
                    "severity"   : "moderate",
                    "severity_icon": "🟡",
                    "advice"     : "Consult an endocrinologist..."
                },
                ...
            ],
            "disclaimer": "..."
        }
    """
    if not text or not text.strip():
        return {"error": "Input text is empty."}

    if tokenizer is None or model is None:
        return {"error": "Model not loaded. Call load_model() first."}

    # ── Embed the symptom input ───────────────────────────────────────────────
    symptom_vec = _embed(text)
    text_lower  = text.lower()

    # ── Score each disease ────────────────────────────────────────────────────
    scores = {}
    for disease, profile_vec in disease_embeddings.items():
        cosine_sim    = F.cosine_similarity(symptom_vec.unsqueeze(0),
                                            profile_vec.unsqueeze(0)).item()
        keyword_boost = _keyword_boost(text_lower, disease)
        scores[disease] = cosine_sim + keyword_boost

    # ── Sort and take top-k ───────────────────────────────────────────────────
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # ── Normalise scores to 0–100% for readability ────────────────────────────
    top_score = ranked[0][1] if ranked else 1.0
    predictions = []
    for rank, (disease, score) in enumerate(ranked, start=1):
        info       = DISEASE_KB[disease]
        confidence = min((score / max(top_score, 0.001)) * 100, 99.9)
        predictions.append({
            "rank"          : rank,
            "disease"       : disease,
            "confidence"    : f"{confidence:.1f}%",
            "severity"      : info["severity"],
            "severity_icon" : SEVERITY_COLOR[info["severity"]],
            "advice"        : info["advice"]
        })

    return {
        "predictions" : predictions,
        "disclaimer"  : (
            "⚠️ This is an AI-based screening tool only. "
            "It does NOT replace professional medical diagnosis. "
            "Always consult a qualified healthcare provider."
        )
    }


# ── Flask Blueprint ───────────────────────────────────────────────────────────
model3_bp = Blueprint("model3", __name__)

@model3_bp.route("/predict-risk", methods=["POST"])
def predict_risk_endpoint():
    """
    POST /predict-risk

    Request JSON:
    {
        "text"  : "high sugar, fatigue, frequent urination",   ← required
        "top_k" : 3                                            ← optional (default 3, max 5)
    }

    Response JSON (success):
    {
        "predictions": [
            {
                "rank"          : 1,
                "disease"       : "Type 2 Diabetes",
                "confidence"    : "91.2%",
                "severity"      : "moderate",
                "severity_icon" : "🟡",
                "advice"        : "Consult an endocrinologist..."
            }
        ],
        "disclaimer": "⚠️ This is an AI-based screening tool only..."
    }
    """
    data = request.get_json(force=True, silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "Missing required field: 'text'"}), 400

    text  = data["text"]
    top_k = min(int(data.get("top_k", 3)), 5)   # cap at 5

    try:
        result = predict_risk(text, top_k=top_k)
        return jsonify(result), 400 if "error" in result else 200

    except Exception as e:
        logger.exception("Model 3 inference failed")
        return jsonify({"error": f"Inference error: {str(e)}"}), 500


@model3_bp.route("/predict-risk/diseases", methods=["GET"])
def list_diseases():
    """
    GET /predict-risk/diseases
    Returns all diseases in the knowledge base (useful for Android UI info screen).
    """
    return jsonify({
        "total"    : len(DISEASE_KB),
        "diseases" : [
            {
                "name"     : d,
                "severity" : info["severity"],
                "icon"     : SEVERITY_COLOR[info["severity"]]
            }
            for d, info in DISEASE_KB.items()
        ]
    }), 200


# ── Quick standalone test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_model()

    test_cases = [
        "high blood sugar, fatigue, frequent urination, excessive thirst",
        "severe chest pain radiating to left arm, sweating, nausea",
        "wheezing at night, shortness of breath, tight chest",
        "persistent sadness, loss of interest, sleep problems, hopeless",
        "burning urination, cloudy urine, lower abdominal pain",
    ]

    for symptom_text in test_cases:
        print(f"\n── Input ──────────────────────────────────────────────────────")
        print(f"Symptoms : {symptom_text}")
        result = predict_risk(symptom_text, top_k=3)
        print("Top Predictions:")
        for p in result["predictions"]:
            print(f"  {p['rank']}. {p['severity_icon']} {p['disease']} — {p['confidence']} | {p['advice']}")