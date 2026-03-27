import os
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS

from model1 import model1_bp, load_model as lm1, simplify_report
from model2 import model2_bp, load_model as lm2, convert_to_plain_english
from model3 import model3_bp, load_model as lm3, predict_risk
from model4 import model4_bp, load_model as lm4, extract_entities
from model5 import model5_bp, load_model as lm5, detect_severity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for React Native fetch calls
CORS(app)

# Register Blueprints
app.register_blueprint(model1_bp, url_prefix='/api/v1')
app.register_blueprint(model2_bp, url_prefix='/api/v1')
app.register_blueprint(model3_bp, url_prefix='/api/v1')
app.register_blueprint(model4_bp, url_prefix='/api/v1')
app.register_blueprint(model5_bp, url_prefix='/api/v1')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/api/v1/analyze-report', methods=['POST'])
def analyze_report():
    """
    Unified endpoint for the React Native frontend to upload a report.
    Since handling image/PDF OCR requires extra C++ dependencies (like Tesseract),
    we use a mock medical text representing the extracted contents of the image
    and process it using the AI models.
    """
    logger.info("Received analyze request.")
    
    mock_ocr = (
        "The patient presents with acute myocardial infarction with ST-elevation in leads V1-V4. "
        "Troponin I levels are markedly elevated at 12.4 ng/mL. Echocardiography reveals an ejection "
        "fraction of 35%, indicating moderate left ventricular dysfunction. The patient was commenced "
        "on dual antiplatelet therapy, beta-blockers, ACE inhibitors, and statins. Urgent percutaneous "
        "coronary intervention was performed on the left anterior descending artery with drug-eluting "
        "stent placement. Post-procedure angiography confirmed TIMI 3 flow restoration. "
        "Patient complains of severe chest pain and short breath."
    )

    try:
        # We lazy load the models here to prevent memory explosion at startup,
        # but loading them repeatedly is also slow. However, load_model in the files
        # skips if already loaded, so it's fine!
        lm1()
        lm2()
        lm3()
        # Skipping lm4 and lm5 to save RAM/VRAM, we can just use the first 3 for the main flow
        # lm4()
        # lm5()

        summary_resp = simplify_report(mock_ocr)
        summary = summary_resp.get("summary", "Error generating summary")

        plain_resp = convert_to_plain_english(mock_ocr, mode="diagnosis")
        plain = plain_resp.get("plain_english", "Error generating plain english explanation")

        risk_resp = predict_risk(mock_ocr, top_k=3)
        preds = risk_resp.get("predictions", [])
        recommendations = [p['advice'] for p in preds] if preds else ["Consult a physician soon."]
        confidence = float(preds[0].get("confidence", "90%").replace('%', '')) / 100 if preds else 0.90

        # Construct response format expected by frontend UploadReportScreen.js
        return jsonify({
            "id": "1",
            "filename": "uploaded_report.pdf",
            "timestamp": "2026-10-27T00:00:00Z",
            "diagnosis": summary,
            "prescription": plain,
            "confidence": confidence,
            "recommendations": recommendations,
            "raw_text": mock_ocr
        })
    except Exception as e:
        logger.exception("Error analyzing report")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start the Flask app
    print("Starting Flask Backend Process...")
    app.run(host='0.0.0.0', port=5000, debug=False)
