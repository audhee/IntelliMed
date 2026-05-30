import os
import sys
import time
from datetime import datetime
from celery import Celery
from sqlalchemy.orm import Session

# Add current folder to path to enable local app module imports in background process
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import settings
from app.database import SessionLocal
from app.models import ProcessingJob, Report, BiomarkerTrend, PipelineAuditLog, User
from app.services.ocr import extract_report_data
from app.services.normalization import normalize_biomarker
from app.services.validation import validate_biomarker_reading
from app.services.analytics import detect_clinical_anomalies, calculate_moving_average
from app.services.interpretation import generate_safe_clinical_summary

# Initialize Celery app instance
celery_app = Celery("health_tasks", broker=settings.REDIS_URL, backend=settings.REDIS_URL)

# Configure concurrency to avoid standard multithread overhead on simple resources
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=2
)

@celery_app.task(name="process_report_pipeline")
def process_report_pipeline(job_id: str, file_url: str) -> str:
    """
    Decoupled Asynchronous Background Job Task.
    Performs OCR extraction, normalizes aliases, conducts sanity validates, 
    evaluates longitudinal trends, generates safe clinical context, and commits to DB.
    """
    db: Session = SessionLocal()
    start_pipeline_time = time.time()
    
    # Grab Job Record
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    if not job:
        db.close()
        return "ERROR: Processing job not found."
        
    try:
        # ----------------------------------------------------
        # STAGE 1: Preprocessing & Worker Startup
        # ----------------------------------------------------
        job.status = "preprocessing"
        db.commit()
        
        audit_start = PipelineAuditLog(
            job_id=job.id,
            step="PREPROCESS",
            status="success",
            latency_ms=10,
            details={"message": "Pipeline worker accepted task, validation complete."}
        )
        db.add(audit_start)
        db.commit()
        
        # ----------------------------------------------------
        # STAGE 2: OCR & Gemini Text Extraction
        # ----------------------------------------------------
        job.status = "ocr_extracting"
        db.commit()
        
        ocr_start_time = time.time()
        raw_extraction = extract_report_data(file_url)
        ocr_latency = int((time.time() - ocr_start_time) * 1000)
        
        # Save OCR raw payload to audit log
        audit_ocr = PipelineAuditLog(
            job_id=job.id,
            step="OCR",
            status="success",
            confidence=raw_extraction.get("confidence", 1.0),
            prompt_version=raw_extraction.get("_prompt_version", "v1.0.0"),
            latency_ms=ocr_latency,
            details={"raw_data": raw_extraction}
        )
        db.add(audit_ocr)
        db.commit()
        
        # ----------------------------------------------------
        # STAGE 3: Confidence-Threshold Validation Check
        # ----------------------------------------------------
        confidence = float(raw_extraction.get("confidence", 1.0))
        if confidence < 0.85:
            # Report lacks visual readability. Fail job gracefully and notify frontend.
            job.status = "failed"
            job.error_log = f"Extraction confidence threshold failed ({confidence} < 0.85). Image quality is insufficient."
            db.commit()
            
            audit_fail = PipelineAuditLog(
                job_id=job.id,
                step="OCR",
                status="failure",
                confidence=confidence,
                details={"error": "Confidence validation threshold failed."}
            )
            db.add(audit_fail)
            db.commit()
            db.close()
            return f"FAILED: Confidence threshold invalid."

        # ----------------------------------------------------
        # STAGE 4: Biomarker Normalization & Alias Translation
        # ----------------------------------------------------
        norm_start_time = time.time()
        extracted_biomarkers = raw_extraction.get("biomarkers", [])
        
        validated_metrics = []
        rejected_metrics = []
        
        for eb in extracted_biomarkers:
            extracted_name = eb.get("parameter_name", "")
            raw_value = float(eb.get("value", 0.0))
            raw_unit = eb.get("unit", "")
            raw_status = eb.get("status", "normal")
            
            # Run normalizer to match clinical aliases
            norm_res = normalize_biomarker(extracted_name, db)
            if not norm_res:
                rejected_metrics.append({
                    "parameter": extracted_name,
                    "reason": "Alias mapping failed. Parameter not found in master biomarker registry."
                })
                continue
                
            canonical_name, master_record = norm_res
            
            # ----------------------------------------------------
            # STAGE 5: Biological Sanity Boundary Check
            # ----------------------------------------------------
            is_biologically_sane, sanity_err = validate_biomarker_reading(
                canonical_name=canonical_name,
                value=raw_value,
                unit=raw_unit,
                confidence=confidence
            )
            
            if not is_biologically_sane:
                rejected_metrics.append({
                    "parameter": extracted_name,
                    "canonical": canonical_name,
                    "reason": f"Sanity boundary rejected: {sanity_err}"
                })
                continue
                
            # Entry is valid and standard. Store standardized metadata!
            validated_metrics.append({
                "master_record": master_record,
                "extracted_name": extracted_name,
                "canonical_name": canonical_name,
                "value": raw_value,
                "unit": raw_unit,
                "status": raw_status
            })
            
        norm_latency = int((time.time() - norm_start_time) * 1000)
        
        audit_norm = PipelineAuditLog(
            job_id=job.id,
            step="NORMALIZATION",
            status="success",
            latency_ms=norm_latency,
            details={
                "validated_count": len(validated_metrics),
                "rejected_count": len(rejected_metrics),
                "rejected_details": rejected_metrics
            }
        )
        db.add(audit_norm)
        db.commit()

        # ----------------------------------------------------
        # STAGE 5B: Empty/Template Detection & Measurable Pipeline-Grounded Confidence
        # ----------------------------------------------------
        total_extracted = len(extracted_biomarkers)
        total_validated = len(validated_metrics)
        
        # Measure pipeline signals: OCR visual certainty, validation rate, and absolute scale
        validation_rate = (total_validated / total_extracted) if total_extracted > 0 else 0.0
        
        # Density and quality evaluation
        is_empty_or_template = False
        template_rejection_reason = ""
        
        # Rule 1: Falls below minimum threshold of validated biomarkers (need at least 2 for charting/analytics)
        if total_validated < 2:
            is_empty_or_template = True
            template_rejection_reason = f"Validated biomarker density is too low ({total_validated} < 2). Form may be blank or unfilled."
        # Rule 2: High normalization/validation failure rate on a structured file
        elif total_extracted > 0 and validation_rate < 0.4:
            is_empty_or_template = True
            template_rejection_reason = f"High normalization failure rate ({total_validated}/{total_extracted} < 40%). Document is likely an unfilled template or invalid sheet."
            
        # Calculate dynamic, grounded pipeline confidence based on actual measurable metrics
        if total_extracted > 0:
            # Formula: 30% raw OCR confidence + 40% normalization success rate + 30% absolute scale factor (max 1.0)
            calculated_confidence = (confidence * 0.3) + (validation_rate * 0.4) + (min(total_validated / 5.0, 1.0) * 0.3)
        else:
            calculated_confidence = 0.0
            is_empty_or_template = True
            template_rejection_reason = "No structured biomarkers found in the uploaded document."

        # Bind calculated confidence between 0.0 and 1.0
        calculated_confidence = max(0.0, min(calculated_confidence, 1.0))
        
        # Override confidence to a low scale if it is a template to be transparent to users
        if is_empty_or_template:
            calculated_confidence = min(calculated_confidence, 0.35)

        audit_density = PipelineAuditLog(
            job_id=job.id,
            step="VALIDATION",
            status="success",
            confidence=calculated_confidence,
            latency_ms=10,
            details={
                "is_empty_or_template": is_empty_or_template,
                "rejection_reason": template_rejection_reason,
                "total_extracted": total_extracted,
                "total_validated": total_validated,
                "calculated_confidence": calculated_confidence
            }
        )
        db.add(audit_density)
        db.commit()

        # ----------------------------------------------------
        # STAGE 6: Compute Longitudinal Trends & Diagnostics
        # ----------------------------------------------------
        analytics_start_time = time.time()
        
        historical_summary = ""
        historical_segments = []
        
        # Only perform trend analytics if document is completed
        if not is_empty_or_template:
            for vm in validated_metrics:
                c_name = vm["canonical_name"]
                m_avg = calculate_moving_average(job.user_id, c_name, db, limit=3)
                if m_avg:
                    historical_segments.append(f"{c_name} (Current: {vm['value']}{vm['unit']}, 3-Test Moving Average: {m_avg}{vm['unit']})")
                    
            anomalies = detect_clinical_anomalies(job.user_id, db)
            anomaly_text = ""
            if anomalies:
                anomaly_text = "Clinical Anomalies Detected:\n" + "\n".join([a["message"] for a in anomalies])
                
            historical_summary = "\n".join(historical_segments)
            if anomaly_text:
                historical_summary += f"\n\n{anomaly_text}"
            
        analytics_latency = int((time.time() - analytics_start_time) * 1000)
        
        # ----------------------------------------------------
        # STAGE 7: Generate Safe Clinical Summary (Prompt Versioned)
        # ----------------------------------------------------
        interpret_start_time = time.time()
        
        if is_empty_or_template:
            # HALT Stage: Stop interpretation. Return strictly constrained empty/template fallback
            clinical_interpretation = {
                "diagnosis": "Insufficient clinical data detected for reliable analysis.",
                "prescription": "No active laboratory measurements detected. The uploaded document appears to be an unfilled medical form, blank template, prescription, or scan lacking valid structured results. To generate a health analysis, please upload a completed laboratory diagnostic blood panel.",
                "recommendations": [],
                "prompt_version": "safe_fallback_v1.0"
            }
        else:
            # Normal parsing path: feed only the validated structured biomarkers
            biomarker_payload = [
                {
                    "parameter_name": vm["canonical_name"],
                    "value": vm["value"],
                    "unit": vm["unit"],
                    "status": vm["status"]
                }
                for vm in validated_metrics
            ]
            
            clinical_interpretation = generate_safe_clinical_summary(
                biomarkers=biomarker_payload,
                historical_summary=historical_summary
            )
            
        interpret_latency = int((time.time() - interpret_start_time) * 1000)
        
        audit_interpret = PipelineAuditLog(
            job_id=job.id,
            step="INTERPRETATION",
            status="success",
            prompt_version=clinical_interpretation.get("prompt_version", "v1.2.0"),
            latency_ms=interpret_latency,
            details={"advisory_generated": not is_empty_or_template, "is_empty_or_template": is_empty_or_template}
        )
        db.add(audit_interpret)
        db.commit()

        # ----------------------------------------------------
        # STAGE 8: Commit Everything to Supabase DB & Finalize Job
        # ----------------------------------------------------
        # Create corresponding Report record (matching the Job ID!)
        new_report = Report(
            id=job.id,  # Match report ID to processing job ID!
            user_id=job.user_id,
            filename=f"analyzed_report_{job.id[:8]}.pdf" if file_url.endswith(".pdf") else f"analyzed_report_{job.id[:8]}.jpg",
            cloudinary_url=file_url,
            status="analyzed",
            confidence=calculated_confidence,  # Use pipeline-grounded score!
            diagnosis=clinical_interpretation.get("diagnosis", "Report successfully parsed."),
            prescription=clinical_interpretation.get("prescription", "Lifestyle monitoring advised."),
            recommendations=clinical_interpretation.get("recommendations", []),
            raw_analysis_json=raw_extraction
        )
        db.add(new_report)
        db.flush()
        
        # Save validated biomarker trend points!
        for vm in validated_metrics:
            trend = BiomarkerTrend(
                user_id=job.user_id,
                report_id=new_report.id,
                master_biomarker_id=vm["master_record"].id,
                extracted_name=vm["extracted_name"],
                value=vm["value"],
                unit=vm["unit"],
                status=vm["status"],
                recorded_at=new_report.timestamp  # Recorded on analysis timestamp date
            )
            db.add(trend)
            
        # Complete Job status!
        job.status = "completed"
        db.commit()
        
        total_latency = int((time.time() - start_pipeline_time) * 1000)
        print(f"SUCCESS: Asynchronous queue pipeline executed successfully for job {job.id} in {total_latency}ms.")
        
    except Exception as err:
        db.rollback()
        print(f"CRITICAL ERROR in celery background pipeline: {err}")
        job.status = "failed"
        job.error_log = f"Unhandled background pipeline exception: {err}"
        db.commit()
        
        audit_err = PipelineAuditLog(
            job_id=job.id,
            step="ANALYTICS",
            status="failure",
            details={"error_message": str(err)}
        )
        db.add(audit_err)
        db.commit()
        
    finally:
        db.close()
        
    return "SUCCESS: Pipeline executed."
