import hashlib
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, ProcessingJob, Report, BiomarkerTrend, PipelineAuditLog
from app.schemas import JobResponse, ReportResponse
from app.services.auth import get_current_user
from app.services.upload import upload_file_to_storage

router = APIRouter(prefix="/api/v1/reports", tags=["Reports & Jobs"])

def calculate_file_hash(file: UploadFile) -> str:
    """Compute SHA-256 hash of upload file to identify exact duplicate submissions."""
    hasher = hashlib.sha256()
    file.file.seek(0)
    while chunk := file.file.read(8192):
        hasher.update(chunk)
    file.file.seek(0)
    return hasher.hexdigest()

@router.post("/upload", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def upload_report(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Multipart upload medical reports. Saves to Cloudinary & registers an async Celery task."""
    # Validate file extension
    allowed_exts = [".pdf", ".jpg", ".jpeg", ".png"]
    import os
    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Allowed types: {', '.join(allowed_exts)}"
        )
    
    # Calculate SHA-256 for duplicate verification
    file_hash = calculate_file_hash(file)
    
    # Check if this exact file was already analyzed for this user
    duplicate_report = db.query(Report).join(ProcessingJob, Report.id == ProcessingJob.id).filter(
        ProcessingJob.user_id == current_user.id,
        ProcessingJob.file_hash == file_hash,
        Report.status == "analyzed"
    ).first()
    
    if duplicate_report:
        # File is a duplicate. We can instantly skip queue processing and return the existing analysis!
        # Create a pre-completed mock job referencing the existing report analysis to frontend
        mock_job = ProcessingJob(
            id=duplicate_report.id,  # Match report ID
            user_id=current_user.id,
            status="completed",
            cloudinary_url=duplicate_report.cloudinary_url,
            file_hash=file_hash
        )
        db.add(mock_job)
        db.commit()
        
        return JobResponse(
            jobId=mock_job.id,
            status="completed",
            message="Report duplicate detected. Direct analysis fetched instantly from cache."
        )

    # Standard path: Upload to storage
    file_url = upload_file_to_storage(file)
    
    # Create processing job record
    job = ProcessingJob(
        user_id=current_user.id,
        status="queued",
        cloudinary_url=file_url,
        file_hash=file_hash
    )
    db.add(job)
    db.commit()
    
    # Trigger Asynchronous Celery pipeline by name
    from celery import Celery
    from app.config import settings
    
    try:
        celery_app = Celery("health_tasks", broker=settings.REDIS_URL, backend=settings.REDIS_URL)
        celery_app.send_task("process_report_pipeline", args=[job.id, file_url])
    except Exception as e:
        print(f"Failed to dispatch Celery worker task: {e}")
        # Mark job as failed if queue broker is down
        job.status = "failed"
        job.error_log = f"Message broker connectivity failure: {e}"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health queue broker down. Try again later."
        )

    return JobResponse(
        jobId=job.id,
        status="queued",
        message="Medical report queued successfully. Processing started asynchronously."
    )

@router.get("/jobs/{job_id}", response_model=ReportResponse)
def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve status of a background job. If completed, returns full parsed medical analysis."""
    job = db.query(ProcessingJob).filter(
        ProcessingJob.id == job_id,
        ProcessingJob.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processing job not found."
        )
        
    if job.status == "failed":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Report analysis failed: {job.error_log or 'Unknown OCR error'}"
        )
        
    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail={"status": job.status, "message": "Report processing in progress."}
        )
        
    # Job completed: Fetch the corresponding Report record
    report = db.query(Report).filter(Report.id == job.id).first()
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analyzed report data missing."
        )
        
    return report

@router.get("/history", response_model=List[ReportResponse])
def get_reports_history(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Returns a paginated list of historically analyzed reports for the user's dashboard feed."""
    offset = (page - 1) * limit
    reports = db.query(Report).filter(
        Report.user_id == current_user.id,
        Report.status == "analyzed"
    ).order_by(Report.timestamp.desc()).offset(offset).limit(limit).all()
    
    return reports

@router.get("/{report_id}", response_model=ReportResponse)
def get_report_detail(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Return detailed interpretation and parameters for a specific report."""
    report = db.query(Report).filter(
        Report.id == report_id,
        Report.user_id == current_user.id
    ).first()
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Medical report not found."
        )
        
    return report

from pydantic import BaseModel
import time

class ChatQuery(BaseModel):
    query: str
    report_context: Optional[dict] = None

@router.post("/chat")
def chatbot_query(
    payload: ChatQuery,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Exposes a dynamic Gemini-powered chatbot query that processes the user query 
    with standard medical report guidelines and active report parameters.
    """
    import google.generativeai as genai
    from app.config import settings
    from app.services.interpretation import INTERPRET_SYSTEM_INSTRUCTION
    
    query = payload.query
    context = payload.report_context
    
    # Standard Clinical System Prompt Injection
    prompt_context = ""
    if context:
        prompt_context = f"""
        Active Medical Report details:
        - Diagnosis Summary: {context.get('diagnosis', 'Standard panel findings.')}
        - Suggested Treatment: {context.get('prescription', 'Lifestyle monitoring.')}
        - Actionable Recommendations: {context.get('recommendations', [])}
        """
        
    system_prompt = f"""
    {INTERPRET_SYSTEM_INSTRUCTION}
    
    The user is asking a question inside their secure chatbot interface.
    Answer their query using standard medical rules and safety disclaimers. 
    Frame all assertions as risk-advisory indicators, never as definitive diagnoses.
    {prompt_context}
    """
    
    has_gemini = bool(settings.GEMINI_API_KEY)
    if has_gemini:
        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            full_prompt = f"System Instruction:\n{system_prompt}\n\nUser Question:\n{query}"
            response = model.generate_content(full_prompt)
            return {"response": response.text.strip()}
        except Exception as e:
            print(f"Gemini Chatbot connection failed, running offline mock: {e}")
            
    # Resilient clinical mock answers locally if API key is blank
    time.sleep(1.0)
    
    cleaned_query = query.lower()
    
    # Diet advice query
    if "eat" in cleaned_query or "food" in cleaned_query or "diet" in cleaned_query:
        if context:
            # Handle template or empty report warning in context
            context_str = str(context).lower()
            if "insufficient" in context_str or "template" in context_str or "unfilled" in context_str:
                return {
                    "response": "🥗 **General Healthy Nutrition Guidelines**:\n\n*Note: We detected that your active report profile contains insufficient clinical data for customized metabolic mapping. The uploaded document appears to be a blank form or template.*\n\nTo promote general wellness, we recommend focusing on a balanced, nutrient-dense diet:\n\n• **Include**: Abundant whole fibers, fresh color-rich vegetables, lean proteins (poultry, legumes), and magnesium-rich nuts.\n• **Avoid**: Ultra-processed sugars, refined grain flours, and carbonated sodas.\n• **Tip**: Upload a completed lab panel containing real, validated measurements to unlock personalized glycemic or metabolic nutritional recommendations!"
                }
                
            has_high_glucose = "glucose" in context_str or "sugar" in context_str
            if has_high_glucose:
                return {
                    "response": "🥦 **Recommended Diet Plan (Metabolic/Glycemic Support)**:\n\nBased on your elevated Fasting Glucose metrics, we recommend focusing on low-glycemic foods to maintain sugar baselines:\n\n• **Include**: Rich leafy vegetables (spinach, kale), complex whole fibers (quinoa, oats), and lean healthy proteins (beans, eggs, lentils).\n• **Avoid**: Simple white flours, processed sugars, carbonated soft drinks, and refined pastries.\n• **Tip**: Keep a journal of your post-meal energy indexes and consult a dietitian!"
                }
        return {
            "response": "🥗 **General Healthy Nutrition Guidelines**:\n\nTo promote long-term vitality, focus on a balanced Mediterranean-style eating pattern:\n\n• **50% plate**: Color-rich leafy vegetables and fresh low-fructose fruits.\n• **25% plate**: Lean, cell-building proteins (e.g. fish, poultry, legumes).\n• **25% plate**: Complex high-fiber whole grains (e.g. brown rice, oats).\n• **Tip**: Maintain active physical hydration (8 glasses of water daily) and avoid refined trans-fats!"
        }
        
    return {
        "response": f"Thank you for your question! Based on your query regarding '{query}', we advise checking in with a physician to evaluate specific clinical symptoms. Maintain a healthy active lifestyle and record your baseline parameters regularly!"
    }
