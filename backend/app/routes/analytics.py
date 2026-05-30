from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.schemas import BiomarkerTrendResponse
from app.services.auth import get_current_user
from app.services.analytics import get_biomarker_trends_data, detect_clinical_anomalies

router = APIRouter(prefix="/api/v1/analytics", tags=["Longitudinal Analytics"])

@router.get("/trends", response_model=List[Dict[str, Any]])
def get_parameter_trends(
    parameter: str = Query(..., description="Canonical biomarker name (e.g. Fasting Glucose, HbA1c, Hemoglobin)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Returns dynamic chronological readings of a specific parameter for rendering longitudinal charts in the UI.
    """
    trends = get_biomarker_trends_data(
        user_id=current_user.id,
        parameter_name=parameter,
        db=db
    )
    
    if not trends:
        raise HTTPException(
            status_code=404,
            detail=f"No historical tracking data found for biomarker '{parameter}'."
        )
        
    return trends

@router.get("/anomalies", response_model=List[Dict[str, Any]])
def get_biomarker_anomalies(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Scans historical patient trends and returns warning indicators for rapid metrics shifts.
    """
    anomalies = detect_clinical_anomalies(
        user_id=current_user.id,
        db=db
    )
    
    return anomalies
