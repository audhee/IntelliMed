from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.models import BiomarkerTrend, MasterBiomarker

def get_biomarker_trends_data(
    user_id: str,
    parameter_name: str,
    db: Session
) -> List[Dict[str, Any]]:
    """
    Returns dynamic chronological readings of a specific parameter for graphing.
    """
    trends = db.query(BiomarkerTrend).join(
        MasterBiomarker, 
        BiomarkerTrend.master_biomarker_id == MasterBiomarker.id
    ).filter(
        BiomarkerTrend.user_id == user_id,
        MasterBiomarker.canonical_name == parameter_name
    ).order_by(BiomarkerTrend.recorded_at.asc()).all()
    
    return [
        {
            "id": t.id,
            "value": float(t.value),
            "unit": t.unit,
            "status": t.status,
            "date": t.recorded_at.strftime("%Y-%m-%d")
        }
        for t in trends
    ]

def detect_clinical_anomalies(user_id: str, db: Session) -> List[Dict[str, Any]]:
    """
    Scans patient's longitudinal metrics to identify sudden, clinically significant drops/spikes.
    E.g. drop in Hemoglobin > 1.5 g/dL between consecutive tests.
    """
    anomalies = []
    
    # Get all active biomarkers tracked for this user
    tracked_biomarkers = db.query(MasterBiomarker).join(
        BiomarkerTrend,
        MasterBiomarker.id == BiomarkerTrend.master_biomarker_id
    ).filter(BiomarkerTrend.user_id == user_id).distinct().all()
    
    for mb in tracked_biomarkers:
        # Fetch readings chronologically
        readings = db.query(BiomarkerTrend).filter(
            BiomarkerTrend.user_id == user_id,
            BiomarkerTrend.master_biomarker_id == mb.id
        ).order_by(BiomarkerTrend.recorded_at.asc()).all()
        
        if len(readings) < 2:
            continue
            
        for i in range(1, len(readings)):
            prev = readings[i-1]
            curr = readings[i]
            
            diff = float(curr.value - prev.value)
            days = (curr.recorded_at - prev.recorded_at).days or 1
            
            # Anomaly Rule A: Abrupt Hemoglobin drop (>1.5 g/dL within 60 days)
            if mb.canonical_name == "Hemoglobin" and diff < -1.5 and days <= 60:
                anomalies.append({
                    "parameter": mb.canonical_name,
                    "severity": "high",
                    "message": f"Critical drop in Hemoglobin level ({prev.value} to {curr.value} {curr.unit}) within {days} days, indicating potential acute bleed or rapid anemia progression."
                })
                
            # Anomaly Rule B: Sudden Fasting Glucose spike (>30 mg/dL within 30 days)
            elif mb.canonical_name == "Fasting Glucose" and diff > 30.0 and days <= 30:
                anomalies.append({
                    "parameter": mb.canonical_name,
                    "severity": "medium",
                    "message": f"Rapid glucose elevation spike detected ({prev.value} to {curr.value} {curr.unit}). Monitor dietary changes or consult practitioner."
                })
                
    return anomalies

def calculate_moving_average(user_id: str, parameter_name: str, db: Session, limit: int = 3) -> Optional[float]:
    """
    Calculate the baseline moving average for the last 'limit' readings.
    """
    trends = db.query(BiomarkerTrend).join(
        MasterBiomarker,
        BiomarkerTrend.master_biomarker_id == MasterBiomarker.id
    ).filter(
        BiomarkerTrend.user_id == user_id,
        MasterBiomarker.canonical_name == parameter_name
    ).order_by(BiomarkerTrend.recorded_at.desc()).limit(limit).all()
    
    if not trends:
        return None
        
    total_val = sum([float(t.value) for t in trends])
    return round(total_val / len(trends), 2)
