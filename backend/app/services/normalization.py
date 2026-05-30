from typing import Tuple, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.models import MasterBiomarker

# Standard seed database definitions of biomarkers for immediate offline lookup and seeding!
# Ensures that even if the SQL database is clean, the normalization engine functions flawlessly!
BIOMARKER_CANONICAL_REGISTRY = [
    {
        "canonical_name": "Hemoglobin",
        "aliases": ["hb", "hgb", "hemoglobin", "haemoglobin", "blood count hb"],
        "standard_unit": "g/dL",
        "normal_min": 13.5,
        "normal_max": 17.5,
        "category": "Complete Blood Count"
    },
    {
        "canonical_name": "Fasting Glucose",
        "aliases": ["fasting glucose", "fbs", "fasting blood sugar", "glucose fasting", "fpg"],
        "standard_unit": "mg/dL",
        "normal_min": 70.0,
        "normal_max": 100.0,
        "category": "Metabolic Panel"
    },
    {
        "canonical_name": "HbA1c",
        "aliases": ["hba1c", "glycated hemoglobin", "a1c", "hemoglobin a1c"],
        "standard_unit": "%",
        "normal_min": 4.0,
        "normal_max": 5.6,
        "category": "Metabolic Panel"
    },
    {
        "canonical_name": "Vitamin D",
        "aliases": ["vitamin d", "25-hydroxy vitamin d", "vit d", "vitamin d3", "25(oh)d"],
        "standard_unit": "ng/mL",
        "normal_min": 30.0,
        "normal_max": 100.0,
        "category": "Vitamin Panel"
    },
    {
        "canonical_name": "Total Cholesterol",
        "aliases": ["total cholesterol", "cholesterol total", "cholesterol", "tc"],
        "standard_unit": "mg/dL",
        "normal_min": 100.0,
        "normal_max": 199.0,
        "category": "Lipid Panel"
    }
]

def seed_master_biomarkers(db: Session) -> None:
    """Helper to populate the database with clinical canonical biomarkers if empty."""
    for entry in BIOMARKER_CANONICAL_REGISTRY:
        existing = db.query(MasterBiomarker).filter(
            MasterBiomarker.canonical_name == entry["canonical_name"]
        ).first()
        
        if not existing:
            new_biomarker = MasterBiomarker(
                canonical_name=entry["canonical_name"],
                aliases=entry["aliases"],
                standard_unit=entry["standard_unit"],
                normal_range_min=entry["normal_min"],
                normal_range_max=entry["normal_max"],
                category=entry["category"]
            )
            db.add(new_biomarker)
    db.commit()

def normalize_biomarker(extracted_name: str, db: Session) -> Optional[Tuple[str, MasterBiomarker]]:
    """
    Standardize lab parameter variations into standard canonical medical identifiers.
    E.g. maps 'HGB' -> 'Hemoglobin' and links its database record.
    Returns: Tuple of (canonical_name, MasterBiomarker object) or None.
    """
    cleaned_name = extracted_name.strip().lower()
    
    # 1. First attempt exact or substring match in active database registry
    # Ensure database is seeded before lookup
    seed_master_biomarkers(db)
    
    all_biomarkers = db.query(MasterBiomarker).all()
    
    for mb in all_biomarkers:
        # Check canonical name match
        if mb.canonical_name.lower() == cleaned_name:
            return mb.canonical_name, mb
        
        # Check matches within JSON alias arrays
        aliases = mb.aliases if isinstance(mb.aliases, list) else []
        if cleaned_name in [alias.lower() for alias in aliases]:
            return mb.canonical_name, mb
            
    # 2. Fallback fuzzy substring matching on aliases
    for mb in all_biomarkers:
        aliases = mb.aliases if isinstance(mb.aliases, list) else []
        for alias in aliases:
            if alias.lower() in cleaned_name or cleaned_name in alias.lower():
                return mb.canonical_name, mb
                
    return None
