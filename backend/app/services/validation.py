from typing import Dict, Any, Tuple
from app.models import MasterBiomarker

# Standard physiological maximum boundaries for biomarkers to avoid OCR decimal failures (e.g., parsing 14.5 g/dL as 145 g/dL)
PHYSIOLOGICAL_SANITY_BOUNDS = {
    "Hemoglobin": {"min": 2.0, "max": 30.0, "unit": "g/dL"},
    "Fasting Glucose": {"min": 10.0, "max": 1000.0, "unit": "mg/dL"},
    "HbA1c": {"min": 3.0, "max": 25.0, "unit": "%"},
    "Vitamin D": {"min": 1.0, "max": 250.0, "unit": "ng/mL"},
    "Total Cholesterol": {"min": 30.0, "max": 1000.0, "unit": "mg/dL"}
}

def validate_biomarker_reading(
    canonical_name: str,
    value: float,
    unit: str,
    confidence: float
) -> Tuple[bool, str]:
    """
    Validates biological sanity and confidence parameters.
    Returns: Tuple of (is_valid: bool, error_message: str)
    """
    # 1. Enforce direct extraction confidence threshold (min 85%)
    if confidence < 0.85:
        return False, f"Parsed reading confidence '{confidence}' below the safety limit of 0.85"

    # 2. Block negative or empty clinical values
    if value <= 0:
        return False, "Negative or zero biological readings are physiologically invalid."

    # 3. Match against standard physiological maximum boundaries
    if canonical_name in PHYSIOLOGICAL_SANITY_BOUNDS:
        bounds = PHYSIOLOGICAL_SANITY_BOUNDS[canonical_name]
        
        # Check standard units
        if unit.lower().strip() != bounds["unit"].lower():
            # If units mismatch (e.g. mmol/L glucose), we could convert. 
            # For MVP, we flag unit mismatches to avoid corrupt charting.
            return False, f"Unit mismatch for {canonical_name}. Expected '{bounds['unit']}', got '{unit}'."
            
        # Check bounds
        if value < bounds["min"] or value > bounds["max"]:
            return False, f"Biological sanity check failed for {canonical_name}. Value {value} is outside biological extremes ({bounds['min']}-{bounds['max']} {bounds['unit']})."

    return True, "Valid"
