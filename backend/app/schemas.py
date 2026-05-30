from datetime import datetime
from typing import List, Optional, Any
from pydantic import BaseModel, EmailStr, Field

# AUTH SCHEMAS
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, description="Password must be at least 6 characters")
    role: str = Field("patient", pattern="^(patient|doctor)$", description="Role must be patient or doctor")
    full_name: str = Field(..., min_length=1)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    full_name: str

# PROFILE SCHEMAS
class PatientProfileBase(BaseModel):
    full_name: str
    date_of_birth: Optional[datetime] = None
    gender: Optional[str] = None
    blood_type: Optional[str] = None

class PatientProfileResponse(PatientProfileBase):
    id: str
    user_id: str
    created_at: datetime

    class Config:
        from_attributes = True

# DYNAMIC JOB SCHEMAS
class JobResponse(BaseModel):
    jobId: str
    status: str
    message: str

    class Config:
        from_attributes = True

# BIOMARKER MAPPING SCHEMAS
class BiomarkerSchema(BaseModel):
    parameter_name: str
    value: float
    unit: str
    reference_range: Optional[str] = None
    status: str  # normal, high, low, critical

class ReportAnalysisSchema(BaseModel):
    diagnosis: str
    prescription: str
    recommendations: List[str]
    confidence: float
    biomarkers: List[BiomarkerSchema]

# REPORT RETRIEVAL SCHEMAS
class ReportResponse(BaseModel):
    id: str
    filename: str
    cloudinary_url: str
    status: str
    confidence: Optional[float] = None
    diagnosis: Optional[str] = None
    prescription: Optional[str] = None
    recommendations: Optional[Any] = None
    timestamp: datetime

    class Config:
        from_attributes = True

class BiomarkerTrendResponse(BaseModel):
    id: int
    extracted_name: str
    canonical_name: str
    value: float
    unit: str
    status: str
    recorded_at: datetime

    class Config:
        from_attributes = True
