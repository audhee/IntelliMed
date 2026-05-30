import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Numeric, JSON, Text, ForeignKey, BigInteger, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base

# Helper to handle UUID columns cross-compatibly (Postgres native UUID vs string fallback for SQLite testing)
def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="patient")  # patient, doctor
    created_at = Column(DateTime, default=datetime.utcnow)

    profile = relationship("PatientProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="user", cascade="all, delete-orphan")
    jobs = relationship("ProcessingJob", back_populates="user", cascade="all, delete-orphan")
    biomarker_trends = relationship("BiomarkerTrend", back_populates="user", cascade="all, delete-orphan")

class PatientProfile(Base):
    __tablename__ = "patient_profiles"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    full_name = Column(String(255), nullable=False)
    date_of_birth = Column(DateTime, nullable=True)
    gender = Column(String(50), nullable=True)
    blood_type = Column(String(10), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="profile")

class MasterBiomarker(Base):
    __tablename__ = "master_biomarkers"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    canonical_name = Column(String(100), unique=True, nullable=False, index=True)  # e.g., "Hemoglobin"
    aliases = Column(JSON, nullable=False)  # JSON array: ["Hb", "HGB", "Hemoglobin", "hgb"]
    standard_unit = Column(String(50), nullable=False)  # e.g., "g/dL"
    normal_range_min = Column(Numeric, nullable=True)
    normal_range_max = Column(Numeric, nullable=True)
    critical_range_min = Column(Numeric, nullable=True)
    critical_range_max = Column(Numeric, nullable=True)
    category = Column(String(100), nullable=False)  # e.g., "Complete Blood Count", "Lipid Panel"

    trends = relationship("BiomarkerTrend", back_populates="master_biomarker")

class Report(Base):
    __tablename__ = "reports"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(255), nullable=False)
    cloudinary_url = Column(Text, nullable=False)
    status = Column(String(50), nullable=False, default="analyzed")  # pending, analyzed, failed
    confidence = Column(Numeric, nullable=True)  # Overall OCR/AI parse confidence
    diagnosis = Column(Text, nullable=True)  # Medically safe AI clinical interpretation summary
    prescription = Column(Text, nullable=True)  # Medically safe AI prescriptions & advices
    recommendations = Column(JSON, nullable=True)  # JSON list of actionable items (strings)
    raw_analysis_json = Column(JSON, nullable=True)  # Complete raw JSON stored for audit reproducibility
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="reports")
    trends = relationship("BiomarkerTrend", back_populates="report", cascade="all, delete-orphan")

class BiomarkerTrend(Base):
    __tablename__ = "biomarker_trends"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    report_id = Column(String(36), ForeignKey("reports.id", ondelete="CASCADE"), nullable=False)
    master_biomarker_id = Column(String(36), ForeignKey("master_biomarkers.id", ondelete="RESTRICT"), nullable=False)
    extracted_name = Column(String(100), nullable=False)  # Original parsed label (e.g., "HGB")
    value = Column(Numeric, nullable=False)  # Validated canonical numerical reading
    unit = Column(String(50), nullable=False)  # Canonical standard unit
    status = Column(String(50), nullable=False)  # normal, high, low, critical_high, critical_low
    recorded_at = Column(DateTime, nullable=False, index=True)

    user = relationship("User", back_populates="biomarker_trends")
    report = relationship("Report", back_populates="trends")
    master_biomarker = relationship("MasterBiomarker", back_populates="trends")

class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(50), nullable=False, default="queued")  # queued, preprocessing, ocr_extracting, completed, failed
    cloudinary_url = Column(Text, nullable=True)
    file_hash = Column(String(64), nullable=True)  # SHA-256 duplicate check
    error_log = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="jobs")
    audit_logs = relationship("PipelineAuditLog", back_populates="job", cascade="all, delete-orphan")

class PipelineAuditLog(Base):
    __tablename__ = "pipeline_audit_logs"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("processing_jobs.id", ondelete="CASCADE"), nullable=False)
    step = Column(String(50), nullable=False)  # PREPROCESS, OCR, NORMALIZATION, INTERPRETATION, ANALYTICS
    status = Column(String(50), nullable=False)  # success, failure
    confidence = Column(Numeric, nullable=True)  # Stage parse confidence
    prompt_version = Column(String(50), nullable=True)  # Registered prompt version used (e.g. "v1.0.0")
    latency_ms = Column(Integer, nullable=True)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("ProcessingJob", back_populates="audit_logs")
