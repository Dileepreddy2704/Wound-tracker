import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Float,
    DateTime,
    ForeignKey,
    Text,
    Integer,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.database import Base


def gen_uuid():
    return str(uuid.uuid4())


class Patient(Base):
    __tablename__ = "patients"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    # Use a non-identifying reference code rather than real patient name/PII
    reference_code = Column(String, unique=True, nullable=False, index=True)
    age = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    visits = relationship("Visit", back_populates="patient", cascade="all, delete-orphan")


class Visit(Base):
    __tablename__ = "visits"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    patient_id = Column(UUID(as_uuid=False), ForeignKey("patients.id"), nullable=False)
    visit_date = Column(DateTime, default=datetime.utcnow)
    image_path = Column(String, nullable=False)
    reference_object_diameter_cm = Column(Float, nullable=True)  # for pixel-to-cm2 calibration
    clinical_notes = Column(Text, nullable=True)

    patient = relationship("Patient", back_populates="visits")
    analysis = relationship(
        "WoundAnalysis", back_populates="visit", uselist=False, cascade="all, delete-orphan"
    )


class WoundAnalysis(Base):
    __tablename__ = "wound_analyses"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    visit_id = Column(UUID(as_uuid=False), ForeignKey("visits.id"), nullable=False, unique=True)

    mask_path = Column(String, nullable=True)          # path to saved segmentation mask
    area_px = Column(Float, nullable=True)
    area_cm2 = Column(Float, nullable=True)

    tissue_type = Column(String, nullable=True)         # granulation / necrosis / slough / mixed
    tissue_confidence = Column(Float, nullable=True)

    wound_type = Column(String, nullable=True)           # incision / laceration / abrasion / burn / avulsion / puncture
    wound_type_confidence = Column(Float, nullable=True)

    infection_risk_flag = Column(String, nullable=True)  # low / medium / high
    infection_indicators = Column(Text, nullable=True)   # JSON string of matched signals

    area_change_pct = Column(Float, nullable=True)       # vs. previous visit for same patient
    healing_trend = Column(String, nullable=True)        # improving / stable / worsening

    report_text = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    visit = relationship("Visit", back_populates="analysis")
