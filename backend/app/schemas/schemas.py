from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class PatientCreate(BaseModel):
    reference_code: str
    age: Optional[int] = None
    notes: Optional[str] = None


class PatientOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    reference_code: str
    age: Optional[int]
    notes: Optional[str]
    created_at: datetime


class VisitCreate(BaseModel):
    patient_id: str
    reference_object_diameter_cm: Optional[float] = None
    clinical_notes: Optional[str] = None


class VisitOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    patient_id: str
    visit_date: datetime
    image_path: str
    reference_object_diameter_cm: Optional[float]
    clinical_notes: Optional[str]


class WoundAnalysisOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    visit_id: str
    mask_path: Optional[str]        # path to the saved segmentation mask PNG
    area_px: Optional[float]
    area_cm2: Optional[float]
    tissue_type: Optional[str]
    tissue_confidence: Optional[float]
    wound_type: Optional[str]           # incision / laceration / abrasion / burn / avulsion / puncture
    wound_type_confidence: Optional[float]
    infection_risk_flag: Optional[str]
    infection_indicators: Optional[str]
    area_change_pct: Optional[float]
    healing_trend: Optional[str]
    report_text: Optional[str]
    created_at: datetime
