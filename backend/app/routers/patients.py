from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db import models
from app.schemas import schemas

router = APIRouter(prefix="/patients", tags=["patients"])


@router.post("/", response_model=schemas.PatientOut)
def create_patient(patient: schemas.PatientCreate, db: Session = Depends(get_db)):
    existing = db.query(models.Patient).filter_by(reference_code=patient.reference_code).first()
    if existing:
        raise HTTPException(status_code=400, detail="reference_code already exists")
    db_patient = models.Patient(**patient.model_dump())
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient


@router.get("/", response_model=list[schemas.PatientOut])
def list_patients(db: Session = Depends(get_db)):
    return db.query(models.Patient).all()


@router.get("/{patient_id}", response_model=schemas.PatientOut)
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    patient = db.query(models.Patient).filter_by(id=patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient
