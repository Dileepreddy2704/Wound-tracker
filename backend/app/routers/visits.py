import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db import models
from app.schemas import schemas

router = APIRouter(prefix="/visits", tags=["visits"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/", response_model=schemas.VisitOut)
async def create_visit(
    patient_id: str = Form(...),
    reference_object_diameter_cm: float | None = Form(None),
    clinical_notes: str | None = Form(None),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    patient = db.query(models.Patient).filter_by(id=patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    ext = os.path.splitext(image.filename)[1] or ".jpg"
    filename = f"{uuid.uuid4()}{ext}"
    save_path = os.path.join(UPLOAD_DIR, filename)

    with open(save_path, "wb") as f:
        f.write(await image.read())

    db_visit = models.Visit(
        patient_id=patient_id,
        image_path=save_path,
        reference_object_diameter_cm=reference_object_diameter_cm,
        clinical_notes=clinical_notes,
    )
    db.add(db_visit)
    db.commit()
    db.refresh(db_visit)
    return db_visit


@router.get("/{patient_id}", response_model=list[schemas.VisitOut])
def list_visits_for_patient(patient_id: str, db: Session = Depends(get_db)):
    return (
        db.query(models.Visit)
        .filter_by(patient_id=patient_id)
        .order_by(models.Visit.visit_date)
        .all()
    )
