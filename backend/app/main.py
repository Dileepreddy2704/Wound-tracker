from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import Base, engine
from app.routers import patients, visits, analyze, images

app = FastAPI(title="AI Wound Assessment & Healing Tracker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


app.include_router(patients.router)
app.include_router(visits.router)
app.include_router(analyze.router)
app.include_router(images.router)


@app.get("/")
def root():
    return {"status": "ok", "service": "wound-tracker-api"}
