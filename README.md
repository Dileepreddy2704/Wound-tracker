# 🩹 AI Wound Assessment & Healing Tracker

A full-stack clinical AI application that takes wound photographs and runs an automated pipeline:

**Image upload → MedSAM segmentation → area measurement → tissue classification → infection risk assessment → visit-over-visit healing trend → clinical report**

Built with **FastAPI + PostgreSQL** (backend), **MedSAM** (segmentation), **OpenCV HSV analysis** (tissue classification), and **Streamlit** (frontend dashboard).

---

## 📸 Demo

| Upload & Analyze | Mask Overlay | Clinical Report |
|---|---|---|
| Upload wound photo via the Streamlit dashboard | MedSAM segments the wound region; a red overlay shows exactly what was detected | Full tissue composition breakdown + infection risk + healing trend |

**Example output for a deep traumatic wound:**
```
Tissue type     : Granulation  (83% confidence)
  Composition breakdown:
    granulation    ████████████████     83%
    slough         ██                    8%
    necrosis       █                     5%
Infection risk  : MEDIUM
  • Slough/fibrin areas detected (8%) — monitor and consider debridement
  • ⚠ Wound depth cannot be assessed from 2D photography — clinical examination required
Healing trend   : baseline visit — no prior visit to compare
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
│  Patient mgmt │ Image upload │ Results dashboard        │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP (localhost:8000)
┌──────────────────────▼──────────────────────────────────┐
│                    FastAPI Backend                       │
│  /patients  /visits  /analyze  /uploads                 │
│                                                         │
│  ┌─────────────────┐   ┌──────────────────────────────┐ │
│  │  MedSAM (SAM    │   │  HSV Tissue Classifier       │ │
│  │  ViT-B)         │   │  + Rule-based Infection Risk │ │
│  │  segmentation.py│   │  analyze.py                  │ │
│  └─────────────────┘   └──────────────────────────────┘ │
└──────────────────────┬──────────────────────────────────┘
                       │ SQLAlchemy
┌──────────────────────▼──────────────────────────────────┐
│              PostgreSQL                                  │
│  patients │ visits │ wound_analyses                     │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Layout

```
wound-tracker/
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI entrypoint, CORS, router registration
│   │   ├── db/
│   │   │   ├── database.py       # SQLAlchemy engine + session factory
│   │   │   └── models.py         # Patient, Visit, WoundAnalysis ORM models
│   │   ├── schemas/
│   │   │   └── schemas.py        # Pydantic request/response models
│   │   ├── routers/
│   │   │   ├── patients.py       # POST/GET /patients
│   │   │   ├── visits.py         # POST /visits  (multipart image upload)
│   │   │   ├── analyze.py        # POST /analyze/{visit_id}  (full pipeline)
│   │   │   └── images.py         # GET /uploads/{filename}  (serve images/masks)
│   │   └── models/
│   │       ├── segmentation.py   # MedSAM wrapper: crop → HSV localize → SAM predict
│   │       └── measurement.py    # px→cm², area change %, healing trend
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .env.example
│   └── uploads/                  # Saved wound images + mask PNGs (git-ignored)
├── frontend/
│   ├── streamlit_app.py          # Full Streamlit dashboard
│   └── requirements.txt
├── ml/
│   ├── download_medsam_checkpoint.py  # Download MedSAM weights
│   ├── train_segmentation.py          # Fine-tuning skeleton (SegFormer / MedSAM)
│   ├── data/                          # Dataset prep scripts
│   └── checkpoints/                   # Model weights (git-ignored)
└── docker-compose.yml            # Postgres + backend for local dev
```

---

## ⚙️ Pipeline Details

### 1. Wound Segmentation (MedSAM)
- Strips black padding from the image (handles padded clinical photos)
- Localizes the wound using **HSV color thresholding** — bright saturated reds (wound tissue) vs pale pink skin — using connected component analysis ranked by **mean saturation**, not area, to avoid selecting the whole arm
- Passes a tight bounding box to **`SamPredictor`** (MedSAM ViT-B) for precise mask prediction
- Falls back to a center-crop stub mask when no checkpoint is loaded (keeps the API testable)

### 2. Area Measurement
- **Pixel area**: direct sum of mask pixels
- **cm² area**: requires a physical reference object (coin, ruler) in the frame — user supplies the real-world diameter; the pixel diameter is currently a placeholder pending reference-object detection
- **DPI fallback**: attempts to estimate cm² from EXIF metadata if present and non-generic

### 3. Tissue Classification (HSV color analysis)
Classifies wound tissue from pixel colors inside the mask:

| Tissue | HSV Rule |
|---|---|
| **Granulation** | Hue 0–12° or 168–179° (red), Saturation ≥ 75, Value ≥ 55 |
| **Necrosis** | Value < 65 (dark), or brownish dark tones |
| **Slough / Fibrin** | Hue 12–50° (yellow), or pale/desaturated bright (white fibrin) |
| **Mixed** | No single category > 40% of wound pixels |

Returns the dominant type + full percentage composition used for nuanced risk scoring.

### 4. Infection Risk Assessment (rule-based)
Uses the **full tissue composition** — not just the dominant type — so a wound that is 80% granulation but 15% slough still scores MEDIUM risk:

| Condition | Score |
|---|---|
| Necrosis ≥ 30% | +3 |
| Necrosis 10–30% | +2 |
| Necrosis 5–10% | +1 |
| Slough ≥ 20% | +2 |
| Slough 5–20% | +1 |
| Mixed dominant type | +1 |
| Wound area > 20 cm² | +1 |

Score ≥ 3 → **HIGH** · Score ≥ 1 → **MEDIUM** · Score 0 → **LOW**

Always includes a depth disclaimer: wound depth cannot be assessed from 2D photography.

### 5. Healing Trend
Compares wound cm² area against the previous visit for the same patient:
- **Improving**: area decreased ≥ 10%
- **Stable**: area change between −10% and +10%
- **Worsening**: area increased ≥ 10%

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL (or Docker)
- 4 GB RAM minimum (8 GB recommended for MedSAM on CPU)

### 1. Clone the repo
```bash
git clone https://github.com/Dileepreddy2704/wound-tracker.git
cd wound-tracker
```

### 2. Set up environment
```bash
cp backend/.env.example backend/.env
# Edit backend/.env and set your DATABASE_URL
```

### 3a. Run with Docker (recommended)
```bash
docker compose up --build
```
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

### 3b. Run without Docker
```bash
# Backend
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend (new terminal)
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```
- API: http://localhost:8000
- Dashboard: http://localhost:8501

### 4. Download MedSAM checkpoint
Without the checkpoint the API still works (uses a stub mask for testing).
To enable real segmentation:
```bash
# Get the file ID from https://github.com/bowang-lab/MedSAM
python ml/download_medsam_checkpoint.py --gdrive_id <FILE_ID>
```
Then set in `backend/.env`:
```
MEDSAM_CHECKPOINT_PATH=../ml/checkpoints/medsam_vit_b.pth
```
Restart the backend — real MedSAM inference will now run.

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/patients/` | Create a patient (reference code, no PII) |
| `GET` | `/patients/` | List all patients |
| `GET` | `/patients/{id}` | Get a patient by ID |
| `POST` | `/visits/` | Upload wound photo (multipart: image + patient_id) |
| `GET` | `/visits/{patient_id}` | List all visits for a patient |
| `POST` | `/analyze/{visit_id}` | Run full analysis pipeline on a visit |
| `GET` | `/uploads/{filename}` | Serve an uploaded image or saved mask |

Interactive Swagger docs: **http://localhost:8000/docs**

---

## 🗺️ Roadmap

- [ ] Reference-object auto-detection (ArUco marker / circle detection) for reliable cm²
- [ ] Fine-tune SegFormer on the AZH wound dataset for mask precision
- [ ] Replace rule-based tissue classifier with a trained CNN/ViT classifier
- [ ] LLM-generated clinical reports (swap template text in `_build_report`)
- [ ] Wound depth estimation using structured light or stereo imaging
- [ ] Patient authentication + role-based access (clinician vs. patient views)
- [ ] Export reports as PDF

---

## ⚠️ Known Limitations

| Area | Current State |
|---|---|
| **Segmentation** | MedSAM with HSV-guided box prompt — works well on vivid red wounds; may miss necrotic/slough-covered wounds |
| **cm² measurement** | Requires a physical reference object in the photo; EXIF DPI fallback unreliable on smartphones |
| **Tissue classification** | HSV rule-based; fails on unusual lighting, very dark skin tones, or heavy dressing residue |
| **Infection risk** | Rule-based, not a validated clinical classifier — should not replace clinical judgment |
| **Wound depth** | Cannot be assessed from 2D photography |
| **Report generation** | Template text — not LLM-generated |

> **This tool is for research and educational purposes only. It is not a medical device and must not be used as a substitute for clinical assessment by a qualified healthcare professional.**

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI 0.115, Python 3.11 |
| Database | PostgreSQL 16, SQLAlchemy 2.0 |
| Segmentation | MedSAM (Meta SAM ViT-B fine-tuned on medical images) |
| Image processing | OpenCV, Pillow, NumPy |
| Frontend | Streamlit 1.37 |
| Infrastructure | Docker Compose |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built as a portfolio project demonstrating clinical AI system design: real ML inference (MedSAM), multi-visit tracking, quantitative wound measurement, and a full-stack deployment pipeline.*
