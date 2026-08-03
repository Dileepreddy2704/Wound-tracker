# 🩹 AI Wound Assessment & Healing Tracker

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-FF4B4B?logo=streamlit&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A full-stack clinical AI application that takes wound photographs and runs an automated pipeline:

**Image upload → MedSAM segmentation → area measurement → tissue classification → infection risk assessment → visit-over-visit healing trend → clinical report**

Built with **FastAPI + PostgreSQL** (backend), **MedSAM** (segmentation), **OpenCV HSV analysis** (tissue classification), and **Streamlit** (frontend dashboard).

---

## 📸 Screenshots

### Analysis Dashboard
<img width="1907" height="841" alt="image" src="https://github.com/user-attachments/assets/c48cf4ba-e2e2-488f-b815-db661836c2c5" />


### Visit History
<img width="1597" height="816" alt="image" src="https://github.com/user-attachments/assets/536df7f8-4054-45a5-bef8-207a60220936" />

---

## 🔬 Example Output

```
WOUND ASSESSMENT SUMMARY
----------------------------------
Wound area      : not calibrated
                  (provide reference object diameter > 0 for cm² measurement)
Tissue type     : Granulation  (83% confidence)
  Composition breakdown:
    granulation    ████████████████     83%
    slough         ██                    8%
    necrosis       █                     5%
Infection risk  : MEDIUM
  • Slough/fibrin areas detected (8%) — monitor and consider debridement
  • ⚠ Wound depth cannot be assessed from 2D photography — clinical examination required
Healing trend   : baseline visit — no prior visit to compare
----------------------------------
NOTE: Granulation tissue indicates active healing. Maintain moist wound environment.
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
│   └── .env.example
├── frontend/
│   ├── streamlit_app.py          # Full Streamlit dashboard
│   └── requirements.txt
├── ml/
│   ├── download_medsam_checkpoint.py  # Download MedSAM weights
│   ├── train_segmentation.py          # Fine-tuning skeleton (SegFormer / MedSAM)
│   └── data/                          # Dataset prep scripts
├── docs/screenshots/             # README images
└── docker-compose.yml            # Postgres + backend for local dev
```

---

## ⚙️ Pipeline Details

### 1. Wound Segmentation (MedSAM)
- Strips black padding from the image (handles padded clinical photos)
- Localizes the wound using **HSV color thresholding** — bright saturated reds (wound tissue) vs pale pink skin — using connected component analysis ranked by **mean saturation**, not area, to avoid selecting the whole arm
- Passes a tight bounding box to **`SamPredictor`** (MedSAM ViT-B) for precise mask prediction
- Falls back to a center-crop stub mask when no checkpoint is loaded (keeps the API testable without downloading the 375 MB model)

### 2. Area Measurement
- **Pixel area**: direct sum of mask pixels
- **cm² area**: requires a physical reference object (coin, ruler) in the frame — user supplies the real-world diameter
- **DPI fallback**: attempts to estimate cm² from EXIF metadata if present and non-generic

### 3. Tissue Classification (HSV color analysis)

Classifies wound tissue from pixel colors inside the mask:

| Tissue | HSV Rule | Clinical Meaning |
|---|---|---|
| **Granulation** | Hue 0–12° or 168–179°, Sat ≥ 75, Val ≥ 55 | Active healing, new vasculature |
| **Necrosis** | Value < 65, or dark brownish tones | Dead tissue, urgent debridement needed |
| **Slough** | Hue 12–50° (yellow), or pale desaturated bright | Fibrinous debris, impedes healing |
| **Mixed** | No single type > 40% of wound pixels | Complex wound, monitor closely |

Returns the dominant type + **full percentage composition** used for nuanced risk scoring.

### 4. Infection Risk Assessment (rule-based)

Uses the **full tissue composition** — not just the dominant type — so a wound that is 80% granulation but 15% slough still scores **MEDIUM** risk:

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

> Always includes a depth disclaimer — wound depth cannot be assessed from 2D photography alone.

### 5. Healing Trend (visit-over-visit)
Compares wound cm² area against the **previous visit** for the same patient:

| Change | Trend |
|---|---|
| Area decreased ≥ 10% | ↓ Improving |
| Change between −10% and +10% | → Stable |
| Area increased ≥ 10% | ↑ Worsening |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL (or Docker)
- 4 GB RAM minimum (8 GB recommended for MedSAM on CPU)

### 1. Clone the repo
```bash
git clone https://github.com/Dileepreddy2704/Wound-tracker.git
cd Wound-tracker
```

### 2. Set up environment
```bash
cp backend/.env.example backend/.env
# Edit backend/.env — set your DATABASE_URL
```

### 3. Run with Docker (recommended)
```bash
docker compose up --build
```
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs

### 4. Run without Docker
```bash
# Terminal 1 — Backend
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload

# Terminal 2 — Frontend
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```
- API: http://localhost:8000
- Dashboard: http://localhost:8501

### 5. Download MedSAM checkpoint

Without the checkpoint the API still works (uses a stub mask for testing). To enable real segmentation:

```bash
# Get the file ID from https://github.com/bowang-lab/MedSAM
python ml/download_medsam_checkpoint.py --gdrive_id <FILE_ID>
```

Then set in `backend/.env`:
```
MEDSAM_CHECKPOINT_PATH=../ml/checkpoints/medsam_vit_b.pth
```
Restart the backend — real MedSAM inference will now run on every `/analyze/{visit_id}` call.

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/patients/` | Create a patient (reference code only, no PII) |
| `GET` | `/patients/` | List all patients |
| `GET` | `/patients/{id}` | Get a patient by ID |
| `POST` | `/visits/` | Upload wound photo (multipart: image + patient_id) |
| `GET` | `/visits/{patient_id}` | List all visits for a patient (ordered by date) |
| `POST` | `/analyze/{visit_id}` | Run full analysis pipeline on a visit |
| `GET` | `/uploads/{filename}` | Serve an uploaded image or saved mask PNG |

Interactive Swagger docs: **http://localhost:8000/docs**

---

## 🗺️ Roadmap

- [ ] Reference-object auto-detection (ArUco marker / circle detection) for reliable cm² without manual input
- [ ] Fine-tune SegFormer on the AZH wound dataset for improved mask precision
- [ ] Replace HSV tissue classifier with a trained CNN/ViT classifier
- [ ] LLM-generated clinical reports (replace template text in `_build_report`)
- [ ] Wound depth estimation using structured light or stereo imaging
- [ ] Patient authentication + role-based access (clinician vs. patient views)
- [ ] Export reports as PDF

---

## ⚠️ Known Limitations

| Area | Current State |
|---|---|
| **Segmentation** | MedSAM with HSV-guided box prompt — works well on vivid red wounds; may miss necrotic/slough-covered wounds |
| **cm² measurement** | Requires a physical reference object in the photo; EXIF DPI fallback unreliable on smartphones |
| **Tissue classification** | HSV rule-based; can misfire on unusual lighting, dark skin tones, or dressing residue |
| **Infection risk** | Rule-based scoring, not a validated clinical classifier |
| **Wound depth** | Cannot be assessed from 2D photography |
| **Report generation** | Template text — not LLM-generated |

> ⚠️ **Disclaimer:** This tool is for research and educational purposes only. It is not a certified medical device and must not replace clinical assessment by a qualified healthcare professional.

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
