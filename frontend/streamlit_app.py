"""
Streamlit front-end for the AI Wound Assessment & Healing Tracker.

Run from the project root:
    streamlit run frontend/streamlit_app.py

The FastAPI backend must be running first:
    uvicorn app.main:app --reload   (from the backend/ directory)
"""

import io
import json
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Wound Tracker",
    page_icon="🩹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS – clean clinical look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}

    /* ── App header ─────────────────────────────────────────── */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0ea5e9 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.4rem;
    }
    .app-header h1 { margin:0; font-size:1.9rem; font-weight:800; color:#fff; }
    .app-header p  { margin:0.3rem 0 0; color:#bae6fd; font-size:0.88rem; }

    /* ── Section titles ─────────────────────────────────────── */
    .section-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #64748b;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    /* ── Status badges ──────────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .badge-low       { background:#dcfce7; color:#166534; }
    .badge-medium    { background:#fef9c3; color:#854d0e; }
    .badge-high      { background:#fee2e2; color:#991b1b; }
    .badge-improving { background:#dcfce7; color:#166534; }
    .badge-stable    { background:#dbeafe; color:#1e40af; }
    .badge-worsening { background:#fee2e2; color:#991b1b; }
    .badge-baseline  { background:#f1f5f9; color:#475569; }

    /* ── Report box ─────────────────────────────────────────── */
    .report-box {
        background: #0f172a;
        color: #e2e8f0;
        font-family: 'Courier New', monospace;
        font-size: 0.84rem;
        line-height: 1.8;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        white-space: pre-wrap;
    }

    /* ── Image captions ─────────────────────────────────────── */
    .img-caption {
        text-align: center;
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 0.3rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* Round images */
    .stImage > img { border-radius: 10px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _handle(r: requests.Response):
    r.raise_for_status()
    return r.json()


def api_get(path: str):
    try:
        return _handle(requests.get(f"{API_BASE}{path}", timeout=10))
    except requests.ConnectionError:
        st.error(f"❌ Cannot reach API at **{API_BASE}** — is the backend running?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post_json(path: str, payload: dict, timeout: int = 15):
    try:
        return _handle(requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout))
    except requests.ConnectionError:
        st.error("❌ Cannot reach API.")
        return None
    except requests.Timeout:
        st.error(
            f"⏱️ Request timed out after {timeout}s. "
            "The backend is still running — try increasing the timeout or check server logs."
        )
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post_multipart(path: str, files=None, data=None):
    try:
        return _handle(
            requests.post(f"{API_BASE}{path}", files=files, data=data, timeout=60)
        )
    except requests.ConnectionError:
        st.error("❌ Cannot reach API.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def fetch_image_from_url(url: str) -> Image.Image | None:
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def overlay_mask(
    original: Image.Image,
    mask: Image.Image,
    color: tuple = (220, 38, 38),
    alpha: float = 0.45,
) -> Image.Image:
    """
    Blend a semi-transparent coloured mask over the original image.
    Returns an RGB PIL image.
    """
    orig_rgba = original.convert("RGBA")
    mask_arr  = np.array(mask.convert("L"))

    overlay_arr = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
    overlay_arr[mask_arr > 127] = (*color, int(255 * alpha))

    blended = Image.alpha_composite(orig_rgba, Image.fromarray(overlay_arr, "RGBA"))
    return blended.convert("RGB")


def badge(text: str, css_class: str) -> str:
    return f'<span class="badge {css_class}">{text}</span>'


def infection_badge(flag: str | None) -> str:
    flag = (flag or "low").lower()
    labels = {"low": "LOW RISK", "medium": "MEDIUM RISK", "high": "HIGH RISK"}
    return badge(labels.get(flag, flag.upper()), f"badge-{flag}")


def trend_badge(trend: str | None) -> str:
    if not trend:
        return badge("BASELINE", "badge-baseline")
    icons = {"improving": "↓ IMPROVING", "stable": "→ STABLE", "worsening": "↑ WORSENING"}
    return badge(icons.get(trend.lower(), trend.upper()), f"badge-{trend.lower()}")


# ---------------------------------------------------------------------------
# ── Page header ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
      <h1>🩹 AI Wound Assessment & Healing Tracker</h1>
      <p>Upload wound photos → automatic segmentation → area measurement → healing trend analysis</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# API health check
health = api_get("/")
if health is None:
    st.stop()
st.success(f"✅ Backend connected — `{health.get('service', 'wound-tracker-api')}`")

# ---------------------------------------------------------------------------
# ── Sidebar – patient management ─────────────────────────────────────────────
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 👤 Patient Management")

    # ── Create patient form ───────────────────────────────────────────────
    with st.expander("➕ Create new patient", expanded=False):
        with st.form("form_create_patient", clear_on_submit=True):
            ref_code = st.text_input("Reference code *", placeholder="e.g. PT-2026-001")
            age      = st.number_input("Age", min_value=0, max_value=130, value=0)
            notes_p  = st.text_area("Notes", placeholder="Optional clinical notes…")
            if st.form_submit_button("Create patient", use_container_width=True):
                if not ref_code.strip():
                    st.error("Reference code is required.")
                else:
                    resp = api_post_json(
                        "/patients/",
                        {
                            "reference_code": ref_code.strip(),
                            "age":   int(age) if age > 0 else None,
                            "notes": notes_p.strip() or None,
                        },
                    )
                    if resp:
                        st.success(f"✅ Created: **{resp['reference_code']}**")
                        st.rerun()

    # ── Patient selector ──────────────────────────────────────────────────
    patients = api_get("/patients/") or []
    if not patients:
        st.warning("No patients yet. Create one above.")
        st.stop()

    options = {
        f"{p['reference_code']}  (…{p['id'][-6:]})" : p
        for p in patients
    }
    selected_label   = st.selectbox("Select patient", list(options.keys()))
    selected_patient = options[selected_label]

    st.divider()
    st.caption("SELECTED PATIENT")
    st.markdown(f"**Code:** `{selected_patient['reference_code']}`")
    st.markdown(f"**ID:** `{selected_patient['id']}`")
    if selected_patient.get("age"):
        st.markdown(f"**Age:** {selected_patient['age']}")
    if selected_patient.get("notes"):
        st.markdown(f"**Notes:** {selected_patient['notes']}")
    st.markdown(
        f"**Created:** {selected_patient.get('created_at', '')[:10]}"
    )

# ---------------------------------------------------------------------------
# ── Main tabs ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
tab_analyze, tab_history = st.tabs(["🔬 New Visit & Analysis", "📈 Visit History"])


# ═══════════════════════════════════════════════════════════════════════════
# Tab 1 – New Visit & Analysis
# ═══════════════════════════════════════════════════════════════════════════
with tab_analyze:
    left, right = st.columns([1, 1.7], gap="large")

    # ── Upload form ───────────────────────────────────────────────────────
    with left:
        st.markdown('<div class="section-title">Upload Wound Photo</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Use a well-lit, close-up photo. Add a reference object (coin, ruler) for cm² calibration.",
        )

        ref_diam = st.number_input(
            "Reference object diameter (cm)",
            min_value=0.0, value=0.0, step=0.1, format="%.1f",
            help="Real-world diameter of any circular reference in the photo. Leave 0 to skip.",
        )
        clin_notes = st.text_area(
            "Clinical notes",
            placeholder="Wound location, dressing type, pain level…",
        )

        if uploaded:
            st.image(uploaded, caption="Preview", use_container_width=True)

        run_btn = st.button(
            "🔬  Run Analysis",
            type="primary",
            disabled=(uploaded is None),
            use_container_width=True,
        )

    # ── Results panel ─────────────────────────────────────────────────────
    with right:
        st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)

        if run_btn and uploaded:
            # 1. Create visit (uploads image)
            with st.spinner("Uploading image and creating visit…"):
                form_data: dict = {"patient_id": selected_patient["id"]}
                if ref_diam > 0:
                    form_data["reference_object_diameter_cm"] = str(ref_diam)
                if clin_notes.strip():
                    form_data["clinical_notes"] = clin_notes.strip()

                visit = api_post_multipart(
                    "/visits/",
                    files={"image": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    data=form_data,
                )

            if visit:
                # 2. Run analysis pipeline
                with st.spinner(
                    "⏳ Running wound segmentation… "
                    "This can take 30–120 s on CPU. Please wait."
                ):
                    analysis = api_post_json(
                        f"/analyze/{visit['id']}", {}, timeout=300
                    )

                if analysis:
                    st.session_state["analysis"] = analysis
                    st.session_state["raw_img"]   = uploaded.getvalue()

        # ── Display stored analysis ───────────────────────────────────────
        if "analysis" in st.session_state:
            analysis = st.session_state["analysis"]
            raw_img  = st.session_state["raw_img"]

            orig_img = Image.open(io.BytesIO(raw_img)).convert("RGB")

            # Fetch the segmentation mask from the backend
            mask_path = analysis.get("mask_path") or ""
            mask_filename = os.path.basename(mask_path) if mask_path else None
            mask_img = None
            if mask_filename:
                mask_img = fetch_image_from_url(f"{API_BASE}/uploads/{mask_filename}")

            # ── Side-by-side image comparison ────────────────────────────
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(orig_img, use_container_width=True)
                st.markdown('<div class="img-caption">Original</div>', unsafe_allow_html=True)

            with img_col2:
                if mask_img is not None:
                    blended = overlay_mask(orig_img, mask_img)
                    st.image(blended, use_container_width=True)
                    st.markdown(
                        '<div class="img-caption">🔴 Wound mask overlay</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.image(orig_img, use_container_width=True)
                    st.warning("Mask image not available from API.")

            st.divider()

            # ── Metric row ────────────────────────────────────────────────
            area_px         = analysis.get("area_px")
            area_cm2        = analysis.get("area_cm2")
            tissue          = (analysis.get("tissue_type") or "unclassified").capitalize()
            tissue_conf     = analysis.get("tissue_confidence")
            wound_type      = (analysis.get("wound_type") or "unknown").capitalize()
            wound_type_conf = analysis.get("wound_type_confidence")
            change          = analysis.get("area_change_pct")

            # Row 1: wound type (prominent, spans full width)
            wt_conf_str = f"  ({wound_type_conf*100:.0f}% confidence)" if wound_type_conf else ""

            # Wound type icon map
            wt_icons = {
                "Incision":  "🔪", "Laceration": "⚡", "Abrasion": "🛞",
                "Burn":      "🔥", "Avulsion":   "⚠️", "Puncture": "📍",
                "Unknown":   "❓",
            }
            wt_icon = wt_icons.get(wound_type, "🩹")
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.05);border-radius:10px;"
                f"padding:12px 20px;margin-bottom:12px;border-left:4px solid #e74c3c'>"
                f"<span style='font-size:0.8em;color:#aaa;text-transform:uppercase;"
                f"letter-spacing:1px'>Wound Type</span><br>"
                f"<span style='font-size:1.6em;font-weight:700'>"
                f"{wt_icon} {wound_type}</span>"
                f"<span style='font-size:0.85em;color:#aaa'>{wt_conf_str}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Row 2: area + tissue + change metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Wound area (px)",  f"{int(area_px):,}" if area_px else "—")
            m2.metric(
                "Wound area (cm²)",
                f"{area_cm2:.2f}" if area_cm2 else "—",
                help="Set reference object diameter > 0 cm to enable this measurement",
            )
            m3.metric(
                "Tissue type",
                tissue,
                delta=f"{tissue_conf*100:.0f}% conf" if tissue_conf else None,
                delta_color="off",
            )
            m4.metric(
                "Area change",
                f"{change:+.1f}%" if change is not None else "—",
                delta_color="inverse" if change else "off",
                help="vs. previous visit for this patient",
            )

            # Show calibration hint if cm² is missing
            if not area_cm2:
                st.info(
                    "💡 **Tip:** To get cm² measurement, add a reference object "
                    "(e.g. a 2.5 cm coin) to the photo and enter its diameter above "
                    "before clicking Run Analysis."
                )

            # ── Badges row ────────────────────────────────────────────────
            b1, b2 = st.columns(2)
            with b1:
                st.markdown(
                    f"**Infection risk** &nbsp; {infection_badge(analysis.get('infection_risk_flag'))}",
                    unsafe_allow_html=True,
                )
            with b2:
                st.markdown(
                    f"**Healing trend** &nbsp; {trend_badge(analysis.get('healing_trend'))}",
                    unsafe_allow_html=True,
                )

            # Infection indicators detail
            indicators_raw = analysis.get("infection_indicators") or "[]"
            try:
                indicators = json.loads(indicators_raw)
                if indicators:
                    with st.expander("🔍 Risk indicators detail", expanded=False):
                        for ind in indicators:
                            st.markdown(f"- {ind}")
            except Exception:
                pass

            st.divider()

            # ── Clinical report ───────────────────────────────────────────
            with st.expander("📄 Clinical Report", expanded=True):
                report_text = analysis.get("report_text") or "No report generated."
                st.markdown(
                    f'<div class="report-box">{report_text}</div>',
                    unsafe_allow_html=True,
                )

            # Analysis meta
            st.caption(
                f"Analysis ID: `{analysis.get('id', '—')}`  ·  "
                f"Visit ID: `{analysis.get('visit_id', '—')}`  ·  "
                f"Created: `{str(analysis.get('created_at', ''))[:19]}`"
            )

        else:
            st.info(
                "👆 Select a patient in the sidebar, upload a wound photo, then click **Run Analysis**."
            )


# ═══════════════════════════════════════════════════════════════════════════
# Tab 2 – Visit History
# ═══════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown('<div class="section-title">Visit History</div>', unsafe_allow_html=True)

    if st.button("🔄 Refresh", key="refresh_history"):
        st.rerun()

    visits = api_get(f"/visits/{selected_patient['id']}") or []

    if not visits:
        st.info(f"No visits recorded for **{selected_patient['reference_code']}** yet.")
    else:
        st.markdown(f"**{len(visits)} visit(s)** for patient `{selected_patient['reference_code']}`")
        st.divider()

        rows = []
        for v in visits:
            rows.append(
                {
                    "Visit date":   str(v.get("visit_date", ""))[:19].replace("T", " "),
                    "Visit ID":     "…" + v.get("id", "")[-8:],
                    "Image file":   os.path.basename(v.get("image_path", "") or ""),
                    "Ref Ø (cm)":   v.get("reference_object_diameter_cm") or "—",
                    "Notes":        (v.get("clinical_notes") or "")[:60] or "—",
                }
            )

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Preview any visit's image ─────────────────────────────────────
        st.divider()
        st.markdown('<div class="section-title">Preview a Visit</div>', unsafe_allow_html=True)

        visit_labels = {
            f"Visit {i+1}  —  {str(v.get('visit_date',''))[:10]}  (…{v.get('id','')[-6:]})" : v
            for i, v in enumerate(visits)
        }
        chosen_label = st.selectbox("Select visit to preview", list(visit_labels.keys()))
        chosen_visit = visit_labels[chosen_label]

        img_filename = os.path.basename(chosen_visit.get("image_path") or "")
        mask_filename = img_filename.rsplit(".", 1)[0] + "_mask.png" if img_filename else None

        if img_filename:
            pv1, pv2 = st.columns(2)
            with pv1:
                orig = fetch_image_from_url(f"{API_BASE}/uploads/{img_filename}")
                if orig:
                    st.image(orig, use_container_width=True)
                    st.markdown('<div class="img-caption">Original</div>', unsafe_allow_html=True)
                else:
                    st.warning("Image not available.")

            with pv2:
                if mask_filename:
                    mask = fetch_image_from_url(f"{API_BASE}/uploads/{mask_filename}")
                    if mask and orig:
                        st.image(overlay_mask(orig, mask), use_container_width=True)
                        st.markdown(
                            '<div class="img-caption">🔴 Wound mask overlay</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("No mask yet — run analysis on this visit first.")
