"""Streamlit web application for Project 21: Multimodal ICU Mortality Prediction."""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DATA_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR, ProjectConfig
from src.data_loader import load_icu_dataset
from src.evaluate import evaluate_bundle
from src.predict import predict_patient_mortality
from src.train import train_all_models

# ---------------------------------------------------------------------------
# Page Config & Dark Navy / White Healthcare AI Theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ICU Multimodal Mortality Prediction | Project 21",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME_CSS = """
<style>
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #0B192C;
        color: #FFFFFF;
    }
    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }
    .hero-banner {
        background: linear-gradient(135deg, #0B192C 0%, #1E3A5F 100%);
        color: #FFFFFF;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.25rem;
        border-left: 6px solid #38BDF8;
    }
    .hero-banner h1 {
        color: #FFFFFF;
        margin: 0 0 0.4rem 0;
        font-size: 1.75rem;
        font-weight: 700;
    }
    .hero-banner p {
        color: #CBD5E1;
        margin: 0;
        font-size: 0.95rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #0B192C;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(11, 25, 44, 0.08);
    }
    .metric-card .label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #475569;
        font-weight: 600;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0B192C;
        margin-top: 0.25rem;
    }
    .metric-card .sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.15rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border: 1px solid #F59E0B;
        border-left: 5px solid #D97706;
        color: #78350F;
        padding: 0.85rem 1.1rem;
        border-radius: 6px;
        font-size: 0.88rem;
        margin-bottom: 1rem;
    }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

DISCLAIMER_TEXT = (
    "EDUCATIONAL PROTOTYPE DISCLAIMER: This application demonstrates Project 21 "
    "(Multimodal Fusion of Clinical Notes and Structured Data for ICU Mortality Prediction) "
    "using entirely fictional synthetic patient records. Predicted probabilities and risk tiers "
    "are NOT clinically validated medical risk scores and must NOT be used for clinical care."
)


@st.cache_data(show_spinner=False)
def get_cohort_df() -> pd.DataFrame:
    return load_icu_dataset()


def load_json_if_exists(path: Path) -> Dict[str, Any] | None:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def render_metric_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_gauge(prob: float, tier: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.0, 1.5), dpi=150)
    ax.barh([0], [0.25], left=0.0, height=0.45, color="#10B981", alpha=0.85)
    ax.barh([0], [0.30], left=0.25, height=0.45, color="#F59E0B", alpha=0.85)
    ax.barh([0], [0.45], left=0.55, height=0.45, color="#EF4444", alpha=0.85)
    ax.plot([prob, prob], [-0.32, 0.32], color="#0B192C", linewidth=3.5)
    ax.scatter([prob], [0.34], color="#0B192C", s=60, zorder=5, marker="v")
    ax.text(0.125, -0.42, "Low (<0.25)", ha="center", va="top", fontsize=8.5, color="#065F46", fontweight="bold")
    ax.text(0.40, -0.42, "Moderate (0.25-0.55)", ha="center", va="top", fontsize=8.5, color="#92400E", fontweight="bold")
    ax.text(0.775, -0.42, "High (>=0.55)", ha="center", va="top", fontsize=8.5, color="#991B1B", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.55)
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.25, 0.50, 0.55, 0.75, 1.0])
    ax.set_title(f"Mortality Probability: {prob*100:.2f}% — {tier}", fontsize=10.5, fontweight="bold", color="#0B192C")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def render_architecture_diagram() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=160)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    def draw_box(x: float, y: float, w: float, h: float, title: str, subtitle: str, facecolor: str, edgecolor: str = "#0B192C", textcolor: str = "#0B192C") -> None:
        box = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=1.6,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=9.2, fontweight="bold", color=textcolor)
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=7.8, color=textcolor)

    def draw_arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.6, color="#1E3A5F"))

    # Branch A
    draw_box(0.2, 3.6, 2.0, 1.15, "Branch A: Structured", "21 Numerics + 21 Flags\n+ 3 Categoricals (52-dim)", "#E0F2FE")
    draw_box(2.7, 3.6, 2.1, 1.15, "Structured Encoder", "Hidden [64, 32] + ReLU\nDropout 0.25 -> h_s (24-dim)", "#DBEAFE")
    draw_arrow(2.28, 4.18, 2.62, 4.18)

    # Branch B
    draw_box(0.2, 0.8, 2.0, 1.15, "Branch B: Clinical Notes", "Clean, De-ID, Expand\nBio_ClinicalBERT / TF-IDF+SVD", "#E0F2FE")
    draw_box(2.7, 0.8, 2.1, 1.15, "Text Encoder", "Hidden [32] + ReLU\nDropout 0.25 -> h_t (24-dim)", "#DBEAFE")
    draw_arrow(2.28, 1.38, 2.62, 1.38)

    # Fusion
    draw_box(5.4, 2.05, 2.2, 1.45, "Gated Multimodal Fusion", "Concat [h_s; h_t] (48-dim)\ng = sigmoid(W_g * cat + b_g)\nh_f = ReLU(W_f*(g*cat)) -> 24d", "#0B192C", textcolor="#FFFFFF")
    draw_arrow(4.88, 4.05, 5.34, 3.0)
    draw_arrow(4.88, 1.50, 5.34, 2.55)

    # Head
    draw_box(8.1, 2.2, 1.65, 1.15, "Classification Head", "Linear(24->1) + Sigmoid\nP(Mortality) in [0, 1]", "#1E3A5F", textcolor="#FFFFFF")
    draw_arrow(7.68, 2.78, 8.02, 2.78)

    ax.set_title("Project 21: Dual-Branch Sigmoid-Gated Multimodal Architecture", fontsize=11.5, fontweight="bold", color="#0B192C", pad=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🏥 Project 21")
st.sidebar.caption("Multimodal ICU Mortality Prediction")
page = st.sidebar.radio(
    "Navigation",
    [
        "1. Dashboard",
        "2. Patient Risk Prediction",
        "3. Model Comparison",
        "4. Model Architecture",
        "5. Dataset Explorer",
        "6. Training & Evaluation",
    ],
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Cohort:** Synthetic (2,500 stays)\n\n"
    "**Split:** Patient-level 70/15/15\n\n"
    "**Seed:** 42 | **Threshold:** 0.50"
)

st.markdown(
    """
    <div class="hero-banner">
        <h1>Multimodal Fusion of Clinical Notes & Structured Data for ICU Mortality Prediction</h1>
        <p>Project 21 — Educational Healthcare AI Research Prototype (First-24h Observation Window)</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(f'<div class="disclaimer-box">⚠️ {DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)

cfg = ProjectConfig()
df = get_cohort_df()
eval_results = load_json_if_exists(METRICS_DIR / "evaluation_results.json")
eval_curves = load_json_if_exists(METRICS_DIR / "evaluation_curves.json")
train_hist = load_json_if_exists(METRICS_DIR / "training_histories.json")
bundle_exists = (MODELS_DIR / "multimodal_bundle.pkl").exists()

# ---------------------------------------------------------------------------
# PAGE 1: DASHBOARD
# ---------------------------------------------------------------------------
if page == "1. Dashboard":
    st.subheader("System & Cohort Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Total Synthetic Stays", f"{len(df):,}", "First-24h ICU observation window")
    with c2:
        render_metric_card(
            "Unique Fictional Patients",
            f"{df['subject_id'].nunique():,}",
            f"From {cfg.n_patients:,} configured IDs",
        )
    with c3:
        mort_rate = df["hospital_expire_flag"].mean() * 100
        render_metric_card(
            "Cohort Mortality Rate",
            f"{mort_rate:.2f}%",
            f"{int(df['hospital_expire_flag'].sum())} positive stays",
        )
    with c4:
        status_str = "Trained & Ready" if bundle_exists else "Not Trained Yet"
        backend_str = train_hist.get("backend", "numpy") if train_hist else "N/A"
        render_metric_card("Model Status", status_str, f"Active backend: {backend_str}")

    st.markdown("### Patient-Level Splits & Mortality Distribution")
    col_l, col_r = st.columns([1.2, 1.0])
    with col_l:
        if train_hist and "split_summary" in train_hist:
            ss = train_hist["split_summary"]
            split_df = pd.DataFrame(
                [
                    {"Split": "Training (70%)", "ICU Stays": ss["train_stays"], "Mortality Rate": f"{ss['train_mortality_rate']*100:.2f}%"},
                    {"Split": "Validation (15%)", "ICU Stays": ss["val_stays"], "Mortality Rate": f"{ss['val_mortality_rate']*100:.2f}%"},
                    {"Split": "Held-Out Test (15%)", "ICU Stays": ss["test_stays"], "Mortality Rate": f"{ss['test_mortality_rate']*100:.2f}%"},
                ]
            )
            st.dataframe(split_df, use_container_width=True, hide_index=True)
        else:
            st.info("Train models on Page 6 (Training & Evaluation) to populate exact split summary.")

        st.markdown("### Held-Out Test Performance Summary")
        if eval_results:
            rows = []
            label_map = {
                "structured_only": "Structured-Only",
                "text_only": "Clinical-Text-Only",
                "multimodal_fusion": "Gated Multimodal Fusion",
            }
            for k, m in eval_results.items():
                rows.append(
                    {
                        "Model": label_map.get(k, k),
                        "Accuracy": m["accuracy"],
                        "Precision": m["precision"],
                        "Recall": m["recall"],
                        "F1-Score": m["f1_score"],
                        "ROC-AUC": m["roc_auc"],
                        "PR-AUC": m["pr_auc"],
                        "Brier Score": m["brier_score"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("Evaluation metrics not found yet. Run Training & Evaluation on Page 6.")

    with col_r:
        fig, ax = plt.subplots(figsize=(5.0, 3.4), dpi=130)
        counts = df["hospital_expire_flag"].value_counts().sort_index()
        bars = ax.bar(
            ["Survived (0)", "Expired (1)"],
            [counts.get(0, 0), counts.get(1, 0)],
            color=["#1E3A5F", "#DC2626"],
            width=0.5,
        )
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 30, f"{int(h)}", ha="center", fontweight="bold")
        ax.set_ylabel("ICU Stays")
        ax.set_title("Synthetic Cohort Outcome Distribution", fontweight="bold", color="#0B192C")
        ax.set_ylim(0, max(counts.values) * 1.15)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        fig.tight_layout()
        st.pyplot(fig)

# ---------------------------------------------------------------------------
# PAGE 2: PATIENT RISK PREDICTION
# ---------------------------------------------------------------------------
elif page == "2. Patient Risk Prediction":
    st.subheader("Interactive Multimodal Patient Risk Prediction")
    if not bundle_exists:
        st.error("No trained model bundle found. Please go to **6. Training & Evaluation** and train the models first.")
    else:
        preset = st.selectbox(
            "Load Demonstration Preset (Fictional):",
            [
                "High Risk — Septic Shock (DEMO-ICU-9042)",
                "Moderate Risk — Acute Decompensated Heart Failure",
                "Low Risk — Elective Post-Op Observation",
            ],
        )
        if preset.startswith("High"):
            defaults = {
                "age": 72.0, "gender": "M", "admission_type": "EMERGENCY", "first_careunit": "MICU",
                "heart_rate_mean": 114.0, "heart_rate_max": 136.0, "sbp_mean": 88.0, "sbp_min": 74.0,
                "dbp_mean": 52.0, "resp_rate_mean": 26.0, "spo2_mean": 90.0, "spo2_min": 84.0,
                "temperature_c": 38.4, "wbc_count": 18.2, "hemoglobin": 9.4, "platelets": 112.0,
                "creatinine": 2.9, "bun": 48.0, "lactate": 4.8, "sodium": 135.0, "potassium": 5.1,
                "bicarbonate": 16.0, "glucose": 210.0, "gcs_total": 10,
                "note": (
                    "ICU 24h Progress Note (MICU): 72-year-old male emergency admission presenting "
                    "with septic shock, worsening lactic acidosis, and refractory hypotension. "
                    "Patient remains hemodynamically unstable on norepinephrine infusion. "
                    "Oliguric renal failure noted; guarded prognosis."
                ),
            }
        elif preset.startswith("Moderate"):
            defaults = {
                "age": 66.0, "gender": "F", "admission_type": "URGENT", "first_careunit": "CCU",
                "heart_rate_mean": 94.0, "heart_rate_max": 110.0, "sbp_mean": 110.0, "sbp_min": 94.0,
                "dbp_mean": 62.0, "resp_rate_mean": 22.0, "spo2_mean": 94.0, "spo2_min": 90.0,
                "temperature_c": 37.2, "wbc_count": 11.8, "hemoglobin": 10.9, "platelets": 185.0,
                "creatinine": 1.7, "bun": 30.0, "lactate": 2.1, "sodium": 134.0, "potassium": 4.5,
                "bicarbonate": 21.0, "glucose": 158.0, "gcs_total": 14,
                "note": (
                    "ICU 24h Progress Note (CCU): 66-year-old female urgent admission presenting "
                    "with acute decompensated heart failure with pulmonary edema. "
                    "Oxygenation improved on supplemental nasal cannula. Responding gradually to diuresis."
                ),
            }
        else:
            defaults = {
                "age": 54.0, "gender": "M", "admission_type": "ELECTIVE", "first_careunit": "CVICU",
                "heart_rate_mean": 74.0, "heart_rate_max": 86.0, "sbp_mean": 124.0, "sbp_min": 112.0,
                "dbp_mean": 70.0, "resp_rate_mean": 16.0, "spo2_mean": 98.0, "spo2_min": 96.0,
                "temperature_c": 36.8, "wbc_count": 8.4, "hemoglobin": 12.8, "platelets": 230.0,
                "creatinine": 0.9, "bun": 16.0, "lactate": 1.1, "sodium": 139.0, "potassium": 4.1,
                "bicarbonate": 24.5, "glucose": 118.0, "gcs_total": 15,
                "note": (
                    "ICU 24h Progress Note (CVICU): 54-year-old male elective admission presenting "
                    "for post-operative observation after cardiac catheterization. "
                    "Patient comfortable, alert and oriented x3. Normal urine output and stable vital signs."
                ),
            }

        with st.form("patient_prediction_form"):
            st.markdown("#### Branch A: Structured Clinical Features (All 21 Numerical + 3 Categorical)")
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                age = st.number_input("Age (years)", 18.0, 95.0, float(defaults["age"]), step=1.0)
            with d2:
                gender = st.selectbox("Gender", ["M", "F"], index=0 if defaults["gender"] == "M" else 1)
            with d3:
                admission_type = st.selectbox(
                    "Admission Type",
                    ["EMERGENCY", "URGENT", "ELECTIVE"],
                    index=["EMERGENCY", "URGENT", "ELECTIVE"].index(defaults["admission_type"]),
                )
            with d4:
                first_careunit = st.selectbox(
                    "First Care Unit",
                    ["MICU", "SICU", "CCU", "CVICU", "TSICU"],
                    index=["MICU", "SICU", "CCU", "CVICU", "TSICU"].index(defaults["first_careunit"]),
                )

            v1, v2, v3, v4, v5 = st.columns(5)
            with v1:
                hr_mean = st.number_input("HR Mean (bpm)", 30.0, 200.0, float(defaults["heart_rate_mean"]))
                hr_max = st.number_input("HR Max (bpm)", 30.0, 220.0, float(defaults["heart_rate_max"]))
            with v2:
                sbp_mean = st.number_input("SBP Mean (mmHg)", 40.0, 220.0, float(defaults["sbp_mean"]))
                sbp_min = st.number_input("SBP Min (mmHg)", 30.0, 200.0, float(defaults["sbp_min"]))
            with v3:
                dbp_mean = st.number_input("DBP Mean (mmHg)", 20.0, 140.0, float(defaults["dbp_mean"]))
                rr_mean = st.number_input("Resp Rate Mean (/min)", 5.0, 50.0, float(defaults["resp_rate_mean"]))
            with v4:
                spo2_mean = st.number_input("SpO2 Mean (%)", 50.0, 100.0, float(defaults["spo2_mean"]))
                spo2_min = st.number_input("SpO2 Min (%)", 40.0, 100.0, float(defaults["spo2_min"]))
            with v5:
                temp_c = st.number_input("Temperature (C)", 32.0, 42.0, float(defaults["temperature_c"]))
                gcs_total = st.slider("GCS Total", 3, 15, int(defaults["gcs_total"]))

            l1, l2, l3, l4, l5 = st.columns(5)
            with l1:
                wbc = st.number_input("WBC (K/uL)", 0.5, 60.0, float(defaults["wbc_count"]))
                hgb = st.number_input("Hemoglobin (g/dL)", 4.0, 20.0, float(defaults["hemoglobin"]))
            with l2:
                plt_cnt = st.number_input("Platelets (K/uL)", 5.0, 800.0, float(defaults["platelets"]))
                creat = st.number_input("Creatinine (mg/dL)", 0.2, 15.0, float(defaults["creatinine"]))
            with l3:
                bun = st.number_input("BUN (mg/dL)", 2.0, 150.0, float(defaults["bun"]))
                lactate = st.number_input("Lactate (mmol/L)", 0.3, 20.0, float(defaults["lactate"]))
            with l4:
                sodium = st.number_input("Sodium (mEq/L)", 110.0, 170.0, float(defaults["sodium"]))
                potassium = st.number_input("Potassium (mEq/L)", 2.0, 8.5, float(defaults["potassium"]))
            with l5:
                bicarb = st.number_input("Bicarbonate (mEq/L)", 5.0, 45.0, float(defaults["bicarbonate"]))
                glucose = st.number_input("Glucose (mg/dL)", 30.0, 600.0, float(defaults["glucose"]))

            missing_sim = st.multiselect(
                "Optional: Simulate Missing Lab/Vital Values (triggers 21-dim missingness flags & train-median imputation)",
                ["lactate", "bicarbonate", "platelets", "temperature_c", "spo2_min", "glucose"],
            )

            st.markdown("#### Branch B: Unstructured Clinical Note (First 24h)")
            clinical_note = st.text_area("Anonymized 24h Progress / Admission Note", value=defaults["note"], height=140)
            submitted = st.form_submit_button("Estimate Mortality Risk Probability", type="primary")

        if submitted:
            payload: Dict[str, Any] = {
                "subject_id": "STREAMLIT-DEMO-01",
                "age": age,
                "gender": gender,
                "admission_type": admission_type,
                "first_careunit": first_careunit,
                "heart_rate_mean": hr_mean,
                "heart_rate_max": hr_max,
                "sbp_mean": sbp_mean,
                "sbp_min": sbp_min,
                "dbp_mean": dbp_mean,
                "resp_rate_mean": rr_mean,
                "spo2_mean": spo2_mean,
                "spo2_min": np.nan if "spo2_min" in missing_sim else spo2_min,
                "temperature_c": np.nan if "temperature_c" in missing_sim else temp_c,
                "wbc_count": wbc,
                "hemoglobin": hgb,
                "platelets": np.nan if "platelets" in missing_sim else plt_cnt,
                "creatinine": creat,
                "bun": bun,
                "lactate": np.nan if "lactate" in missing_sim else lactate,
                "sodium": sodium,
                "potassium": potassium,
                "bicarbonate": np.nan if "bicarbonate" in missing_sim else bicarb,
                "glucose": np.nan if "glucose" in missing_sim else glucose,
                "gcs_total": gcs_total,
                "clinical_note": clinical_note,
            }
            res = predict_patient_mortality(payload)
            prob = res["predicted_mortality_probability"]

            st.markdown("### Prediction Output")
            m1, m2, m3 = st.columns(3)
            with m1:
                render_metric_card("Predicted Mortality Probability", f"{prob*100:.2f}%", "Threshold: 0.50")
            with m2:
                render_metric_card("Binary Classification", res["predicted_label"], f"Class {res['predicted_class']}")
            with m3:
                render_metric_card("Demonstration Risk Tier", res["risk_tier"], "Educational tier only")

            st.pyplot(render_probability_gauge(prob, res["risk_tier"]))

            exp_c1, exp_c2 = st.columns(2)
            with exp_c1:
                st.markdown("#### Gated Fusion Modality Weighting")
                sg = res.get("structured_gate_mean")
                tg = res.get("text_gate_mean")
                if sg is not None and tg is not None:
                    st.write(f"- **Structured Branch Mean Gate Weight:** `{sg:.4f}`")
                    st.write(f"- **Clinical Text Branch Mean Gate Weight:** `{tg:.4f}`")
                    st.caption("The learned sigmoid gate dynamically weights each 24-dim branch representation per patient.")
                st.write(f"- **Missingness Flags Triggered:** `{len(missing_sim)}` ({', '.join(missing_sim) if missing_sim else 'None'})")

            with exp_c2:
                st.markdown("#### Key Physiological Input Summary")
                summary_items = {
                    "Age / Gender": f"{age:.0f} yrs / {gender}",
                    "Unit / Admission": f"{first_careunit} / {admission_type}",
                    "Lactate / Creatinine": f"{'MISSING (Imputed)' if 'lactate' in missing_sim else f'{lactate:.2f} mmol/L'} / {creat:.2f} mg/dL",
                    "SBP Min / SpO2 Min / GCS": f"{sbp_min:.0f} mmHg / {'MISSING' if 'spo2_min' in missing_sim else f'{spo2_min:.0f}%'} / {gcs_total}",
                }
                st.table(pd.DataFrame(summary_items.items(), columns=["Feature Group", "Value"]))

# ---------------------------------------------------------------------------
# PAGE 3: MODEL COMPARISON
# ---------------------------------------------------------------------------
elif page == "3. Model Comparison":
    st.subheader("Ablation & Modality Comparison (Held-Out Test Set)")
    if not eval_results:
        st.warning("No evaluation results found. Run training and evaluation on Page 6 first.")
    else:
        label_map = {
            "structured_only": "Structured-Only",
            "text_only": "Clinical-Text-Only",
            "multimodal_fusion": "Gated Multimodal Fusion",
        }
        table_rows = []
        for k, m in eval_results.items():
            cm = m["confusion_matrix"]
            table_rows.append(
                {
                    "Model": label_map.get(k, k),
                    "Accuracy": m["accuracy"],
                    "Precision": m["precision"],
                    "Recall": m["recall"],
                    "Specificity": m["specificity"],
                    "F1-Score": m["f1_score"],
                    "ROC-AUC": m["roc_auc"],
                    "PR-AUC": m["pr_auc"],
                    "Brier Score": m["brier_score"],
                    "Confusion Matrix (TN/FP/FN/TP)": f"{cm['tn']} / {cm['fp']} / {cm['fn']} / {cm['tp']}",
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            if (FIGURES_DIR / "fig_roc_curves.png").exists():
                st.image(str(FIGURES_DIR / "fig_roc_curves.png"), caption="ROC Curve Comparison Across Modalities")
            if (FIGURES_DIR / "fig_confusion_matrix.png").exists():
                st.image(str(FIGURES_DIR / "fig_confusion_matrix.png"), caption="Multimodal Fusion Confusion Matrix")
        with c2:
            if (FIGURES_DIR / "fig_pr_curves.png").exists():
                st.image(str(FIGURES_DIR / "fig_pr_curves.png"), caption="Precision-Recall Curve Comparison")
            if (FIGURES_DIR / "fig_model_comparison.png").exists():
                st.image(str(FIGURES_DIR / "fig_model_comparison.png"), caption="Grouped Metric Comparison")

        if eval_curves:
            st.markdown("### All-Model Confusion Matrices")
            cm_cols = st.columns(3)
            for idx, (k, m) in enumerate(eval_curves.items()):
                cm = m["confusion_matrix"]
                with cm_cols[idx]:
                    st.markdown(f"**{label_map.get(k, k)}**")
                    st.dataframe(
                        pd.DataFrame(
                            [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                            index=["Actual 0", "Actual 1"],
                            columns=["Pred 0", "Pred 1"],
                        ),
                        use_container_width=True,
                    )

# ---------------------------------------------------------------------------
# PAGE 4: MODEL ARCHITECTURE
# ---------------------------------------------------------------------------
elif page == "4. Model Architecture":
    st.subheader("Dual-Branch Sigmoid-Gated Multimodal Architecture")
    st.pyplot(render_architecture_diagram())

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("#### Branch A: Structured Clinical Data")
        st.markdown(
            """
            - **Inputs:** 21 continuous vitals/labs + 21 binary missingness flags + 3 one-hot categoricals (**52** total processed features).
            - **Leakage Control:** Train-only median imputation and Z-score standardization.
            - **Encoder:** Hidden layers `[64, 32]` with ReLU & Dropout (`0.25`) -> **24-dim** latent representation $h_s$.
            """
        )
    with a2:
        st.markdown("#### Branch B: Clinical Notes")
        st.markdown(
            """
            - **Text Cleaning:** Lowercasing, `[**...**]` -> `deid token`, abbreviation expansion (`HR`, `SBP`, `SpO2`, `Cr`, `GCS`, etc.).
            - **Primary Encoder:** `emilyalsentzer/Bio_ClinicalBERT` (max length `256`, 768-dim `[CLS]`).
            - **Offline Alternative:** Unigram/bigram TF-IDF (`1200` vocab) + `64`-dim Truncated SVD.
            - **Projection:** `32`-dim hidden -> **24-dim** latent representation $h_t$.
            """
        )
    with a3:
        st.markdown("#### Branch C: Gated Multimodal Fusion")
        st.markdown(
            """
            - **Concatenation:** $h_{cat} = [h_s ; h_t] \\in \\mathbb{R}^{48}$.
            - **Sigmoid Gate:** $g = \\sigma(W_g h_{cat} + b_g) \\in (0, 1)^{48}$.
            - **Modulation:** $h_{gated} = g \\odot h_{cat}$.
            - **Projection & Head:** $\\text{ReLU}(W_f h_{gated} + b_f) \\in \\mathbb{R}^{24} \\rightarrow \\sigma(W_o h_f + b_o) \\in [0, 1]$.
            """
        )

# ---------------------------------------------------------------------------
# PAGE 5: DATASET EXPLORER
# ---------------------------------------------------------------------------
elif page == "5. Dataset Explorer":
    st.subheader("Synthetic ICU Cohort Explorer (Fictional Records Only)")
    st.dataframe(df.head(20), use_container_width=True)

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("#### Missing Value Statistics")
        miss_series = df[cfg.numeric_features].isna().sum()
        miss_df = pd.DataFrame(
            {
                "Feature": miss_series.index,
                "Missing Count": miss_series.values,
                "Missing %": (miss_series.values / len(df) * 100).round(2),
            }
        ).sort_values("Missing Count", ascending=False)
        st.dataframe(miss_df, use_container_width=True, hide_index=True)

    with e2:
        st.markdown("#### Categorical Distributions")
        cat_choice = st.selectbox("Select Categorical Feature", cfg.categorical_features)
        cat_counts = df[cat_choice].value_counts()
        fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=130)
        ax.bar(cat_counts.index.astype(str), cat_counts.values, color="#1E3A5F")
        ax.set_title(f"Distribution of {cat_choice}", fontweight="bold", color="#0B192C")
        ax.set_ylabel("Stays")
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("#### Numerical Feature Distribution by Outcome")
    num_choice = st.selectbox("Select Numerical Feature", cfg.numeric_features, index=15)
    fig2, ax2 = plt.subplots(figsize=(8.0, 3.0), dpi=130)
    surv = df.loc[df["hospital_expire_flag"] == 0, num_choice].dropna()
    exp = df.loc[df["hospital_expire_flag"] == 1, num_choice].dropna()
    ax2.hist(surv, bins=30, alpha=0.65, label="Survived (0)", color="#1E3A5F", density=True)
    ax2.hist(exp, bins=30, alpha=0.65, label="Expired (1)", color="#DC2626", density=True)
    ax2.set_title(f"{num_choice} Density by In-Hospital Mortality", fontweight="bold", color="#0B192C")
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2)

    st.markdown("#### Sample Fictional Clinical Notes")
    for _, row in df[["stay_id", "hospital_expire_flag", "clinical_note"]].head(3).iterrows():
        label = "Expired (1)" if row["hospital_expire_flag"] == 1 else "Survived (0)"
        st.info(f"**Stay `{row['stay_id']}` | Outcome: {label}**\n\n{row['clinical_note']}")

# ---------------------------------------------------------------------------
# PAGE 6: TRAINING AND EVALUATION
# ---------------------------------------------------------------------------
elif page == "6. Training & Evaluation":
    st.subheader("Model Training, Early Stopping & Evaluation Control")
    st.write(
        "Training runs only when explicitly triggered below (never on page refresh). "
        "All 3 models (`Structured-Only`, `Clinical-Text-Only`, `Multimodal Fusion`) are trained "
        "with seed `42`, Adam (`lr=0.002`, `weight_decay=0.015`), class-weighted BCE (`cap=6.0`), "
        "and early stopping (`patience=12`)."
    )

    backend_choice = st.radio(
        "Select Execution Backend",
        ["numpy", "pytorch"],
        format_func=lambda x: "Offline NumPy/SciPy Neural Engine (TF-IDF + SVD)" if x == "numpy" else "PyTorch + Transformers Backend (requires torch)",
        horizontal=True,
    )

    if st.button("Train & Evaluate All 3 Models", type="primary"):
        with st.spinner("Training Structured-Only, Clinical-Text-Only, and Multimodal Fusion models..."):
            try:
                bundle = train_all_models(backend=backend_choice)
                results = evaluate_bundle()
                st.success(f"Training & evaluation complete using `{bundle['backend']}` backend!")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    if train_hist:
        st.markdown("#### Early Stopping & Convergence Summary")
        hist_rows = []
        for m_name, h in train_hist["histories"].items():
            best_ep = h.get("best_epoch", [len(h["train_loss"])])[0]
            hist_rows.append(
                {
                    "Model": m_name,
                    "Epochs Executed": len(h["train_loss"]),
                    "Best Val Epoch": best_ep,
                    "Final Train Loss": round(h["train_loss"][-1], 4),
                    "Best Val Loss": round(min(h["val_loss"]), 4),
                }
            )
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

        if (FIGURES_DIR / "fig_loss_curves.png").exists():
            st.image(str(FIGURES_DIR / "fig_loss_curves.png"), caption="Multimodal Fusion Training & Validation Loss Convergence")

    st.markdown("#### Saved Artifact Status")
    art_rows = [
        {"Artifact": "Model Bundle (multimodal_bundle.pkl)", "Exists": (MODELS_DIR / "multimodal_bundle.pkl").exists()},
        {"Artifact": "Evaluation Metrics (evaluation_results.json)", "Exists": (METRICS_DIR / "evaluation_results.json").exists()},
        {"Artifact": "Training Histories (training_histories.json)", "Exists": (METRICS_DIR / "training_histories.json").exists()},
    ]
    st.dataframe(pd.DataFrame(art_rows), use_container_width=True, hide_index=True)
