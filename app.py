"""
Thyroid Disease Classification — Clinical Decision Support System

Production-grade ML dashboard with:
- Real-time XGBoost prediction with SHAP waterfall explanations
- RAG-powered clinical Q&A over 25 indexed medical documents
- LLM-generated clinical reports (with template fallback)
- Differential diagnosis / counterfactual analysis
- Patient comparison against training population
- Batch prediction from CSV upload
- PDF report export

Author: Prasad Kanade | Northeastern University
Published: Springer. Leak-free evaluation: 85.09% accuracy, macro F1 0.7213
on 150,000 synthetic records against a 72.97% majority baseline.

Usage: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import io
from datetime import datetime

st.set_page_config(
    page_title="Thyroid Disease — Clinical Decision Support",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Clean clinical look */
    .stApp { font-family: 'Inter', -apple-system, sans-serif; }
    
    /* Diagnosis banners */
    .diagnosis-banner {
        padding: 20px 28px;
        border-radius: 12px;
        margin: 10px 0 20px 0;
        text-align: center;
    }
    .diagnosis-negative {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border: 1px solid #86efac;
        color: #166534;
    }
    .diagnosis-hypothyroid {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border: 1px solid #fbbf24;
        color: #92400e;
    }
    .diagnosis-hyperthyroid {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 1px solid #f87171;
        color: #991b1b;
    }
    .diagnosis-banner h2 { margin: 0; font-size: 28px; font-weight: 700; }
    .diagnosis-banner p { margin: 4px 0 0 0; font-size: 16px; opacity: 0.8; }
    
    /* Lab value indicators */
    .lab-indicator {
        display: flex; align-items: center; gap: 8px;
        padding: 4px 10px; border-radius: 6px; font-size: 13px; font-weight: 500;
        margin: 2px 0;
    }
    .lab-elevated { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
    .lab-low { background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; }
    .lab-normal { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }
    
    /* Patient summary card */
    .patient-card {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 16px 20px; margin: 10px 0;
    }
    .patient-card h4 { margin: 0 0 8px 0; color: #475569; font-size: 13px;
        text-transform: uppercase; letter-spacing: 1px; }
    .patient-card .values {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
    }
    .patient-card .val-item { font-size: 14px; }
    .patient-card .val-label { color: #94a3b8; font-size: 11px; }
    .patient-card .val-num { font-weight: 600; color: #1e293b; font-size: 16px; }
    
    /* Report styling */
    .clinical-report {
        background: #fafafa; border: 1px solid #e5e7eb; border-radius: 10px;
        padding: 20px 24px; margin: 12px 0; line-height: 1.7;
    }
    
    /* Gauge */
    .gauge-container { text-align: center; margin: 10px 0; }
    
    /* Section dividers */
    .section-divider {
        border: none; border-top: 1px solid #e2e8f0; margin: 24px 0;
    }
    
    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background: #f8fafc;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Motion: subtle, clinical */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .diagnosis-banner,
    .patient-card,
    .clinical-report,
    .cf-card {
        animation: fadeUp 0.35s ease-out both;
    }
    .patient-card, .cf-card {
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }
    .patient-card:hover, .cf-card:hover {
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
        transform: translateY(-1px);
    }
    .stButton > button {
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }
    @media (prefers-reduced-motion: reduce) {
        * { animation: none !important; transition: none !important; }
    }

    /* Counterfactual card */
    .cf-card {
        background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px;
        padding: 12px 16px; margin: 6px 0; font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────
REF_RANGES = {
    "TSH": (0.4, 4.0, "mIU/L"),
    "T3": (0.8, 2.0, "ng/dL"),
    "T4": (60, 120, "nmol/L"),
    "T4U": (0.7, 1.2, ""),
    "FTI": (60, 120, ""),
}

PRESET_PATIENTS = {
    "— Select preset —": {},
    "🟡 Hypothyroid Patient": {
        "TSH": 12.0, "T3": 0.8, "T4": 60.0, "T4U": 1.0,
        "age": 55, "sex": 0, "on_thyroxine": 1, "on_antithyroid": 0,
        "sick": 0, "pregnant": 0, "thyroid_surgery": 0,
    },
    "🔴 Hyperthyroid Patient": {
        "TSH": 0.1, "T3": 5.2, "T4": 180.0, "T4U": 1.3,
        "age": 35, "sex": 0, "on_thyroxine": 0, "on_antithyroid": 1,
        "sick": 0, "pregnant": 0, "thyroid_surgery": 0,
    },
    "🟢 Healthy Patient": {
        "TSH": 2.5, "T3": 1.8, "T4": 105.0, "T4U": 1.0,
        "age": 30, "sex": 1, "on_thyroxine": 0, "on_antithyroid": 0,
        "sick": 0, "pregnant": 0, "thyroid_surgery": 0,
    },
    "⚪ Borderline / Subclinical": {
        "TSH": 5.5, "T3": 1.5, "T4": 85.0, "T4U": 0.9,
        "age": 62, "sex": 0, "on_thyroxine": 0, "on_antithyroid": 0,
        "sick": 0, "pregnant": 0, "thyroid_surgery": 0,
    },
    "🤰 Pregnant Patient": {
        "TSH": 0.3, "T3": 1.9, "T4": 115.0, "T4U": 1.1,
        "age": 29, "sex": 0, "on_thyroxine": 0, "on_antithyroid": 0,
        "sick": 0, "pregnant": 1, "thyroid_surgery": 0,
    },
}


# ── Model Loading ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    models_dir = "models"
    meta_files = sorted(
        [f for f in os.listdir(models_dir) if f.startswith("metadata_")], reverse=True
    )
    if not meta_files:
        st.error("No trained model found. Run `python train.py` first.")
        st.stop()
    timestamp = meta_files[0].replace("metadata_", "").replace(".json", "")
    try:
        model = joblib.load(f"{models_dir}/best_model_{timestamp}.pkl")
        scaler = joblib.load(f"{models_dir}/scaler_{timestamp}.pkl")
        label_encoder = joblib.load(f"{models_dir}/label_encoder_{timestamp}.pkl")
    except Exception as e:
        st.error(f"Failed to load model: {e}. Run `python train.py` to retrain.")
        st.stop()
    with open(f"{models_dir}/metadata_{timestamp}.json") as f:
        metadata = json.load(f)
    # Extract XGBoost for SHAP
    xgb_model = None
    if hasattr(model, 'named_estimators_'):
        for name, est in model.named_estimators_.items():
            if 'XGB' in type(est).__name__:
                xgb_model = est
                break
    if xgb_model is None:
        xgb_model = model
    return model, xgb_model, scaler, label_encoder, metadata


@st.cache_resource
def load_rag():
    try:
        from rag.retriever import ThyroidRetriever
        return ThyroidRetriever()
    except Exception:
        return None


@st.cache_resource
def load_report_generator():
    try:
        from llm.report_generator import ClinicalReportGenerator
        return ClinicalReportGenerator()
    except Exception:
        return None


@st.cache_resource
def load_qa():
    try:
        from llm.clinical_qa import ClinicalQA
        return ClinicalQA()
    except Exception:
        return None


@st.cache_data
def load_training_data():
    """Load training data for patient comparison."""
    try:
        return pd.read_csv("data/training_stats.csv")
    except Exception:
        return None


# ── Helper Functions ──────────────────────────────────────────────────────
def get_shap_values(xgb_model, input_scaled, features, prediction_idx):
    try:
        import shap
        explainer = shap.TreeExplainer(xgb_model)
        shap_vals = explainer.shap_values(input_scaled)
        idx = int(prediction_idx)
        if isinstance(shap_vals, list):
            sv = np.array(shap_vals[idx]).flatten()
        elif isinstance(shap_vals, np.ndarray):
            if shap_vals.ndim == 3:
                sv = shap_vals[0, :, idx]
            elif shap_vals.ndim == 2:
                sv = shap_vals[0]
            else:
                sv = shap_vals.flatten()
        else:
            return None
        if len(sv) == len(features):
            return dict(zip(features, [float(v) for v in sv]))
        return None
    except Exception:
        return None


def lab_indicator_html(marker, value, ref_range):
    low, high, unit = ref_range
    unit_str = f" {unit}" if unit else ""
    if value > high:
        return f'<div class="lab-indicator lab-elevated">⬆ {marker}: {value:.2f}{unit_str} (High, ref: {low}-{high})</div>'
    elif value < low:
        return f'<div class="lab-indicator lab-low">⬇ {marker}: {value:.2f}{unit_str} (Low, ref: {low}-{high})</div>'
    else:
        return f'<div class="lab-indicator lab-normal">✓ {marker}: {value:.2f}{unit_str} (Normal)</div>'


def diagnosis_banner_html(prediction, confidence):
    css_class = f"diagnosis-{prediction}"
    emoji = {"negative": "✅", "hypothyroid": "⚠️", "hyperthyroid": "🔴"}.get(prediction, "❓")
    label = prediction.upper()
    return f"""
    <div class="diagnosis-banner {css_class}">
        <h2>{emoji} {label}</h2>
        <p>Model confidence: {confidence:.1%}</p>
    </div>
    """


def confidence_gauge_html(confidence):
    pct = confidence * 100
    color = "#16a34a" if pct > 90 else "#eab308" if pct > 70 else "#dc2626"
    return f"""
    <div class="gauge-container">
        <svg width="200" height="120" viewBox="0 0 200 120">
            <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#e5e7eb" stroke-width="12" stroke-linecap="round"/>
            <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round"
                  stroke-dasharray="{pct * 2.51} 251" />
            <text x="100" y="85" text-anchor="middle" font-size="28" font-weight="700" fill="{color}">{pct:.1f}%</text>
            <text x="100" y="105" text-anchor="middle" font-size="11" fill="#94a3b8">CONFIDENCE</text>
        </svg>
    </div>
    """


def compute_counterfactuals(model, scaler, label_encoder, features, patient, current_prediction):
    """Find what changes would flip the prediction."""
    counterfactuals = []
    key_markers = ["TSH", "T3", "T4"]

    for marker in key_markers:
        if marker not in patient:
            continue
        original = patient[marker]
        # Try range of values
        if marker == "TSH":
            test_values = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 25.0]
        elif marker == "T3":
            test_values = [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        elif marker == "T4":
            test_values = [40, 60, 80, 100, 120, 140, 160, 180, 200]
        else:
            continue

        for val in test_values:
            if abs(val - original) < 0.1:
                continue
            modified = patient.copy()
            modified[marker] = val
            if "FTI" in features:
                modified["FTI"] = modified.get("T4", 105) / (modified.get("T4U", 1.0) + 0.01)

            input_df = pd.DataFrame([{f: modified.get(f, 0) for f in features}])
            input_scaled = scaler.transform(input_df)
            pred_idx = model.predict(input_scaled)[0]
            new_pred = label_encoder.inverse_transform([int(pred_idx)])[0]

            if new_pred != current_prediction:
                direction = "increased" if val > original else "decreased"
                counterfactuals.append({
                    "marker": marker,
                    "from_val": original,
                    "to_val": val,
                    "direction": direction,
                    "new_prediction": new_pred,
                })
                break  # Take first flip point per marker

    return counterfactuals


# ── Load Resources ────────────────────────────────────────────────────────
model, xgb_model, scaler, label_encoder, metadata = load_model()
features = metadata["features_selected"]
if isinstance(features, int):
    features = [f"feature_{i}" for i in range(features)]

# Lazy-loaded inside pages that need them
rag_sys = None
report_gen = None
api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.markdown("### Thyroid CDSS")
st.sidebar.caption("Clinical Decision Support System")
st.sidebar.markdown("---")
best_metrics = metadata["performance_metrics"][metadata["best_model"]]
st.sidebar.metric("Model Accuracy", f"{best_metrics['accuracy']:.1%}")
st.sidebar.markdown(f"**Model:** {metadata['best_model']}")
st.sidebar.markdown(f"**Features:** {len(features)}")
st.sidebar.markdown(f"**Dataset:** {metadata['dataset_size']:,} patients")
st.sidebar.markdown("---")
st.sidebar.caption("RAG index: 25 documents")
st.sidebar.caption(f"LLM: {'API connected' if api_key else 'Template mode'}")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Predict & Explain", "Clinical Q&A", "Model Performance",
     "Batch Prediction", "About & Methodology"],
)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1: Predict & Explain
# ══════════════════════════════════════════════════════════════════════════
if page == "Predict & Explain":
    report_gen = load_report_generator()
    st.title("Thyroid Disease Prediction")
    st.markdown("Enter patient lab values for ML-powered classification with explainability.")

    preset = st.selectbox("Quick-fill preset:", list(PRESET_PATIENTS.keys()))
    pv = PRESET_PATIENTS[preset]

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("#### Lab Values")
        patient = {}
        patient["TSH"] = st.slider("TSH (mIU/L)", 0.01, 50.0, pv.get("TSH", 2.5), 0.1,
                                   help="Normal: 0.4-4.0 mIU/L")
        # Inline reference indicator
        st.markdown(lab_indicator_html("TSH", patient["TSH"], REF_RANGES["TSH"]), unsafe_allow_html=True)

        patient["T3"] = st.slider("T3 (ng/dL)", 0.4, 6.5, pv.get("T3", 1.8), 0.1,
                                  help="Normal: 0.8-2.0 ng/dL")
        st.markdown(lab_indicator_html("T3", patient["T3"], REF_RANGES["T3"]), unsafe_allow_html=True)

        patient["T4"] = st.slider("T4 (nmol/L)", 35.0, 230.0, pv.get("T4", 105.0), 1.0,
                                  help="Normal: 60-120 nmol/L")
        st.markdown(lab_indicator_html("T4", patient["T4"], REF_RANGES["T4"]), unsafe_allow_html=True)

        patient["T4U"] = st.slider("T4 Uptake", 0.5, 1.8, pv.get("T4U", 1.0), 0.05)
        patient["FTI"] = patient["T4"] / (patient["T4U"] + 0.01)

        st.markdown("#### Demographics")
        c1, c2 = st.columns(2)
        patient["age"] = c1.slider("Age", 18, 90, pv.get("age", 45))
        patient["sex"] = c2.selectbox("Sex", [0, 1], index=pv.get("sex", 0),
                                       format_func=lambda x: "Female" if x == 0 else "Male")

        st.markdown("#### Medical History")
        c3, c4 = st.columns(2)
        patient["on_thyroxine"] = c3.selectbox("On Thyroxine?", [0, 1],
                                                index=pv.get("on_thyroxine", 0),
                                                format_func=lambda x: "No" if x == 0 else "Yes")
        patient["on_antithyroid"] = c4.selectbox("On Antithyroid?", [0, 1],
                                                  index=pv.get("on_antithyroid", 0),
                                                  format_func=lambda x: "No" if x == 0 else "Yes")
        c5, c6, c7 = st.columns(3)
        patient["sick"] = c5.selectbox("Sick?", [0, 1], index=pv.get("sick", 0),
                                       format_func=lambda x: "No" if x == 0 else "Yes")
        patient["pregnant"] = c6.selectbox("Pregnant?", [0, 1], index=pv.get("pregnant", 0),
                                           format_func=lambda x: "No" if x == 0 else "Yes")
        patient["thyroid_surgery"] = c7.selectbox("Surgery?", [0, 1],
                                                   index=pv.get("thyroid_surgery", 0),
                                                   format_func=lambda x: "No" if x == 0 else "Yes")

    for f in features:
        if f not in patient:
            patient[f] = 0

    predict_btn = st.button("Analyze Patient", type="primary", use_container_width=True)

    if predict_btn:
        input_df = pd.DataFrame([{f: patient.get(f, 0) for f in features}])
        input_scaled = scaler.transform(input_df)
        prediction_idx = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        prediction = label_encoder.inverse_transform([int(prediction_idx)])[0]
        confidence = float(probabilities[int(prediction_idx)])

        with col_result:
            # Diagnosis banner
            st.markdown(diagnosis_banner_html(prediction, confidence), unsafe_allow_html=True)

            # Confidence gauge
            st.markdown(confidence_gauge_html(confidence), unsafe_allow_html=True)

            # Probabilities
            st.markdown("#### Class Probabilities")
            for cls, prob in sorted(zip(label_encoder.classes_, probabilities),
                                     key=lambda x: x[1], reverse=True):
                st.progress(float(prob), text=f"{cls}: {float(prob):.1%}")

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # SHAP
            st.markdown("#### Feature Contributions (SHAP)")
            shap_dict = get_shap_values(xgb_model, input_scaled, features, prediction_idx)

            if shap_dict and any(abs(v) > 0.001 for v in shap_dict.values()):
                sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
                shap_df = pd.DataFrame(sorted_shap, columns=["Feature", "SHAP Value"])
                # Color positive/negative
                colors = ["#dc2626" if v > 0 else "#2563eb" for _, v in sorted_shap]
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 3.5))
                bars = ax.barh(
                    [f[0] for f in reversed(sorted_shap)],
                    [f[1] for f in reversed(sorted_shap)],
                    color=[c for c in reversed(colors)],
                    edgecolor="none", height=0.6
                )
                ax.set_xlabel("SHAP Contribution", fontsize=10)
                ax.axvline(0, color="#94a3b8", linewidth=0.5)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(labelsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.markdown("*SHAP unavailable — showing reference range analysis:*")
                for marker, ref in REF_RANGES.items():
                    val = patient.get(marker)
                    if val is not None:
                        st.markdown(lab_indicator_html(marker, val, ref), unsafe_allow_html=True)
                shap_dict = None

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # Differential / Counterfactual
            st.markdown("#### Differential Analysis")
            st.markdown("*What would need to change for a different diagnosis?*")
            cfs = compute_counterfactuals(model, scaler, label_encoder, features, patient, prediction)
            if cfs:
                for cf in cfs:
                    ref = REF_RANGES.get(cf["marker"], (None, None, ""))
                    unit = ref[2] if ref else ""
                    st.markdown(
                        f'<div class="cf-card">If <b>{cf["marker"]}</b> {cf["direction"]} from '
                        f'{cf["from_val"]:.1f} → <b>{cf["to_val"]:.1f}</b> {unit}, '
                        f'prediction would change to <b>{cf["new_prediction"].upper()}</b></div>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown("*Model is highly confident — no single marker change would flip the diagnosis.*")

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # Patient comparison
            st.markdown("#### Patient vs Population")
            train_df = load_training_data()
            if train_df is not None:
                class_data = train_df[train_df["target"] == prediction]
                comp_data = []
                for marker in ["TSH", "T3", "T4"]:
                    if marker in class_data.columns:
                        pop_mean = class_data[marker].mean()
                        pop_std = class_data[marker].std()
                        pat_val = patient.get(marker, 0)
                        z = (pat_val - pop_mean) / (pop_std + 0.001)
                        comp_data.append({
                            "Marker": marker,
                            "Patient": f"{pat_val:.2f}",
                            f"Avg ({prediction})": f"{pop_mean:.2f}",
                            "Z-score": f"{z:+.1f}",
                        })
                if comp_data:
                    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # Clinical Report
            st.markdown("#### Clinical Report")
            rag_context = ""
            if rag_sys:
                rag_context = rag_sys.get_context(
                    f"thyroid {prediction} TSH {patient.get('TSH', '')} diagnosis", n_results=3
                )
            if report_gen:
                report = report_gen.generate(
                    patient_data=patient, prediction=prediction,
                    confidence=confidence,
                    shap_values=shap_dict if shap_dict else {f: 0.0 for f in features},
                    rag_context=rag_context,
                )
                if "[LLM unavailable" in report:
                    report = report_gen.generate_from_data(patient, prediction, confidence, REF_RANGES)
                st.markdown(report)

                # PDF export
                report_text = f"THYROID CLINICAL REPORT\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{report}"
                st.download_button(
                    "📄 Download Report",
                    data=report_text,
                    file_name=f"thyroid_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                )


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2: Clinical Q&A
# ══════════════════════════════════════════════════════════════════════════
elif page == "Clinical Q&A":
    rag_sys = load_rag()
    report_gen = load_report_generator()
    qa = load_qa()
    st.title("Clinical Q&A")
    st.markdown("Ask questions about thyroid disease. Answers grounded in **25 indexed medical documents**.")

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    if "qa_question" not in st.session_state:
        st.session_state.qa_question = ""

    examples = [
        "What causes elevated TSH levels?",
        "How does SMOTE improve thyroid classification?",
        "What is the role of SHAP in clinical ML?",
        "Treatment options for hyperthyroidism?",
        "How does pregnancy affect thyroid function?",
        "What are subclinical thyroid disorders?",
        "How does XGBoost handle class imbalance?",
        "What is the Free Thyroxine Index?",
    ]

    col_q, col_ex = st.columns([2, 1])
    with col_ex:
        st.markdown("**Quick questions:**")
        for ex in examples[:4]:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state.qa_question = ex
                st.rerun()
    with col_q:
        question = st.text_input(
            "Ask a clinical question:",
            value=st.session_state.qa_question,
            placeholder="e.g., What causes elevated TSH?",
        )

    if question:
        qa = load_qa()
        if qa:
            with st.spinner("Searching medical literature..."):
                result = qa.ask(question)

            st.markdown("### Answer")
            st.markdown(result["answer"])

            st.markdown("### Sources")
            for src in result["sources"]:
                st.markdown(f"📄 **{src['title']}** — *{src['source']}*")

            # Suggested follow-ups
            follow_ups = {
                "TSH": ["What is the relationship between TSH and T4?", "How do age-specific TSH ranges differ?"],
                "hypothyroid": ["What is Hashimoto thyroiditis?", "How is levothyroxine dosing determined?"],
                "hyperthyroid": ["What is Graves disease?", "What are the treatment options?"],
                "SHAP": ["How do counterfactual explanations work?", "Why is interpretability important in medical AI?"],
                "SMOTE": ["What other techniques handle class imbalance?", "How does SMOTE affect model calibration?"],
                "pregnancy": ["What are trimester-specific TSH ranges?", "Risks of untreated hypothyroidism in pregnancy?"],
            }
            suggested = []
            for kw, qs in follow_ups.items():
                if kw.lower() in question.lower():
                    suggested.extend(qs)
            if suggested:
                st.markdown("### 💡 Related Questions")
                for sq in suggested[:3]:
                    st.markdown(f"→ *{sq}*")

            st.session_state.qa_history.append({"q": question, "a": result["answer"]})
        else:
            if rag_sys:
                docs = rag_sys.search(question)
                st.markdown("### Retrieved Literature")
                for doc in docs:
                    with st.expander(f"📄 {doc['title']}"):
                        st.markdown(f"*{doc['source']}*")
                        st.markdown(doc["text"])
            else:
                st.error("RAG system unavailable. Install: `pip install chromadb`")

    if st.session_state.qa_history:
        with st.expander("📝 Conversation History"):
            for item in reversed(st.session_state.qa_history[-5:]):
                st.markdown(f"**Q:** {item['q']}")
                st.markdown(f"**A:** {item['a'][:200]}...")
                st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3: Model Performance
# ══════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("Model Performance")

    c1, c2, c3, c4 = st.columns(4)
    best = metadata["best_model"]
    m = metadata["performance_metrics"][best]
    c1.metric("Best Model", best)
    c2.metric("Accuracy", f"{m['accuracy']:.1%}")
    c3.metric("Precision", f"{m['precision']:.1%}")
    c4.metric("F1 Score", f"{m['f1_score']:.1%}")

    st.markdown("### All Models Comparison")
    perf_df = pd.DataFrame(metadata["performance_metrics"]).T
    st.dataframe(perf_df.style.format("{:.4f}").highlight_max(axis=0, color="#dcfce7"),
                 use_container_width=True)

    # Bar chart comparison
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    perf_df.plot(kind="bar", ax=ax, edgecolor="none", width=0.7)
    # Was ylim(0.95, 1.0), tuned for the pre-correction scores. The
    # leak-free run sits at 0.72-0.85, which that floor would clip entirely.
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Training visualizations
    plots_dir = "outputs/plots"
    if os.path.exists(plots_dir):
        st.markdown("### Training Visualizations")
        images = sorted([f for f in os.listdir(plots_dir) if f.endswith(".png")])
        if images:
            tabs = st.tabs([img.replace("_", " ").replace(".png", "").title() for img in images])
            for tab, img in zip(tabs, images):
                with tab:
                    st.image(f"{plots_dir}/{img}", use_container_width=True)

    # Dataset distribution
    st.markdown("### Dataset Overview")
    train_df = load_training_data()
    if train_df is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Class Distribution**")
            class_counts = train_df["target"].value_counts()
            st.bar_chart(class_counts)
        with c2:
            st.markdown("**Feature Statistics**")
            desc = train_df[["TSH", "T3", "T4", "age"]].describe().round(2)
            st.dataframe(desc, use_container_width=True)

    st.markdown("### Selected Features")
    st.markdown(", ".join(f"`{f}`" for f in features))


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4: Batch Prediction
# ══════════════════════════════════════════════════════════════════════════
elif page == "Batch Prediction":
    st.title("📁 Batch Prediction")
    st.markdown("Upload a CSV of patient data to get predictions for all patients at once.")

    st.markdown("**Required columns:** `TSH`, `T3`, `T4`, `T4U`, `age`, `sex`")
    st.markdown("**Optional:** `on_thyroxine`, `on_antithyroid`, `sick`, `pregnant`, `thyroid_surgery`, `goitre`")

    # Sample CSV download
    sample = pd.DataFrame([
        {"TSH": 12.0, "T3": 0.8, "T4": 60.0, "T4U": 1.0, "age": 55, "sex": 0},
        {"TSH": 0.1, "T3": 5.2, "T4": 180.0, "T4U": 1.3, "age": 35, "sex": 0},
        {"TSH": 2.5, "T3": 1.8, "T4": 105.0, "T4U": 1.0, "age": 30, "sex": 1},
    ])
    st.download_button("📥 Download sample CSV", data=sample.to_csv(index=False),
                       file_name="sample_patients.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload patient CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = pd.DataFrame()

        if df.empty:
            st.warning("The uploaded CSV is empty or could not be parsed.")
            df = None

        if df is not None:
            required = ["TSH", "T3", "T4"]
            missing_req = [c for c in required if c not in df.columns]
            if missing_req:
                st.error(f"Missing required columns: {', '.join(missing_req)}")
                df = None

        if df is not None:
            for col in df.columns:
                if col in features:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.fillna(0)
        st.markdown(f"**Loaded {len(df)} patients**")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🔍 Predict All", type="primary"):
            # Fill missing columns
            for f in features:
                if f not in df.columns:
                    if f == "FTI" and "T4" in df.columns and "T4U" in df.columns:
                        df["FTI"] = df["T4"] / (df["T4U"] + 0.01)
                    elif f == "T4U":
                        df[f] = 1.0
                    else:
                        df[f] = 0

            input_df = df[features].fillna(0)

            # ── Data Drift Detection ──
            train_df_drift = load_training_data()
            if train_df_drift is not None:
                drift_warnings = []
                for col in ["TSH", "T3", "T4"]:
                    if col in input_df.columns and col in train_df_drift.columns:
                        train_mean = train_df_drift[col].mean()
                        train_std = train_df_drift[col].std()
                        input_mean = input_df[col].mean()
                        if train_std > 0:
                            z = abs(input_mean - train_mean) / train_std
                            if z > 2.0:
                                drift_warnings.append(
                                    f"**{col}**: uploaded mean ({input_mean:.2f}) is "
                                    f"{z:.1f}σ from training mean ({train_mean:.2f})"
                                )
                if drift_warnings:
                    st.warning(
                        "⚠️ **Data Distribution Warning**: Some features differ "
                        "significantly from training data. Predictions may be less reliable.\n\n"
                        + "\n".join(f"- {w}" for w in drift_warnings)
                    )

            input_scaled = scaler.transform(input_df)
            predictions = model.predict(input_scaled)
            probas = model.predict_proba(input_scaled)

            df["Prediction"] = label_encoder.inverse_transform(predictions)
            df["Confidence"] = [float(probas[i][int(predictions[i])]) for i in range(len(predictions))]

            # Color code results
            st.markdown("### Results")
            st.dataframe(
                df[["TSH", "T3", "T4", "age", "sex", "Prediction", "Confidence"]].style.format(
                    {"Confidence": "{:.1%}", "TSH": "{:.2f}", "T3": "{:.2f}", "T4": "{:.1f}"}
                ),
                use_container_width=True,
            )

            # Summary
            st.markdown("### Summary")
            summary = df["Prediction"].value_counts()
            st.bar_chart(summary)

            # Download results
            st.download_button(
                "📥 Download Results CSV",
                data=df.to_csv(index=False),
                file_name=f"thyroid_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5: About & Methodology
# ══════════════════════════════════════════════════════════════════════════
elif page == "About & Methodology":
    st.title("ℹ️ About This System")

    st.markdown("""
    ### Thyroid Disease Classification — Clinical Decision Support System

    This system uses machine learning to assist clinicians in thyroid disease diagnosis.
    It combines a high-accuracy Random Forest classifier with explainable AI and retrieval-augmented
    generation to provide transparent, evidence-based clinical decision support.

    ---

    ### 🔬 How It Works

    **1. Classification Model**

    An XGBoost classifier, selected on validation macro F1 and trained on 150,000 synthetic records,
    classifies patients into three categories: **Negative** (euthyroid), **Hypothyroid**, and
    **Hyperthyroid**. It reaches **85.09% accuracy and macro F1 0.7213** against a 72.97%
    majority baseline, with SMOTE applied inside cross-validation folds.

    Per-class recall is published deliberately: 0.98 for Negative but **0.44 for
    Hyperthyroid and 0.55 for Hypothyroid**. Overall accuracy is carried by the
    majority class, which is the opposite of what a screening tool wants, and the
    aggregate alone would hide that.

    **2. Feature Selection**

    Recursive Feature Elimination (RFE), fit on the training fold only, reduces the original 17 clinical features to 12
    most discriminative features, maintaining performance while improving interpretability.
    Key features include TSH, T3, T4, FTI, age, and treatment status.

    **3. SHAP Explainability**

    Every prediction is accompanied by SHAP (SHapley Additive exPlanations) values showing
    exactly how each feature contributes to the diagnosis. This enables clinicians to verify
    that the model's reasoning aligns with clinical knowledge.

    **4. RAG-Powered Knowledge Base**

    25 indexed medical documents covering thyroid physiology, diagnosis, treatment, and ML
    methodology provide evidence-based context. Semantic search via ChromaDB retrieves
    relevant literature for both clinical reports and the Q&A system.

    **5. Clinical Reports**

    Natural language reports interpret predictions in clinical terms, comparing lab values
    against reference ranges and providing recommended next steps. When an LLM API key
    is configured, reports are generated by Claude or GPT-4; otherwise, a clinically-validated
    template system is used.

    **6. Differential Analysis**

    Counterfactual explanations show what changes in lab values would alter the diagnosis,
    helping clinicians understand decision boundaries and borderline cases.

    ---

    ### 📊 Architecture
    """)

    st.code("""
    Patient Data
        → Feature Selection (RFE: 19 → 12 features)
        → SMOTE Class Balancing (3:1 → 1:1)
        → XGBoost + Random Forest + Ensemble (Soft Voting)
        → SHAP Explainability (per-patient feature contributions)
        → RAG Retrieval (ChromaDB + 25 medical documents)
        → LLM Report Generation (Claude / GPT-4 / Template)
        → Streamlit Clinical Dashboard
    """, language="text")

    st.markdown("""
    ---

    ### 📄 Publication

    **Classification and Diagnosis of Thyroid Disease Using XGBoost and SHAP**
    *Springer Conference Proceedings, March 2024*
    [Read Paper →](https://link.springer.com/chapter/10.1007/978-981-97-6106-7_9)

    ---

    ### 👨‍💻 Author

    **Prasad Kanade**
    MS Computer Science, Northeastern University
    [GitHub](https://github.com/prasad0411) · [LinkedIn](https://linkedin.com/in/prasad-kanade-/) · kanade.pra@northeastern.edu

    ---

    *This system is intended for research and educational purposes. It is not a substitute
    for professional medical judgment. All clinical decisions should be made by qualified
    healthcare providers.*
    """)
