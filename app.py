import os
import json
import warnings

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI

warnings.filterwarnings("ignore")
load_dotenv()

st.set_page_config(
    page_title="Symptom Checker",
    page_icon="🩺",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide Streamlit default header/footer */
    #MainMenu, footer, header {visibility: hidden;}

    /* Page background */
    .stApp {background: #f7f9fc;}

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem 2rem 2rem;
        color: white;
        margin-bottom: 1.8rem;
        text-align: center;
    }
    .hero h1 {font-size: 2.4rem; font-weight: 700; margin: 0 0 0.4rem 0;}
    .hero p  {font-size: 1.05rem; margin: 0; opacity: 0.88;}

    /* Card wrapper */
    .card {
        background: white;
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 1.2rem;
    }

    /* Result disease name */
    .result-disease {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a73e8;
        margin: 0.3rem 0 0.1rem 0;
    }

    /* Matched symptoms pills */
    .pill {
        display: inline-block;
        background: #e8f0fe;
        color: #1a73e8;
        border-radius: 20px;
        padding: 3px 12px;
        margin: 3px 3px;
        font-size: 0.82rem;
        font-weight: 500;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.55rem 1rem;
    }

    /* Text area */
    .stTextArea textarea {
        border-radius: 10px;
        font-size: 0.95rem;
    }

    /* Multiselect */
    .stMultiSelect > div {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

MODEL_PATH    = "naive_bayes_model.pkl"
FEATURES_PATH = "features.json"


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Starting up…")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Starting up…")
def load_features():
    with open(FEATURES_PATH) as f:
        return json.load(f)


@st.cache_resource(show_spinner="Starting up…")
def load_explainer(_model, n_features):
    bg = np.zeros((1, n_features))
    return shap.KernelExplainer(_model.predict_proba, bg, silent=True)


model       = load_model()
feature_cols = load_features()
explainer   = load_explainer(model, len(feature_cols))


# ── OpenAI extraction ─────────────────────────────────────────────────────────

def extract_symptoms_with_openai(user_text: str, api_key: str) -> list[str]:
    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a clinical symptom extractor. "
        "The user will describe how they feel in any language. "
        "Map their description to symptoms from the list below.\n\n"
        "Rules:\n"
        "1. Return ONLY a valid JSON array of strings — no explanation, no markdown.\n"
        "2. Each string must be copied EXACTLY from the list (spelling, spacing).\n"
        "3. Only include symptoms clearly mentioned. Return [] if nothing matches.\n\n"
        f"Symptom list:\n{json.dumps(feature_cols)}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_text},
        ],
        temperature=0,
        max_tokens=500,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    matched = json.loads(raw)
    return [s for s in matched if s in feature_cols]


# ── Prediction + explanation renderer ────────────────────────────────────────

def render_results(symptoms: list[str]):
    x = np.zeros((1, len(feature_cols)))
    for sym in symptoms:
        if sym in feature_cols:
            x[0, feature_cols.index(sym)] = 1

    proba    = model.predict_proba(x)[0]
    top_idx  = np.argsort(proba)[::-1][:5]
    top_list = [(model.classes_[i], proba[i]) for i in top_idx]
    pred_disease = top_list[0][0].title()
    pred_idx     = top_idx[0]

    # ── Main prediction card ───────────────────────────────────────────────
    st.markdown(f"""
    <div class="card">
        <div style="color:#6b7280;font-size:0.88rem;font-weight:600;letter-spacing:.05em;
                    text-transform:uppercase;">Most likely condition</div>
        <div class="result-disease">{pred_disease}</div>
        <div style="color:#6b7280;font-size:0.9rem;margin-top:0.3rem;">
            Confidence: <b style="color:#1a73e8">{top_list[0][1]*100:.1f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top 5 alternatives ─────────────────────────────────────────────────
    with st.expander("Other possible conditions", expanded=True):
        df_top = pd.DataFrame(top_list[1:], columns=["Condition", "Probability"])
        df_top["Condition"]   = df_top["Condition"].str.title()
        df_top["Probability"] = df_top["Probability"].apply(lambda v: f"{v*100:.1f}%")
        st.dataframe(df_top, use_container_width=True, hide_index=True)

    # ── Key symptoms chart ─────────────────────────────────────────────────
    st.markdown("#### What's driving this result?")
    st.caption("Symptoms highlighted in red increased the likelihood of this condition.")

    with st.spinner("Analysing…"):
        sv      = explainer.shap_values(x, nsamples=100)
        sv_cls  = sv[0, :, pred_idx]

    top_n   = 12
    abs_idx = np.argsort(np.abs(sv_cls))[::-1][:top_n]
    feats   = [feature_cols[i].replace("_", " ").title() for i in abs_idx]
    vals    = sv_cls[abs_idx]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colors = ["#ef4444" if v > 0 else "#60a5fa" for v in vals[::-1]]
    bars   = ax.barh(feats[::-1], vals[::-1], color=colors, height=0.6, edgecolor="none")
    ax.axvline(0, color="#374151", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Influence on prediction", fontsize=9, color="#6b7280")
    ax.tick_params(axis="y", labelsize=9, colors="#374151")
    ax.tick_params(axis="x", labelsize=8, colors="#9ca3af")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color="#f3f4f6", linewidth=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════════════════════

if "manual_symptoms" not in st.session_state:
    st.session_state.manual_symptoms = []
if "ai_matched" not in st.session_state:
    st.session_state.ai_matched = []
if "ai_error" not in st.session_state:
    st.session_state.ai_error = ""


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <h1>🩺 Symptom Checker</h1>
    <p>Describe or select your symptoms and get an instant assessment of possible conditions.</p>
</div>
""", unsafe_allow_html=True)

tab_manual, tab_ai = st.tabs(["  Select symptoms  ", "  Describe in your own words  "])


# ── Tab 1: Manual ─────────────────────────────────────────────────────────────
with tab_manual:
    col_l, col_r = st.columns([1, 2], gap="large")

    with col_l:
        st.markdown("#### Choose your symptoms")
        symptoms_manual = st.multiselect(
            label="Symptoms",
            options=feature_cols,
            placeholder="Type to search symptoms…",
            label_visibility="collapsed",
        )
        if symptoms_manual:
            st.markdown(
                "**Selected:** " +
                "".join(f'<span class="pill">{s}</span>' for s in symptoms_manual),
                unsafe_allow_html=True,
            )
        predict_btn = st.button(
            "Check symptoms", key="btn_manual",
            use_container_width=True, type="primary",
        )
        if predict_btn:
            if not symptoms_manual:
                st.warning("Please select at least one symptom.")
                st.session_state.manual_symptoms = []
            else:
                st.session_state.manual_symptoms = list(symptoms_manual)

    with col_r:
        if st.session_state.manual_symptoms:
            render_results(st.session_state.manual_symptoms)
        else:
            st.markdown("""
            <div style="text-align:center;padding:4rem 1rem;color:#9ca3af;">
                <div style="font-size:3rem;">💊</div>
                <div style="font-size:1rem;margin-top:0.6rem;">
                    Select symptoms and click <b>Check symptoms</b>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── Tab 2: AI ────────────────────────────────────────────────────────────────
with tab_ai:
    col_l, col_r = st.columns([1, 2], gap="large")

    with col_l:
        st.markdown("#### Tell us how you feel")
        st.caption("You can write in English or Russian.")

        env_key = os.getenv("OPENAI_API_KEY", "")
        if env_key:
            api_key = env_key
        else:
            api_key = st.text_input(
                "API Key", type="password",
                placeholder="Enter your OpenAI key…",
                label_visibility="collapsed",
            )

        description = st.text_area(
            label="Description",
            placeholder=(
                "e.g. I have had a sore throat, dry cough and mild fever for three days. "
                "I also feel very tired and have a slight headache.\n\n"
                "or: У меня болит горло, сухой кашель и небольшая температура уже три дня."
            ),
            height=160,
            label_visibility="collapsed",
        )

        ai_btn = st.button(
            "Analyse & predict", key="btn_ai",
            use_container_width=True, type="primary",
        )

        if ai_btn:
            if not api_key:
                st.warning("Please enter an API key to use this feature.")
            elif not description.strip():
                st.warning("Please describe your symptoms first.")
            else:
                with st.spinner("Reading your description…"):
                    try:
                        matched = extract_symptoms_with_openai(description, api_key)
                        st.session_state.ai_matched = matched
                        st.session_state.ai_error = "" if matched else "no_match"
                    except Exception as e:
                        st.session_state.ai_matched = []
                        st.session_state.ai_error = str(e)

    with col_r:
        if st.session_state.ai_error == "no_match":
            st.warning(
                "No recognisable symptoms were found. "
                "Try describing in more detail or use the manual tab."
            )
        elif st.session_state.ai_error:
            st.error(f"Something went wrong: {st.session_state.ai_error}")
        elif st.session_state.ai_matched:
            matched = st.session_state.ai_matched
            st.markdown(
                "**Symptoms identified:** " +
                "".join(f'<span class="pill">{s}</span>' for s in matched),
                unsafe_allow_html=True,
            )
            st.divider()
            render_results(matched)
        else:
            st.markdown("""
            <div style="text-align:center;padding:4rem 1rem;color:#9ca3af;">
                <div style="font-size:3rem;">✍️</div>
                <div style="font-size:1rem;margin-top:0.6rem;">
                    Write how you feel and click <b>Analyse & predict</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
