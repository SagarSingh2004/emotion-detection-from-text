import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch
import numpy as np
import re
import os
import zipfile
import gdown


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS — Dark Glassmorphism Theme
# =========================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --bg-base:       #080c14;
        --bg-surface:    #0d1220;
        --bg-elevated:   #111827;
        --bg-glass:      rgba(17, 24, 39, 0.7);
        --border:        rgba(99, 102, 241, 0.15);
        --border-active: rgba(99, 102, 241, 0.5);
        --accent:        #6366f1;
        --accent-light:  #818cf8;
        --accent-glow:   rgba(99, 102, 241, 0.25);
        --text-primary:  #f1f5f9;
        --text-muted:    #94a3b8;
        --success:       #10b981;
        --warning:       #f59e0b;
        --danger:        #ef4444;
        --joy:           #f59e0b;
        --sadness:       #60a5fa;
        --anger:         #ef4444;
        --fear:          #a78bfa;
        --love:          #f472b6;
        --surprise:      #34d399;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg-base);
        color: var(--text-primary);
    }

    /* ---- ANIMATED BACKGROUND MESH ---- */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(ellipse 80% 50% at 20% 10%, rgba(99,102,241,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 80%, rgba(244,114,182,0.05) 0%, transparent 60%),
            radial-gradient(ellipse 50% 60% at 60% 30%, rgba(52,211,153,0.04) 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
    }

    /* ---- SIDEBAR ---- */
    section[data-testid="stSidebar"] {
        background: var(--bg-surface);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: var(--text-muted) !important;
        font-size: 14px;
        transition: color 0.2s;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        color: var(--text-primary) !important;
    }

    /* ---- HEADER HERO ---- */
    .hero-wrap {
        position: relative;
        padding: 2.5rem 2.8rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f1b35 0%, #0d1220 60%, #130d2a 100%);
        border: 1px solid var(--border-active);
        box-shadow: 0 0 60px var(--accent-glow), 0 20px 60px rgba(0,0,0,0.5);
        margin-bottom: 2rem;
        overflow: hidden;
    }

    .hero-wrap::after {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 280px; height: 280px;
        background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: var(--text-primary);
        margin: 0 0 0.4rem 0;
    }

    .hero-title span {
        background: linear-gradient(90deg, var(--accent-light), var(--love));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-sub {
        font-size: 0.95rem;
        color: var(--text-muted);
        margin: 0;
        font-weight: 400;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16,185,129,0.12);
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 100px;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 600;
        color: var(--success);
        margin-top: 1rem;
        letter-spacing: 0.5px;
    }

    .hero-badge::before {
        content: '';
        width: 7px; height: 7px;
        background: var(--success);
        border-radius: 50%;
        animation: pulse-dot 1.8s ease-in-out infinite;
    }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%       { opacity: 0.4; transform: scale(0.7); }
    }

    /* ---- KPI CARDS ---- */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }

    .kpi-card {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        position: relative;
        overflow: hidden;
        transition: border-color 0.25s, transform 0.25s;
    }

    .kpi-card:hover {
        border-color: var(--border-active);
        transform: translateY(-3px);
    }

    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent), var(--accent-light));
    }

    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }

    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'DM Mono', monospace;
    }

    .kpi-sub {
        font-size: 11px;
        color: var(--text-muted);
        margin-top: 4px;
    }

    /* ---- TEXTAREA ---- */
    .stTextArea textarea {
        background: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        transition: border-color 0.2s !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }

    /* ---- BUTTON ---- */
    .stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 12px;
        border: none;
        background: linear-gradient(135deg, var(--accent) 0%, #7c3aed 100%);
        color: white;
        font-family: 'DM Sans', sans-serif;
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.3px;
        cursor: pointer;
        transition: opacity 0.2s, transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 4px 20px rgba(99,102,241,0.35);
    }

    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(99,102,241,0.45);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* ---- TABS ---- */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-elevated);
        border-radius: 12px;
        padding: 4px;
        gap: 2px;
        border: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-muted) !important;
        border-radius: 9px !important;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 14px;
        border: none !important;
        transition: all 0.2s !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--accent) !important;
        color: white !important;
    }

    /* ---- RESULT CARD ---- */
    .result-card {
        background: var(--bg-glass);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border-active);
        border-radius: 20px;
        padding: 2rem 2.4rem;
        margin-top: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .result-card::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(99,102,241,0.05) 0%, transparent 60%);
        pointer-events: none;
    }

    .emotion-display {
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -1px;
        line-height: 1;
        margin: 0.3rem 0 0.6rem 0;
    }

    .confidence-bar-wrap {
        background: rgba(255,255,255,0.06);
        border-radius: 100px;
        height: 8px;
        margin-top: 0.5rem;
        overflow: hidden;
    }

    .confidence-bar-fill {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, var(--accent), var(--accent-light));
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .confidence-pct {
        font-family: 'DM Mono', monospace;
        font-size: 1.6rem;
        font-weight: 500;
        color: var(--accent-light);
    }

    .result-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 4px;
    }

    /* ---- EMOTION PILL ---- */
    .emotion-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border-radius: 100px;
        padding: 6px 16px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-top: 8px;
    }

    .pill-joy      { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
    .pill-sadness  { background: rgba(96,165,250,0.15); color: #60a5fa; border: 1px solid rgba(96,165,250,0.3); }
    .pill-anger    { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
    .pill-fear     { background: rgba(167,139,250,0.15);color: #a78bfa; border: 1px solid rgba(167,139,250,0.3); }
    .pill-love     { background: rgba(244,114,182,0.15);color: #f472b6; border: 1px solid rgba(244,114,182,0.3); }
    .pill-surprise { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }

    /* ---- HISTORY CHIP ---- */
    .history-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 0.5rem;
    }

    .history-chip {
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: 100px;
        padding: 4px 14px;
        font-size: 12px;
        color: var(--text-muted);
    }

    /* ---- SECTION LABEL ---- */
    .section-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: var(--accent-light);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border);
    }

    /* ---- ABOUT CARD ---- */
    .about-card {
        background: var(--bg-glass);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-bottom: 1rem;
    }

    .about-card h4 {
        font-size: 14px;
        font-weight: 600;
        color: var(--accent-light);
        margin-bottom: 0.6rem;
    }

    .about-card p, .about-card li {
        font-size: 14px;
        color: var(--text-muted);
        line-height: 1.7;
    }

    /* ---- SPINNER / ALERTS ---- */
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
        background: var(--bg-elevated) !important;
    }

    /* ---- GENERAL ---- */
    h1, h2, h3, h4, h5, h6 { color: var(--text-primary) !important; }
    p, label, span { color: var(--text-primary); }

    .stMarkdown p { color: var(--text-muted); }

    div[data-testid="stVerticalBlock"] > div:has(.stPlotlyChart) {
        background: var(--bg-elevated);
        border-radius: 16px;
        border: 1px solid var(--border);
        padding: 1rem;
    }

    footer { display: none; }

    .footer-custom {
        text-align: center;
        padding: 2.5rem 0 1.5rem;
        color: var(--text-muted);
        font-size: 13px;
        border-top: 1px solid var(--border);
        margin-top: 3rem;
    }

    .footer-custom span {
        color: var(--accent-light);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# EMOTION META
# =========================================================

EMOTION_LABELS = {
    0: "joy",
    1: "sadness",
    2: "anger",
    3: "fear",
    4: "love",
    5: "surprise"
}

EMOTION_EMOJI = {
    "joy":      "😄",
    "sadness":  "😢",
    "anger":    "😡",
    "fear":     "😨",
    "love":     "❤️",
    "surprise": "😲"
}

EMOTION_COLOR = {
    "joy":      "#f59e0b",
    "sadness":  "#60a5fa",
    "anger":    "#ef4444",
    "fear":     "#a78bfa",
    "love":     "#f472b6",
    "surprise": "#34d399"
}


# =========================================================
# MODEL LOADING
# =========================================================

MODEL_DIR = "emotion_model"
ZIP_FILE  = "emotion_model.zip"
FILE_ID   = "18j3RU1sw-2c0Ia6AF3WG3-VdKWw8g_FX"


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_DIR):
        with st.spinner("⬇️  Downloading model weights from Google Drive..."):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, ZIP_FILE, quiet=False)
        with st.spinner("📦  Extracting model..."):
            with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
                zip_ref.extractall(".")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"❌  Model loading failed: {e}")
    st.stop()


# =========================================================
# PREPROCESSING
# =========================================================

def handle_negations(text):
    """
    Replace negated emotion phrases with their semantic opposite
    BEFORE the model sees them, so the dominant keyword is correct.
    """
    # ── Joy negations → Sadness ──────────────────────────────
    text = re.sub(r"\bi(?:'m| am) not (?:happy|joyful|glad|excited|pleased)\b",
                  "i feel sad", text)
    text = re.sub(r"\bi (?:don't|do not|dont) feel (?:happy|joyful|glad|excited|pleased)\b",
                  "i feel sad", text)
    text = re.sub(r"\bi (?:don't|do not|dont) (?:feel good|feel great|feel wonderful)\b",
                  "i feel sad", text)

    # ── Sadness negations → Joy ──────────────────────────────
    text = re.sub(r"\bi(?:'m| am) not (?:sad|unhappy|depressed|miserable|down|upset)\b",
                  "i feel happy", text)
    text = re.sub(r"\bi (?:don't|do not|dont) feel (?:sad|unhappy|depressed|miserable|down|upset)\b",
                  "i feel happy", text)
    text = re.sub(r"\bi(?:'m| am) not (?:crying|crying anymore|grieving)\b",
                  "i feel happy", text)

    # ── Anger negations → Joy ────────────────────────────────
    text = re.sub(r"\bi(?:'m| am) not (?:angry|mad|furious|enraged|irritated|annoyed)\b",
                  "i feel calm and happy", text)
    text = re.sub(r"\bi (?:don't|do not|dont) feel (?:angry|mad|furious|irritated|annoyed)\b",
                  "i feel calm and happy", text)

    # ── Fear negations → Joy/Confidence ─────────────────────
    text = re.sub(r"\bi(?:'m| am) not (?:afraid|scared|fearful|terrified|anxious|worried)\b",
                  "i feel brave and confident", text)
    text = re.sub(r"\bi (?:don't|do not|dont) feel (?:afraid|scared|fearful|terrified|anxious|worried)\b",
                  "i feel brave and confident", text)

    # ── Love negations → Sadness/Anger ───────────────────────
    text = re.sub(r"\bi(?:'m| am) not in love\b",
                  "i feel sad and heartbroken", text)
    text = re.sub(r"\bi (?:don't|do not|dont) (?:love|feel love|care)\b",
                  "i feel cold and indifferent", text)

    # ── Surprise negations → Calm ────────────────────────────
    text = re.sub(r"\bi(?:'m| am) not (?:surprised|shocked|amazed|astonished)\b",
                  "i feel calm and unsurprised", text)
    text = re.sub(r"\bi (?:don't|do not|dont) feel (?:surprised|shocked|amazed)\b",
                  "i feel calm and unsurprised", text)

    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    # Handle negations BEFORE removing punctuation/special chars
    text = handle_negations(text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.strip()


# =========================================================
# PREDICTION
# =========================================================

def predict_emotion(text):
    cleaned = clean_text(text)
    inputs  = tokenizer(
        cleaned,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.nn.functional.softmax(outputs.logits, dim=1)

    probs_np        = probs.cpu().numpy()[0]
    predicted_class = int(np.argmax(probs_np))
    predicted_label = EMOTION_LABELS[predicted_class]
    confidence      = float(np.max(probs_np))

    return {
        "label":         predicted_label,
        "confidence":    confidence,
        "probabilities": probs_np
    }


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding:1.2rem 0 1rem 0;">
            <div style="font-size:1.6rem;font-weight:700;color:#f1f5f9;letter-spacing:-0.5px;">
                🧠 EmotionAI
            </div>
            <div style="font-size:12px;color:#64748b;margin-top:4px;font-weight:500;">
                Transformer-Powered NLP
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊  Dashboard", "ℹ️  About"],
        label_visibility="collapsed"
    )

    st.markdown("---")

# =========================================================
# HERO SECTION
# =========================================================

st.markdown(
    """
    <div class="hero-wrap">
        <p class="hero-title">Emotion Detection <span>AI Dashboard</span></p>
        <p class="hero-sub">
            Fine-tuned DistilBERT · 6 emotion classes · 93.5% test accuracy
        </p>
        <div class="hero-badge">LIVE &nbsp;·&nbsp; Production Ready</div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# KPI CARDS
# =========================================================

st.markdown(
    """
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Architecture</div>
            <div class="kpi-value">DistilBERT</div>
            <div class="kpi-sub">distilbert-base-uncased</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Emotion Classes</div>
            <div class="kpi-value">6</div>
            <div class="kpi-sub">Joy · Sadness · Anger · Fear · Love · Surprise</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Test Accuracy</div>
            <div class="kpi-value">93.35%</div>
            <div class="kpi-sub">On held-out test set (2 000 samples)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">F1-Score</div>
            <div class="kpi-value">0.9332</div>
            <div class="kpi-sub">Weighted macro F1 at epoch 3</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PAGES
# =========================================================

# ─── DASHBOARD PAGE ───────────────────────────────────────
if page == "📊  Dashboard":

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown(
            '<div class="section-label">Text Input</div>',
            unsafe_allow_html=True
        )

        # If an example was clicked, push it into the widget's session state key
        if "prefill" in st.session_state:
            st.session_state["input_box"] = st.session_state.pop("prefill")

        user_input = st.text_area(
            label="input_text",
            height=200,
            key="input_box",
            placeholder="Type or paste any text here — a sentence, tweet, review, diary entry...",
            label_visibility="collapsed"
        )

        predict_btn = st.button("⚡  Analyze Emotion", use_container_width=True)

        # Example prompts
        st.markdown(
            "<div style='margin-top:1rem;font-size:11px;color:#475569;"
            "font-weight:600;letter-spacing:1px;text-transform:uppercase;"
            "margin-bottom:8px;'>Try an example</div>",
            unsafe_allow_html=True
        )

        examples = [
            "I'm so happy today, everything feels perfect!",
            "I lost my job and I don't know what to do.",
            "I can't believe they did that — I'm furious!",
            "I was shocked and amazed by the surprise party.",
        ]

        for ex in examples:
            label = (f'"{ex[:48]}..."' if len(ex) > 48 else f'"{ex}"')
            if st.button(
                label,
                use_container_width=True,
                key=f"ex_{ex[:20]}"
            ):
                st.session_state["prefill"] = ex
                st.rerun()

    with col_result:
        st.markdown(
            '<div class="section-label">Prediction Result</div>',
            unsafe_allow_html=True
        )

        if predict_btn:
            if not user_input or user_input.strip() == "":
                st.warning("⚠️  Please enter some text before analyzing.")
            else:
                with st.spinner("Running transformer inference…"):
                    result = predict_emotion(user_input)

                st.session_state.history.append(result["label"])
                st.session_state.total_analyzed += 1

                em    = result["label"]
                conf  = result["confidence"]
                emoji = EMOTION_EMOJI[em]
                color = EMOTION_COLOR[em]
                conf_pct = f"{conf:.1%}"
                conf_w   = f"{conf * 100:.1f}%"

                result_html = (
                    "<div class='result-card'>"
                    "<div class='result-label'>Detected Emotion</div>"
                    f"<div class='emotion-display' style='color:{color};'>"
                    f"{emoji}&nbsp;{em.capitalize()}</div>"
                    f"<div class='emotion-pill pill-{em}'>{emoji} {em.capitalize()}</div>"
                    "<div style='margin-top:1.4rem;'>"
                    "<div style='display:flex;justify-content:space-between;"
                    "align-items:baseline;margin-bottom:6px;'>"
                    "<div class='result-label'>Confidence Score</div>"
                    f"<div class='confidence-pct'>{conf_pct}</div>"
                    "</div>"
                    "<div class='confidence-bar-wrap'>"
                    f"<div class='confidence-bar-fill' style='width:{conf_w};"
                    f"background:linear-gradient(90deg,{color},{color}aa);'>"
                    "</div></div></div></div>"
                )
                st.markdown(result_html, unsafe_allow_html=True)

                # Probability bar chart
                st.markdown("<br>", unsafe_allow_html=True)
                labels     = list(EMOTION_LABELS.values())
                colors     = [EMOTION_COLOR[l] for l in labels]
                probs_list = [float(p) for p in result["probabilities"]]

                fig = go.Figure(go.Bar(
                    x=labels,
                    y=probs_list,
                    marker=dict(
                        color=colors,
                        opacity=0.85,
                        line=dict(width=0)
                    ),
                    text=[f"{p:.1%}" for p in probs_list],
                    textposition="outside",
                    textfont=dict(size=11, color="#94a3b8")
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans", color="#94a3b8"),
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=260,
                    title=dict(
                        text="Probability Distribution",
                        font=dict(size=13, color="#64748b"),
                        x=0
                    ),
                    xaxis=dict(
                        showgrid=False,
                        tickfont=dict(size=12, color="#94a3b8"),
                        linecolor="rgba(255,255,255,0.05)"
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.04)",
                        tickformat=".0%",
                        tickfont=dict(size=11),
                        range=[0, 1.1]
                    ),
                    bargap=0.35
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown(
                """
                <div style="
                    height:320px;
                    display:flex;
                    flex-direction:column;
                    align-items:center;
                    justify-content:center;
                    border:1px dashed rgba(99,102,241,0.25);
                    border-radius:16px;
                    color:#334155;
                    text-align:center;
                    gap:12px;
                ">
                    <div style="font-size:2.5rem;">🧠</div>
                    <div style="font-size:14px;font-weight:500;">
                        Enter text on the left and click<br>
                        <span style="color:#6366f1;">Analyze Emotion</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ─── ABOUT PAGE ──────────────────────────────────────────
elif page == "ℹ️  About":

    st.markdown(
        '<div class="section-label">Project Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="about-card">
            <h4>📌 What this project does</h4>
            <p>
                This dashboard performs real-time emotion classification on free-form text
                using a fine-tuned <strong>DistilBERT</strong> transformer. It detects one of
                six emotions — joy, sadness, anger, fear, love, and surprise — and returns
                a confidence-calibrated probability distribution.
            </p>
        </div>

        <div class="about-card">
            <h4>🗂️ Dataset</h4>
            <p>
                <strong>Emotions Dataset for NLP</strong> (Kaggle — praveengovi) · CC-BY-SA-4.0<br>
                Train: 16,000 samples &nbsp;|&nbsp; Val: 2,000 &nbsp;|&nbsp; Test: 2,000<br>
                After deduplication: 15,938 training samples.
            </p>
        </div>

        <div class="about-card">
            <h4>🏋️ Training Details</h4>
            <ul>
                <li>Base model: <code>distilbert-base-uncased</code></li>
                <li>Fine-tuned with Hugging Face <code>Trainer</code> API</li>
                <li>3 epochs &nbsp;·&nbsp; batch size 16 &nbsp;·&nbsp; Adam optimizer</li>
                <li>Best validation accuracy: <strong>93.95%</strong> (epoch 1)</li>
                <li>Test accuracy: <strong>93.35%</strong></li>
            </ul>
        </div>

        <div class="about-card">
            <h4>🛠️ Tech Stack</h4>
            <p>
                Python · PyTorch · Hugging Face Transformers · Streamlit · Plotly · Pandas
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer-custom">
        Built with ❤️ using <span>Streamlit</span>,
        <span>Hugging Face Transformers</span> &amp;
        <span>Plotly</span>
    </div>
    """,
    unsafe_allow_html=True
)