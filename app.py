"""
Streamlit dashboard — Sign Language Translator (Dual-Mode).

Modes
-----
  Common Words   — 20 gesture classes, space-separated sentence assembly,
                   auto-speak on commit supported.
  Fingerspelling — 26 A–Z classes, letters concatenated without spaces,
                   auto-speak suppressed (speak on demand / Space button).

Layout (two columns)
---------------------
  Left  → mirrored live webcam feed with hand skeleton overlay
  Right → mode selector, detected sign, hold-timer bar,
           transcript box, action buttons
"""

import cv2
import streamlit as st
from pathlib import Path

from src.config import CAMERA_INDEX, MODELS_DIR
from src.landmarks import HandLandmarkExtractor
from src.predictor import SignPredictor
from src.tts_engine import TTSEngine

# ---------------------------------------------------------------------------
# Alphabet-specific artifact paths (mirrors src/train_alphabet.py)
# ---------------------------------------------------------------------------
_ALPHA_CLASSIFIER = MODELS_DIR / "alphabet_classifier.joblib"
_ALPHA_ENCODER    = MODELS_DIR / "alphabet_label_encoder.joblib"

# ---------------------------------------------------------------------------
# Page config — MUST be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sign Language Translator",
    page_icon=None,
    layout="wide",
)

# ---------------------------------------------------------------------------
# Global CSS — professional dark-slate theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Hide Streamlit chrome ───────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Root / body ─────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0F172A;
    color: #F8FAFC;
    font-family: "Inter", "Segoe UI", system-ui, sans-serif;
}

/* ── App outer container ─────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] > .main {
    background-color: #0F172A;
    padding-top: 0;
}

/* ── Block container width ───────────────────────────────────────────────── */
.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* ── App header ──────────────────────────────────────────────────────────── */
.app-category {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #38BDF8;
    margin-bottom: 0.35rem;
}
.app-title {
    font-size: 2.25rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #F8FAFC;
    margin-top: 0;
    margin-bottom: 0.35rem;
    line-height: 1.2;
}
.app-subtitle {
    font-size: 1rem;
    color: #94A3B8;
    margin-top: 0;
    margin-bottom: 1.5rem;
    letter-spacing: 0.01em;
    line-height: 1.5;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid #1E293B;
    margin: 1rem 0;
}

/* ── Section headings ────────────────────────────────────────────────────── */
.section-heading {
    font-size: 1.05rem;
    font-weight: 600;
    color: #E2E8F0;
    margin-bottom: 0.5rem;
}

/* ── Camera feed card ────────────────────────────────────────────────────── */
.feed-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem;
    min-height: 440px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #475569;
    font-size: 0.875rem;
}

/* ── Transcript box ──────────────────────────────────────────────────────── */
.transcript-box {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    min-height: 100px;
    font-size: 1.5rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    color: #F8FAFC;
    word-break: break-word;
    line-height: 1.6;
}
.transcript-empty {
    color: #334155;
    font-size: 1rem;
    font-weight: 400;
    font-style: italic;
}

/* ── Detected sign badge ─────────────────────────────────────────────────── */
.sign-badge {
    display: inline-block;
    background: #0F172A;
    border: 1px solid #3B82F6;
    color: #3B82F6;
    border-radius: 6px;
    padding: 0.2rem 0.75rem;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    font-family: "JetBrains Mono", "Fira Code", monospace;
}
.sign-idle {
    color: #334155;
    border-color: #334155;
}

/* ── Muted metadata text ─────────────────────────────────────────────────── */
.meta-text {
    font-size: 0.75rem;
    color: #475569;
    margin-top: 0.25rem;
}

/* ── Streamlit progress bar ──────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div > div {
    background-color: #3B82F6 !important;
}
[data-testid="stProgress"] {
    background-color: #1E293B;
}

/* ── Streamlit toggle ────────────────────────────────────────────────────── */
[data-testid="stToggle"] span { background-color: #334155; }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    background-color: #1E293B;
    color: #CBD5E1;
    border: 1px solid #334155;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.5rem 0.75rem;
    transition: background 0.15s, border-color 0.15s, color 0.15s;
}
[data-testid="stButton"] > button:hover {
    background-color: #263348;
    border-color: #3B82F6;
    color: #F8FAFC;
}
[data-testid="stButton"] > button:active {
    background-color: #1D4ED8;
    border-color: #1D4ED8;
    color: #fff;
}

/* ── Radio mode selector ─────────────────────────────────────────────────── */
[data-testid="stRadio"] label {
    color: #94A3B8 !important;
    font-size: 0.85rem;
}
[data-testid="stRadio"] [aria-checked="true"] + div {
    color: #F8FAFC !important;
}

/* ── Checkbox ────────────────────────────────────────────────────────────── */
[data-testid="stCheckbox"] label {
    color: #94A3B8;
    font-size: 0.8rem;
}

/* ── Error / warning banners ─────────────────────────────────────────────── */
[data-testid="stAlert"] {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    color: #94A3B8;
}

/* ── st.image rounded ────────────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid #334155;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached singleton resources
# @st.cache_resource ensures a single instance per process regardless of how
# many times Streamlit reruns the script.
# ---------------------------------------------------------------------------

@st.cache_resource
def get_extractor() -> HandLandmarkExtractor:
    """Create (and cache) the MediaPipe hand-landmark extractor."""
    return HandLandmarkExtractor()


@st.cache_resource
def get_words_predictor() -> SignPredictor:
    """Create (and cache) the Common Words sign predictor."""
    return SignPredictor()


@st.cache_resource
def get_alpha_predictor() -> SignPredictor:
    """Create (and cache) the Fingerspelling (A–Z) predictor."""
    return SignPredictor(
        classifier_path=_ALPHA_CLASSIFIER,
        encoder_path=_ALPHA_ENCODER,
    )


@st.cache_resource
def get_tts() -> TTSEngine:
    """Create (and cache) the pyttsx3 TTS engine (spawns one daemon thread)."""
    return TTSEngine()


# ---------------------------------------------------------------------------
# Session-state bootstrap (runs once per browser session)
# ---------------------------------------------------------------------------
if "sentence"   not in st.session_state: st.session_state.sentence:   list[str] = []
if "auto_speak" not in st.session_state: st.session_state.auto_speak: bool      = True
if "camera_on"  not in st.session_state: st.session_state.camera_on:  bool      = False
if "mode"       not in st.session_state: st.session_state.mode:       str       = "Common Words"

# ---------------------------------------------------------------------------
# Lazy-load singletons — handle missing model files gracefully
# ---------------------------------------------------------------------------
extractor = get_extractor()

_words_ready, _words_error = True, ""
_alpha_ready, _alpha_error = True, ""

try:
    words_predictor = get_words_predictor()
except FileNotFoundError as e:
    _words_ready = False; _words_error = str(e); words_predictor = None  # type: ignore

try:
    alpha_predictor = get_alpha_predictor()
except FileNotFoundError as e:
    _alpha_ready = False; _alpha_error = str(e); alpha_predictor = None  # type: ignore

tts = get_tts()

# ---------------------------------------------------------------------------
# UI — Typographic header
# ---------------------------------------------------------------------------
st.markdown('<p class="app-category">REAL-TIME ACCESSIBILITY SYSTEM</p>', unsafe_allow_html=True)
st.markdown('<h1 class="app-title">Sign Language Translator</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">'
    'Hold a sign gesture steady for 1.5 seconds to commit it to your transcript.'
    '</p>',
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Two-column layout
# ---------------------------------------------------------------------------
col_feed, col_right = st.columns([3, 2], gap="large")

# ── Left column — Camera feed ─────────────────────────────────────────────
with col_feed:
    st.markdown('<p class="section-heading">Live Camera</p>', unsafe_allow_html=True)

    camera_toggle = st.toggle(
        "Camera",
        value=st.session_state.camera_on,
        key="camera_toggle",
    )
    st.session_state.camera_on = camera_toggle

    feed_placeholder   = st.empty()
    status_placeholder = st.empty()

    if not camera_toggle:
        feed_placeholder.markdown(
            '<div class="feed-card">Camera is off — toggle to begin.</div>',
            unsafe_allow_html=True,
        )

# ── Right column — Controls & Transcript ─────────────────────────────────
with col_right:

    # ── Mode selector ────────────────────────────────────────────────────
    st.markdown('<p class="section-heading">Recognition Mode</p>', unsafe_allow_html=True)
    new_mode = st.radio(
        "Recognition Mode",
        options=["Common Words", "Fingerspelling (A–Z)"],
        horizontal=True,
        label_visibility="collapsed",
        index=0 if st.session_state.mode == "Common Words" else 1,
    )
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.session_state.sentence.clear()
        if words_predictor: words_predictor.reset()
        if alpha_predictor: alpha_predictor.reset()

    is_spell_mode = (st.session_state.mode == "Fingerspelling (A–Z)")

    # Model availability warnings
    if is_spell_mode and not _alpha_ready:
        st.error(f"Alphabet model not found.\n\n```\npython -m src.train_alphabet\n```\n\n{_alpha_error}")
    elif not is_spell_mode and not _words_ready:
        st.error(f"Words model not found.\n\n```\npython -m src.train_model\n```\n\n{_words_error}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Detected sign + hold progress ────────────────────────────────────
    st.markdown('<p class="section-heading">Detected Sign</p>', unsafe_allow_html=True)
    sign_placeholder       = st.empty()
    confidence_placeholder = st.empty()
    progress_placeholder   = st.empty()

    sign_placeholder.markdown(
        '<span class="sign-badge sign-idle">—</span>', unsafe_allow_html=True
    )
    progress_placeholder.progress(0.0, text="")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Transcript ────────────────────────────────────────────────────────
    st.markdown('<p class="section-heading">Transcript</p>', unsafe_allow_html=True)
    if is_spell_mode:
        display_text = "".join(st.session_state.sentence)
    else:
        display_text = " ".join(st.session_state.sentence)

    sentence_placeholder = st.empty()
    if display_text:
        sentence_placeholder.markdown(
            f'<div class="transcript-box">{display_text}</div>',
            unsafe_allow_html=True,
        )
    else:
        sentence_placeholder.markdown(
            '<div class="transcript-box">'
            '<span class="transcript-empty">No input yet</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Action buttons ────────────────────────────────────────────────────
    if is_spell_mode:
        bc1, bc2, bc3, bc4 = st.columns(4)

        with bc1:
            if st.button("Speak", width="stretch"):
                text = "".join(st.session_state.sentence)
                if text: tts.speak(text)

        with bc2:
            if st.button("Space", width="stretch"):
                st.session_state.sentence.append(" ")
                tts.speak("".join(st.session_state.sentence))
                if alpha_predictor: alpha_predictor.reset()
                st.rerun()

        with bc3:
            if st.button("Back", width="stretch"):
                if st.session_state.sentence:
                    st.session_state.sentence.pop()
                    if alpha_predictor: alpha_predictor.reset()
                    st.rerun()

        with bc4:
            if st.button("Clear", width="stretch"):
                st.session_state.sentence.clear()
                tts.clear_queue()
                if alpha_predictor: alpha_predictor.reset()
                st.rerun()

    else:
        bw1, bw2, bw3 = st.columns(3)

        with bw1:
            if st.button("Speak", width="stretch"):
                sentence_text = " ".join(st.session_state.sentence)
                if sentence_text: tts.speak(sentence_text)

        with bw2:
            if st.button("Backspace", width="stretch"):
                if st.session_state.sentence:
                    st.session_state.sentence.pop()
                    if words_predictor: words_predictor.reset()
                    st.rerun()

        with bw3:
            if st.button("Clear", width="stretch"):
                st.session_state.sentence.clear()
                tts.clear_queue()
                if words_predictor: words_predictor.reset()
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Auto-speak checkbox ───────────────────────────────────────────────
    auto_speak = st.checkbox(
        "Auto-speak on commit  (Words mode only)",
        value=st.session_state.auto_speak,
        disabled=is_spell_mode,
    )
    st.session_state.auto_speak = auto_speak

# ---------------------------------------------------------------------------
# Camera loop — only runs while toggle is ON
# ---------------------------------------------------------------------------
if st.session_state.camera_on:

    if is_spell_mode:
        active_predictor = alpha_predictor
        mode_ready       = _alpha_ready
    else:
        active_predictor = words_predictor
        mode_ready       = _words_ready

    if not mode_ready or active_predictor is None:
        status_placeholder.error("Model unavailable — please train the model first.")
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)

        if not cap.isOpened():
            status_placeholder.error(
                f"Could not open camera at index {CAMERA_INDEX}. "
                "Check that your webcam is connected and not in use."
            )
        else:
            status_placeholder.markdown(
                '<p class="meta-text">Camera active</p>', unsafe_allow_html=True
            )

            try:
                while st.session_state.camera_on:
                    ret, frame = cap.read()
                    if not ret:
                        status_placeholder.warning("Frame capture failed — retrying.")
                        continue

                    # 1. Mirror horizontally (selfie view)
                    frame = cv2.flip(frame, 1)

                    # 2. Extract landmarks and annotated skeleton
                    features, annotated = extractor.extract(frame)

                    # 3. Run sign prediction with hold-timer debouncing
                    result         = active_predictor.predict(features)
                    label          = result["label"]
                    confidence     = result["confidence"]
                    hold_progress  = result["hold_progress"]
                    committed_word = result["committed_word"]

                    # 4. Handle committed output based on active mode
                    if committed_word:
                        if is_spell_mode:
                            st.session_state.sentence.append(committed_word)
                            sentence_placeholder.markdown(
                                f'<div class="transcript-box">'
                                f'{"".join(st.session_state.sentence)}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            # No auto-speak in spell mode
                        else:
                            st.session_state.sentence.append(committed_word)
                            sentence_placeholder.markdown(
                                f'<div class="transcript-box">'
                                f'{" ".join(st.session_state.sentence)}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            if st.session_state.auto_speak:
                                tts.speak(committed_word)

                    # 5. Update sign badge + muted confidence metadata
                    if label:
                        sign_placeholder.markdown(
                            f'<span class="sign-badge">{label}</span>',
                            unsafe_allow_html=True,
                        )
                        confidence_placeholder.markdown(
                            f'<p class="meta-text">'
                            f'{st.session_state.mode} &nbsp;·&nbsp; '
                            f'{confidence * 100:.1f}% confidence'
                            f'</p>',
                            unsafe_allow_html=True,
                        )
                    else:
                        sign_placeholder.markdown(
                            '<span class="sign-badge sign-idle">—</span>',
                            unsafe_allow_html=True,
                        )
                        confidence_placeholder.markdown(
                            '<p class="meta-text">No sign detected</p>',
                            unsafe_allow_html=True,
                        )

                    progress_placeholder.progress(
                        hold_progress,
                        text=f"Hold  {hold_progress * 100:.0f}%",
                    )

                    # 6. Convert BGR → RGB and stream to Streamlit
                    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    feed_placeholder.image(rgb_frame, width="stretch")

            finally:
                cap.release()
                status_placeholder.markdown(
                    '<p class="meta-text">Camera stopped.</p>',
                    unsafe_allow_html=True,
                )
                feed_placeholder.markdown(
                    '<div class="feed-card">Camera is off — toggle to begin.</div>',
                    unsafe_allow_html=True,
                )
