"""
Streamlit dashboard — AI Sign Language Translator.

Layout (two columns):
  Left  → mirrored live webcam feed with hand skeleton overlay
  Right → detected sign, confidence, hold-timer bar, sentence builder,
          Speak / Backspace / Clear controls, Auto-Speak checkbox
"""

import cv2
import streamlit as st

from src.config import CAMERA_INDEX
from src.landmarks import HandLandmarkExtractor
from src.predictor import SignPredictor
from src.tts_engine import TTSEngine

# ---------------------------------------------------------------------------
# Page config — MUST be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Sign Language Translator",
    page_icon="🤟",
    layout="wide",
)

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
def get_predictor() -> SignPredictor:
    """Create (and cache) the sklearn sign predictor."""
    return SignPredictor()


@st.cache_resource
def get_tts() -> TTSEngine:
    """Create (and cache) the pyttsx3 TTS engine (spawns one daemon thread)."""
    return TTSEngine()


# ---------------------------------------------------------------------------
# Session-state bootstrap (runs once per browser session)
# ---------------------------------------------------------------------------
if "sentence" not in st.session_state:
    st.session_state.sentence: list[str] = []

if "auto_speak" not in st.session_state:
    st.session_state.auto_speak: bool = True

if "camera_on" not in st.session_state:
    st.session_state.camera_on: bool = False

# ---------------------------------------------------------------------------
# Lazy-load singletons
# ---------------------------------------------------------------------------
extractor = get_extractor()

try:
    predictor = get_predictor()
    _predictor_ready = True
except FileNotFoundError as e:
    _predictor_ready = False
    _predictor_error = str(e)

tts = get_tts()

# ---------------------------------------------------------------------------
# UI — Header
# ---------------------------------------------------------------------------
st.title("🤟 AI Sign Language Translator")
st.caption(
    "Perform a sign gesture in front of your webcam. "
    "Hold it steady for **1.5 s** to commit the word to your sentence."
)

if not _predictor_ready:
    st.error(
        f"⚠️ Model not found — please train first:\n\n"
        f"```\npython -m src.train_model\n```\n\n{_predictor_error}"
    )
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Two-column layout
# ---------------------------------------------------------------------------
col_feed, col_right = st.columns([3, 2], gap="large")

# ── Left column ─────────────────────────────────────────────────────────────
with col_feed:
    st.subheader("📷 Live Feed")

    camera_toggle = st.toggle(
        "Start / Stop Camera",
        value=st.session_state.camera_on,
        key="camera_toggle",
    )
    st.session_state.camera_on = camera_toggle

    feed_placeholder = st.empty()
    status_placeholder = st.empty()

    if not camera_toggle:
        feed_placeholder.info("▶ Toggle the camera on to begin signing.")

# ── Right column ─────────────────────────────────────────────────────────────
with col_right:
    st.subheader("📝 Translation")

    # Sign label & confidence
    sign_placeholder       = st.empty()
    confidence_placeholder = st.empty()
    progress_placeholder   = st.empty()

    # Seed default state text
    sign_placeholder.markdown("**Detected Sign:** —")
    confidence_placeholder.markdown("**Confidence:** —")
    progress_placeholder.progress(0.0, text="Hold timer")

    st.divider()

    # Sentence builder display
    st.markdown("#### 💬 Sentence Builder")
    sentence_placeholder = st.empty()
    sentence_placeholder.info(
        " ".join(st.session_state.sentence) or "_No words yet…_"
    )

    st.divider()

    # Action buttons (row)
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        if st.button("🔊 Speak Sentence", width="stretch"):
            sentence_text = " ".join(st.session_state.sentence)
            if sentence_text:
                tts.speak(sentence_text)

    with btn_col2:
        if st.button("⌫ Backspace", width="stretch"):
            if st.session_state.sentence:
                st.session_state.sentence.pop()
                predictor.reset()
                st.rerun()

    with btn_col3:
        if st.button("🗑️ Clear", width="stretch"):
            st.session_state.sentence.clear()
            predictor.reset()
            tts.clear_queue()
            st.rerun()

    st.divider()

    # Auto-speak checkbox (synced to session state)
    auto_speak = st.checkbox(
        "🔈 Auto-Speak on Commit",
        value=st.session_state.auto_speak,
    )
    st.session_state.auto_speak = auto_speak

# ---------------------------------------------------------------------------
# Camera loop — only runs while toggle is ON
# ---------------------------------------------------------------------------
if st.session_state.camera_on:
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        status_placeholder.error(
            f"❌ Could not open camera at index {CAMERA_INDEX}. "
            "Check that your webcam is connected and not in use by another app."
        )
    else:
        status_placeholder.success("🟢 Camera active")

        try:
            while st.session_state.camera_on:
                ret, frame = cap.read()
                if not ret:
                    status_placeholder.warning("⚠️ Frame capture failed — retrying…")
                    continue

                # 1. Mirror horizontally (selfie view)
                frame = cv2.flip(frame, 1)

                # 2. Extract landmarks and annotated skeleton
                features, annotated = extractor.extract(frame)

                # 3. Run sign prediction with hold-timer debouncing
                result = predictor.predict(features)
                label         = result["label"]
                confidence    = result["confidence"]
                hold_progress = result["hold_progress"]
                committed_word = result["committed_word"]

                # 4. If a word was committed this frame → update sentence
                if committed_word:
                    st.session_state.sentence.append(committed_word)
                    sentence_placeholder.info(
                        " ".join(st.session_state.sentence)
                    )
                    if st.session_state.auto_speak:
                        tts.speak(committed_word)

                # 5. Update right-column placeholders
                if label:
                    sign_placeholder.markdown(f"**Detected Sign:** `{label}`")
                    confidence_placeholder.markdown(
                        f"**Confidence:** {confidence * 100:.1f}%"
                    )
                else:
                    sign_placeholder.markdown("**Detected Sign:** —")
                    confidence_placeholder.markdown("**Confidence:** —")

                progress_placeholder.progress(
                    hold_progress,
                    text=f"Hold timer — {hold_progress * 100:.0f}%",
                )

                # 6. Convert BGR → RGB and stream to Streamlit
                rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                feed_placeholder.image(rgb_frame, width="stretch")

        finally:
            cap.release()
            status_placeholder.info("⏹ Camera stopped.")
            feed_placeholder.info("▶ Toggle the camera on to begin signing.")
