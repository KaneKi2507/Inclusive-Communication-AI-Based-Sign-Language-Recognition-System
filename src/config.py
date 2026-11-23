"""Configuration constants and paths for Sign Language Recognition System."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project Directories & Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure essential runtime directories exist
for _dir in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# File Paths
DATASET_PATH = DATA_DIR / "landmarks.csv"
HAND_LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"
CLASSIFIER_PATH = MODELS_DIR / "sign_classifier.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"

# ---------------------------------------------------------------------------
# Official MediaPipe Float16 Hand Landmarker Model URL
# ---------------------------------------------------------------------------
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# ---------------------------------------------------------------------------
# Inference & Collection Constants
# ---------------------------------------------------------------------------
NUM_HANDS = 2
FEATURE_DIM = 126
SAMPLES_PER_CLASS = 180
CONFIDENCE_THRESHOLD = 0.75
HOLD_DURATION_SEC = 1.5
CAMERA_INDEX = 0

# ---------------------------------------------------------------------------
# Sign Classes (20 visually distinct static signs)
# ---------------------------------------------------------------------------
SIGN_CLASSES = [
    "HELLO",
    "YES",
    "BAD",
    "NO",
    "OK",
    "PEACE",
    "WATER",
    "I_LOVE_YOU",
    "CALL",
    "YOU",
    "COOL",
    "WAIT",
    "LEFT",
    "SMALL",
    "EMERGENCY",
    "HELP",
    "STOP",
    "HAPPY",
    "BIG",
    "DONE",
]
