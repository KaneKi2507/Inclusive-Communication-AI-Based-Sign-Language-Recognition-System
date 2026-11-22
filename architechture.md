# Project Context & Architecture Contract: AI-Based Sign Language Recognition System (Phase 1 — 40% Deliverable)

## 1. Persona & Strict AI Rules
- Act as a senior computer vision and Python systems engineer.
- Follow the exact file tree, tech stack, and data contracts below. Do NOT invent new libraries, rename keys, or change feature vector dimensions.
- **Strictly forbidden:** Do NOT use deprecated `mp.solutions.hands` syntax. Use ONLY the modern MediaPipe Tasks API (`mediapipe.tasks.python.vision.HandLandmarker`).
- Build strictly ONE file per prompt when instructed. Do not rewrite unrelated files.

## 2. Locked Tech Stack & Environment
- **OS & Runtime:** Windows 11 (Native, NOT WSL 2), Python 3.10–3.14 inside `.venv`.
- **Vision & Tracking:** `opencv-python`, `mediapipe==1.2.3` (using `hand_landmarker.task` float16 model).
- **ML & Evaluation:** `scikit-learn`, `numpy`, `pandas`, `joblib`, `matplotlib`, `seaborn`.
- **UI & Audio:** `streamlit` (local `cv2.VideoCapture(0)` loop, NOT `streamlit-webrtc`), `pyttsx3` (offline Windows SAPI5 TTS).

## 3. Repository File Tree
```text
sign-language-ai/
├── architecture.md                 # Single source of truth (this file)
├── requirements.txt                # Pinned dependencies
├── models/
│   ├── hand_landmarker.task        # Official MediaPipe float16 model (auto-downloaded if missing)
│   ├── sign_classifier.joblib      # Best trained sklearn model (RBF SVC or RandomForest)
│   └── label_encoder.joblib        # sklearn LabelEncoder for class strings
├── data/
│   └── landmarks.csv               # Collected dataset: 126 float features + 'label' column
├── reports/
│   ├── confusion_matrix.png        # High-res plot for Phase-I report
│   ├── model_comparison.csv        # SVC vs RandomForest accuracy/F1 comparison table
│   └── classification_report.txt   # Per-class Precision, Recall, F1-Score
├── src/
│   ├── __init__.py
│   ├── config.py                   # Constants, paths, sign classes, confidence & hold thresholds
│   ├── landmarks.py                # HandLandmarker wrapper + 2-hand wrist-normalized extractor
│   ├── collect_data.py             # OpenCV webcam collector (150-200 samples/sign -> CSV)
│   ├── train_model.py              # Benchmarks RBF SVC vs RandomForest, saves best + metrics
│   ├── tts_engine.py               # Thread-safe, non-blocking pyttsx3 speech queue
│   └── predictor.py                # Live inference, confidence gating, stability timer, sentence buffer
└── app.py                          # Streamlit dashboard (video feed, live sign, confidence bar, sentence, TTS controls)
```

## 4. Strict Data Contracts

### A. Landmark Feature Vector (`src/landmarks.py`)
- **Input:** OpenCV BGR frame (`np.ndarray`, shape `(H, W, 3)`).
- **Model Asset:** `models/hand_landmarker.task` (configured with `num_hands=2`).
- **Normalization (per detected hand):**
  1. Translate all 21 landmarks relative to landmark `0` (wrist): `(x - x0, y - y0, z - z0)`.
  2. Divide all 63 values by the maximum absolute value (`max_abs + 1e-6`) for scale invariance.
- **Output Vector:** 1D `np.ndarray` of `dtype=np.float32`, exact shape `(126,)`:
  - Indices `0..62`: Left hand (or Hand 0) normalized `(x, y, z) * 21`.
  - Indices `63..125`: Right hand (or Hand 1) normalized `(x, y, z) * 21`.
  - If only 1 hand is detected, pad the missing 63 values with `0.0`.
  - If 0 hands are detected, return `None`.

### B. Dataset Schema (`data/landmarks.csv`)
- **Columns:** `f0, f1, ..., f125, label` (127 columns total).
- **Target Volume:** 150–200 rows per sign class.

### C. Classifier Contract (`src/train_model.py`)
- **Train/Test Split:** 80/20 stratified (`random_state=42`).
- **Models to Train & Compare:**
  1. `SVC(kernel="rbf", C=10, probability=True)` (inside a `StandardScaler` Pipeline).
  2. `RandomForestClassifier(n_estimators=200, random_state=42)`.
- **Outputs:** Automatically save the higher-F1 model to `models/sign_classifier.joblib` and export `reports/confusion_matrix.png`, `reports/classification_report.txt`, and `reports/model_comparison.csv`.
- **Requirement:** Saved model MUST support `.predict_proba(X)` for UI confidence scoring.

### D. Predictor State Contract (`src/predictor.py`)
- `process_frame(frame_bgr) -> dict` must return:
  - `annotated_frame`: `np.ndarray` (BGR frame with hand skeleton drawn)
  - `predicted_label`: `str | None` (only if `confidence >= CONFIDENCE_THRESHOLD`, default `0.75`)
  - `confidence`: `float` (`0.0` to `1.0`)
  - `hold_progress`: `float` (`0.0` to `1.0`, fills over `HOLD_DURATION_SEC = 1.5`)
  - `sentence`: `list[str]` (committed words in order)
  - `newly_committed`: `str | None` (the word just added on this frame, triggering TTS)

### E. TTS Contract (`src/tts_engine.py`)
- Must use a background daemon thread + `queue.Queue` so calling `speak(text)` NEVER blocks the Streamlit/OpenCV frame loop.

> **Note:** `src/landmarks.py` will auto-download the official 7.5 MB `hand_landmarker.task` float16 model directly from Google Cloud Storage if it is missing.