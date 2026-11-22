# Phase 1 Scope Breakdown (40% Deliverable)

## Completed in Phase 1

### 1. Computer Vision & Feature Engineering (`src/landmarks.py`)
- **2-Hand Tracking:** Integrates the modern MediaPipe Tasks API (`HandLandmarker`) with auto-download for the `hand_landmarker.task` model.
- **Scale & Position Invariance:** Extracts 21 `(x, y, z)` landmarks per hand, translates them relative to the wrist, and normalizes them into a fixed 126-feature vector (63 per hand, zero-padded if only one hand is visible).
- **Visual Overlay:** Draws the live hand skeleton onto the OpenCV webcam feed so reviewers can visually see the tracking working in real time.

### 2. Custom Dataset Pipeline (`src/collect_data.py` & `data/landmarks.csv`)
- **Interactive Data Collector:** A lightweight OpenCV script that lets you press a key to record 150–200 landmark frames per sign directly from your laptop webcam.
- **Structured Dataset:** Saves all 126 normalized coordinates + class label into `data/landmarks.csv` for 10–15 high-impact static signs (e.g., *Hello, Help, Yes, No, Stop, Water, Thanks*, plus a few alphabets).

### 3. Model Training & Report Artifacts (`src/train_model.py` & `reports/`)
- **Dual-Model Benchmarking:** Trains both an RBF Support Vector Classifier (SVC) and a Random Forest Classifier on an 80/20 stratified train/test split.
- **Automated Selection:** Picks the higher-F1 model and saves it alongside the `LabelEncoder` (`models/sign_classifier.joblib` and `models/label_encoder.joblib`).
- **Report-Ready Outputs:** Automatically generates three files for the *Results and Discussion* section of the Phase-I report:
  - `reports/confusion_matrix.png` (high-resolution heatmap)
  - `reports/model_comparison.csv` (Accuracy & F1 table comparing SVC vs. Random Forest)
  - `reports/classification_report.txt` (per-class Precision, Recall, and F1-Score)

### 4. Real-Time Inference & Sentence Builder (`src/predictor.py`)
- **Confidence Gating:** Ignores uncertain predictions below a `0.75` probability threshold to prevent random false positives when your hand is moving into position.
- **Stability Hold Timer:** Requires a sign to be held steadily for `1.5` seconds before committing it to the sentence, eliminating flickering duplicates.
- **Sentence Buffer:** Accumulates committed words into a readable sentence string with support for clearing or deleting the last word.

### 5. Non-Blocking Offline Audio (`src/tts_engine.py`)
- **Threaded Speech Engine:** Uses `pyttsx3` inside a background daemon thread with a `queue.Queue` so your laptop speaks newly committed words (or the full sentence) aloud without freezing the webcam feed.

### 6. Live Demo Interface (`app.py`)
- **Streamlit Dashboard:** Runs 100% locally and offline, displaying:
  - Live annotated webcam feed
  - Currently detected sign + real-time confidence progress bar
  - Hold-timer progress bar showing when the word will lock in
  - Built sentence display with **Speak Sentence**, **Backspace**, and **Clear** buttons

---

## Explicitly Out of Scope for Phase 1 (For Viva Clarity)
Keep this boundary clear to avoid scope creep and to answer reviewer questions cleanly:
- **Dynamic / Moving Signs (LSTM):** Deferred to Phase 2 (Jan 2027).
- **NLP Grammar Reconstruction:** Deferred to Phase 2 (Jan 2027).
- **FastAPI Backend & SQLite History:** Deferred to Phase 2 / 3.
- **KivyMD Cross-Platform App:** Deferred to Phase 3 (Feb–Mar 2027).