# Inclusive Communication: AI-Based Sign Language Recognition System

A real-time, local-first computer vision and machine learning prototype that translates static hand signs into text and offline speech. Built as an **Engineering Project in Community Service (EPICS)** at **VIT Bhopal University**.

> **Current Milestone:** Phase I (40% Deliverable) — Dual-Mode Static Sign Recognition (20 Common Words + 26 A–Z Fingerspelling).

---

## Key Features

- **Dual-Mode Recognition:**
  - **Common Words Mode (20 classes):** `HELLO`, `YES`, `BAD`, `NO`, `OK`, `PEACE`, `WATER`, `I_LOVE_YOU`, `CALL`, `YOU`, `COOL`, `WAIT`, `LEFT`, `SMALL`, `EMERGENCY`, `HELP`, `STOP`, `HAPPY`, `BIG`, `DONE`.
  - **Fingerspelling Mode (26 classes):** Custom webcam-calibrated `A–Z` static alphabet for spelling out-of-vocabulary words.
- **Two-Hand Landmark Pipeline:** Uses the modern **MediaPipe Tasks `HandLandmarker` API** (up to 2 hands) to extract 21 3D landmarks per hand, normalized relative to the wrist and max-absolute scaled into a fixed **126-dimensional feature vector** (zero-padded for single-hand frames).
- **Ambidextrous Training Pipeline:** Stratified 80:20 train-test split with **post-split horizontal mirror augmentation** (swapping left/right hand blocks and negating x-coordinates on training data only to prevent data leakage).
- **Confidence Gating & Temporal Debouncing:** Requires `>= 0.75` prediction confidence held steadily for `1.5 seconds` before committing a sign, with an anti-repeat hold guard to prevent duplicate insertions.
- **Non-Blocking Offline Text-to-Speech:** Queue-based `pyttsx3` background daemon thread (Windows SAPI5) so audio playback never stalls the live camera loop.
- **Minimalist Streamlit Dashboard:** Dark-slate live UI with annotated webcam feed, mode toggle, confidence and hold-progress bars, and sentence controls (`Space`, `Backspace`, `Clear`, `Speak`).

---

## System Architecture

```text
Webcam (OpenCV)
   │
   ▼
MediaPipe Tasks HandLandmarker (up to 2 hands × 21 landmarks × 3D)
   │
   ▼
Wrist-Relative Translation + Max-Abs Scaling ──► 126-D Feature Vector
   │
   ▼
Classifier Inference (StandardScaler + SVC_RBF)
   │
   ▼
Confidence Gate (>= 0.75) ──► Stability Timer (1.5s Hold) ──► Anti-Repeat Guard
   │
   ▼
Sentence Buffer (Streamlit UI) ──► Queued Background TTS (pyttsx3)
```

---

## Phase-I Evaluation Results

Both modes were evaluated on held-out single-signer test partitions (20% stratified split, unaugmented):

### Common Words (20 Classes — 3,600 Total / 720 Test Samples)

| Model | Accuracy | Weighted F1-Score | Notes |
| :--- | :---: | :---: | :--- |
| **SVC (RBF Kernel, C=10)** | **1.0000 (100.00%)** | **1.0000** | **Selected** (`StandardScaler` + RBF) |
| Random Forest (200 trees) | 0.9986 (99.86%) | 0.9986 | 1 misclassification out of 720 |

### Fingerspelling A–Z (26 Classes — 1,560 Total / 312 Test Samples)

| Model | Accuracy | Weighted F1-Score | Notes |
| :--- | :---: | :---: | :--- |
| **SVC (RBF Kernel, C=10)** | **1.0000 (100.00%)** | **1.0000** | **Selected** (consistent with Words mode) |
| Random Forest (200 trees) | 1.0000 (100.00%) | 1.0000 | Tied at 1.0000 |

> **Note:** The `1.0000` scores reflect clean geometric separability on controlled, single-signer webcam test sets. Multi-signer generalization across varied lighting and backgrounds is scheduled for Phases II and III.

---

## Repository Structure

```text
├── app.py                              # Streamlit dashboard & live Dual-Mode UI
├── requirements.txt                    # Python dependencies
├── architecture.md                     # System architecture & data contracts
├── breakdown.md                        # Phase-I scope & deferred roadmap
├── src/
│   ├── config.py                       # Paths, class labels, thresholds, camera index
│   ├── landmarks.py                    # MediaPipe Tasks wrapper & 126-D normalizer
│   ├── collect_data.py                 # Webcam collector for 20 Common Words
│   ├── collect_alphabet.py             # Webcam collector for 26 A–Z letters
│   ├── extract_alphabet.py             # Kaggle ASL image benchmark extractor
│   ├── train_model.py                  # Common Words training & evaluation pipeline
│   ├── train_alphabet.py               # Fingerspelling A–Z training & evaluation pipeline
│   ├── predictor.py                    # Real-time inference, confidence gate & hold timer
│   └── tts_engine.py                   # Threaded offline pyttsx3 speech worker
├── data/
│   ├── landmarks.csv                   # 20-class Common Words dataset (3,600 rows)
│   └── alphabet_landmarks.csv          # 26-class Fingerspelling dataset (1,560 rows)
├── models/                             # Saved .joblib classifiers & MediaPipe asset (gitignored)
└── reports/                            # Confusion matrices, CSV benchmarks & text reports
```

---

## Quick Start (Windows 11)

### 1. Clone & Set Up Virtual Environment
```powershell
git clone https://github.com/KaneKi2507/Inclusive-Communication-AI-Based-Sign-Language-Recognition-System.git
cd Inclusive-Communication-AI-Based-Sign-Language-Recognition-System

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Train Models (Generates `.joblib` Artifacts Locally)
> Trained `.joblib` binaries are excluded from Git to keep the repository lightweight. Run the training scripts once using the included CSV datasets:
```powershell
python -m src.train_model
python -m src.train_alphabet
```

### 3. Launch the Application
```powershell
streamlit run app.py
```

*(Optional) To record or recalibrate your own webcam samples:*
```powershell
python -m src.collect_data
python -m src.collect_alphabet
```

---

## Roadmap (Phases II & III)

- [x] **Phase I:** Two-hand 126-D landmark pipeline, Dual-Mode static recognition (Words + A–Z), Streamlit UI, offline TTS.
- [ ] **Phase II:** Dynamic / motion-based sign recognition (LSTM / 1D-CNN sequence models, including true motion `J` and `Z`) and pose-anchored context.
- [ ] **Phase III:** NLP grammar reconstruction, bidirectional speech-to-sign, FastAPI backend with SQLite history, multi-signer field validation, and cross-platform deployment.

---

## Project Team (VIT Bhopal University)

- **Manya Jajodia** (`24BAI10691`)
- **Maitri Daga** (`24BAI10431`)
- **Nishtha Manish Wadaskar** (`24BCE10608`)
- **Ayush Kumar Thakur** (`24BAI10582`)
- **Aryan Srivastava** (`24MEI10118`)
- **Ritambhara Pragya Barman** (`24MEI10099`)

**Supervisor:** Dr. Debraj Gogoi
