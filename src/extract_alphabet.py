"""
Batch ASL alphabet feature extractor.

Scans data/raw_asl_alphabet/ for A–Z letter folders, randomly samples up to
MAX_SAMPLES_PER_LETTER images per letter (seed=42), runs MediaPipe
HandLandmarker in IMAGE mode on each image, and writes identical 126-D
wrist-normalised features + label to data/alphabet_landmarks.csv.

Feature format is exactly the same as data/landmarks.csv:
    columns: f0, f1, ..., f125, label
    Left-hand slot  : indices  0..62
    Right-hand slot : indices 63..125
    Missing hand    : zero-padded (skipped entirely if 0 hands detected)

Usage:
    python -m src.extract_alphabet
"""

import csv
import random
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from src.config import (
    DATA_DIR,
    FEATURE_DIM,
    HAND_LANDMARKER_PATH,
    HAND_LANDMARKER_URL,
    MODELS_DIR,
)

# ---------------------------------------------------------------------------
# Script-level constants
# ---------------------------------------------------------------------------
RAW_ALPHABET_DIR: Path = DATA_DIR / "raw_asl_alphabet"
OUTPUT_CSV: Path = DATA_DIR / "alphabet_landmarks.csv"
MAX_SAMPLES_PER_LETTER: int = 350
RANDOM_SEED: int = 42

# Letters to extract (skip 'nothing', 'space', 'del')
ALPHABET: list[str] = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

# Supported image extensions
_IMG_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Helpers — identical normalisation logic as src/landmarks.py
# ---------------------------------------------------------------------------

def _download_model_if_missing() -> None:
    """Download the MediaPipe float16 hand-landmarker model if absent."""
    if HAND_LANDMARKER_PATH.exists():
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[extract_alphabet] hand_landmarker.task not found.\n"
        f"[extract_alphabet] Downloading from:\n  {HAND_LANDMARKER_URL}\n"
        f"[extract_alphabet] Saving to: {HAND_LANDMARKER_PATH}"
    )
    urllib.request.urlretrieve(HAND_LANDMARKER_URL, HAND_LANDMARKER_PATH)
    print("[extract_alphabet] Download complete.")


def _normalise_hand(landmarks: list) -> np.ndarray:
    """
    Wrist-centre + max-abs normalisation for a single hand.
    Exactly mirrors the logic in src/landmarks.py._normalise_hand().

    Returns np.ndarray of shape (63,), dtype float32.
    """
    coords = np.array(
        [(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32
    )  # (21, 3)
    coords -= coords[0]                         # translate to wrist origin
    max_abs = np.max(np.abs(coords))
    coords /= max_abs + 1e-6                    # scale invariance
    return coords.flatten()                     # (63,)


def _extract_features(
    result,                                     # HandLandmarkerResult
) -> np.ndarray | None:
    """
    Assemble the 126-D feature vector from a HandLandmarkerResult.

    Left-hand slot  : indices  0..62
    Right-hand slot : indices 63..125
    Duplicate handedness collision → second hand placed in the free slot.
    Returns None if no hands were detected.
    """
    hand_landmarks_list = result.hand_landmarks
    handedness_list = result.handedness

    if not hand_landmarks_list:
        return None

    slot_features: dict[int, np.ndarray] = {}
    for landmarks, handedness in zip(hand_landmarks_list, handedness_list):
        label = handedness[0].category_name     # "Left" or "Right"
        slot = 0 if label == "Left" else 1
        if slot in slot_features:               # collision guard (same side detected twice)
            slot = 1 - slot
        slot_features[slot] = _normalise_hand(landmarks)

    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    for slot, vec in slot_features.items():
        features[slot * 63 : slot * 63 + 63] = vec

    return features


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def extract_alphabet() -> None:
    """Run the full alphabet batch extraction pipeline."""
    t_start = time.monotonic()

    # Validate source directory
    if not RAW_ALPHABET_DIR.exists():
        raise FileNotFoundError(
            f"Raw alphabet dataset not found at: {RAW_ALPHABET_DIR}\n"
            "Please download the ASL Alphabet dataset and place letter folders "
            "(A–Z) inside 'data/raw_asl_alphabet/'."
        )

    _download_model_if_missing()

    # Build IMAGE-mode landmarker (independent images → no timestamp needed)
    options = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(HAND_LANDMARKER_PATH)
        ),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # Prepare output CSV (write header; overwrite any existing file)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    headers = [f"f{i}" for i in range(FEATURE_DIM)] + ["label"]
    csv_file = open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(headers)

    rng = random.Random(RANDOM_SEED)
    grand_total_saved = 0

    try:
        for letter in ALPHABET:
            letter_dir = RAW_ALPHABET_DIR / letter

            if not letter_dir.is_dir():
                print(f"[{letter}] ⚠ Folder not found — skipping.")
                continue

            # Gather all image files and sample up to MAX_SAMPLES_PER_LETTER
            all_images = [
                p for p in letter_dir.iterdir()
                if p.suffix.lower() in _IMG_EXTS
            ]
            sampled = rng.sample(
                all_images,
                k=min(MAX_SAMPLES_PER_LETTER, len(all_images)),
            )

            detected = 0
            skipped = 0

            for img_path in sampled:
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    skipped += 1
                    continue

                # Convert BGR → RGB, wrap in mp.Image
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                result = landmarker.detect(mp_image)
                features = _extract_features(result)

                if features is None:
                    skipped += 1
                    continue

                row = [f"{v:.6f}" for v in features] + [letter]
                writer.writerow(row)
                detected += 1

            grand_total_saved += detected
            skip_note = f"  ({skipped} skipped — no hand / unreadable)" if skipped else ""
            print(f"[{letter}] {detected}/{len(sampled)} hands detected{skip_note}")

    finally:
        csv_file.close()
        landmarker.close()

    elapsed = time.monotonic() - t_start
    print(
        f"\n[extract_alphabet] Done — {grand_total_saved} rows saved to: {OUTPUT_CSV}"
        f"\n[extract_alphabet] Total runtime: {elapsed:.1f}s"
    )


if __name__ == "__main__":
    extract_alphabet()
