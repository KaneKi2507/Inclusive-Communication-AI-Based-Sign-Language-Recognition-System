"""
Sign Language Dataset Collector.

Collects 126-dimensional wrist-normalized hand landmark vectors from a live
webcam feed and appends them to data/landmarks.csv.

Controls:
    SPACE : Start / Pause sample recording for the current target sign
    N     : Switch to next sign class
    P     : Switch to previous sign class
    Q/ESC : Quit and cleanly save dataset
"""

import csv
from pathlib import Path
import cv2
import numpy as np

from src.config import (
    CAMERA_INDEX,
    DATA_DIR,
    DATASET_PATH,
    FEATURE_DIM,
    SAMPLES_PER_CLASS,
    SIGN_CLASSES,
)
from src.landmarks import HandLandmarkExtractor


def init_dataset_file(filepath: Path) -> None:
    """Ensure the dataset CSV file exists with proper column headers."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not filepath.exists() or filepath.stat().st_size == 0:
        headers = [f"f{i}" for i in range(FEATURE_DIM)] + ["label"]
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def load_existing_counts(filepath: Path) -> dict[str, int]:
    """Count existing samples per class in data/landmarks.csv."""
    counts = {sign: 0 for sign in SIGN_CLASSES}
    if not filepath.exists() or filepath.stat().st_size == 0:
        return counts

    try:
        with open(filepath, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return counts
            label_idx = (
                header.index("label") if "label" in header else (len(header) - 1)
            )
            for row in reader:
                if row and len(row) > label_idx:
                    lbl = row[label_idx].strip()
                    counts[lbl] = counts.get(lbl, 0) + 1
    except Exception as exc:
        print(f"[collect_data] Warning while loading existing counts: {exc}")

    return counts


def append_sample(filepath: Path, features: np.ndarray, label: str) -> None:
    """Append one 126-D feature vector and label to the dataset CSV."""
    row = [f"{val:.6f}" for val in features] + [label]
    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def draw_hud(
    frame: np.ndarray,
    current_sign: str,
    class_idx: int,
    sample_count: int,
    recording: bool,
    hand_detected: bool,
) -> None:
    """Render HUD elements, status banner, progress bar, and key instructions."""
    h, w = frame.shape[:2]

    # Semi-transparent overlay banners
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Top row: Sign class info and recording state
    sign_text = f"Sign: {current_sign} [{class_idx + 1}/{len(SIGN_CLASSES)}]"
    cv2.putText(
        frame, sign_text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
    )

    if recording:
        status_text = "[RECORDING]"
        status_color = (0, 0, 255)
    else:
        status_text = "[PAUSED]"
        status_color = (180, 180, 180)

    cv2.putText(
        frame, status_text, (w - 220, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA
    )

    # Second row: Sample count & progress bar
    count_text = f"Samples: {sample_count} / {SAMPLES_PER_CLASS}"
    cv2.putText(
        frame, count_text, (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA
    )

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = 280, 54, 220, 14
    pct = min(1.0, sample_count / SAMPLES_PER_CLASS) if SAMPLES_PER_CLASS > 0 else 0.0
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
    fill_w = int(bar_w * pct)
    bar_color = (0, 255, 0) if pct >= 1.0 else (0, 200, 255)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

    # Hand presence alert when recording
    if recording and not hand_detected:
        cv2.putText(
            frame,
            "NO HAND DETECTED",
            (w // 2 - 130, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # Bottom banner: Key instructions
    instructions = "SPACE: Record/Pause | N: Next | P: Prev | Q: Quit & Save"
    cv2.putText(
        frame, instructions, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA
    )


def main() -> None:
    """Run data collection camera loop."""
    init_dataset_file(DATASET_PATH)
    counts = load_existing_counts(DATASET_PATH)

    extractor = HandLandmarkExtractor()
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[collect_data] Error: Could not open camera with index {CAMERA_INDEX}.")
        extractor.close()
        return

    cv2.namedWindow("Sign Language Dataset Collector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sign Language Dataset Collector", 960, 640)

    class_idx = 0
    recording = False

    print("\n[collect_data] Dataset collection started.")
    print("[collect_data] Controls: SPACE = Record/Pause, N = Next, P = Prev, Q = Quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[collect_data] Warning: Failed to read frame from camera.")
                break

            # 1. Mirror horizontally
            frame = cv2.flip(frame, 1)

            # 2. Extract landmark features and annotated skeleton
            features, annotated = extractor.extract(frame)
            hand_detected = features is not None

            current_sign = SIGN_CLASSES[class_idx]
            current_count = counts.get(current_sign, 0)

            # 3. Record sample if recording is active and at least one hand is detected
            if recording and hand_detected:
                append_sample(DATASET_PATH, features, current_sign)
                counts[current_sign] = current_count + 1
                current_count = counts[current_sign]

                if current_count >= SAMPLES_PER_CLASS:
                    recording = False
                    print(f"[collect_data] Target of {SAMPLES_PER_CLASS} reached for {current_sign}!")

            # 4. Render HUD on annotated frame
            draw_hud(
                annotated,
                current_sign=current_sign,
                class_idx=class_idx,
                sample_count=current_count,
                recording=recording,
                hand_detected=hand_detected,
            )

            cv2.imshow("Sign Language Dataset Collector", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key == ord(" "):
                recording = not recording
            elif key in (ord("n"), ord("N")):
                recording = False
                class_idx = (class_idx + 1) % len(SIGN_CLASSES)
            elif key in (ord("p"), ord("P")):
                recording = False
                class_idx = (class_idx - 1) % len(SIGN_CLASSES)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        print("\n[collect_data] Camera released and resources closed.")
        print(f"[collect_data] Current dataset location: {DATASET_PATH}")
        print("[collect_data] Sample totals:")
        for sign in SIGN_CLASSES:
            print(f"  - {sign}: {counts.get(sign, 0)}/{SAMPLES_PER_CLASS}")


if __name__ == "__main__":
    main()
