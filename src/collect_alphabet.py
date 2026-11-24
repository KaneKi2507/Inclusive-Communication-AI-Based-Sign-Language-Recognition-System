"""
ASL Alphabet Dataset Collector (Webcam).

Records 60 wrist-normalised landmark frames per letter (A–Z) using the same
HandLandmarkExtractor (VIDEO mode) and 126-D feature contract used throughout
the project, then writes data/alphabet_landmarks.csv.

Controls
--------
  SPACE : Record 60 frames for the current letter
  R     : Redo — discard the current letter's collected frames and restart it
  N     : Skip to the next letter without recording
  Q/ESC : Quit and save all data collected so far

Usage
-----
    python -m src.collect_alphabet
"""

import csv
from pathlib import Path

import cv2
import numpy as np

from src.config import (
    CAMERA_INDEX,
    DATA_DIR,
    FEATURE_DIM,
)
from src.landmarks import HandLandmarkExtractor

# ---------------------------------------------------------------------------
# Script-level constants
# ---------------------------------------------------------------------------
ALPHABET: list[str] = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
FRAMES_PER_LETTER: int = 60
OUTPUT_CSV: Path = DATA_DIR / "alphabet_landmarks.csv"

# HUD colours
_COL_WHITE  = (255, 255, 255)
_COL_YELLOW = (0, 255, 255)
_COL_RED    = (0, 0, 255)
_COL_GREEN  = (0, 255, 0)
_COL_GREY   = (160, 160, 160)
_COL_DARK   = (20, 20, 20)


# ---------------------------------------------------------------------------
# HUD drawing
# ---------------------------------------------------------------------------

def _draw_hud(
    frame: np.ndarray,
    letter: str,
    letter_idx: int,
    recorded: int,
    recording: bool,
    hand_detected: bool,
) -> None:
    """Overlay status banners, progress bar, and key instructions."""
    h, w = frame.shape[:2]

    # Semi-transparent top and bottom banners
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), _COL_DARK, -1)
    cv2.rectangle(overlay, (0, h - 48), (w, h), _COL_DARK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Letter and index
    letter_text = f"Letter: {letter}  [{letter_idx + 1} / {len(ALPHABET)}]"
    cv2.putText(frame, letter_text, (15, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, _COL_YELLOW, 2, cv2.LINE_AA)

    # Recording indicator
    if recording:
        status_text, status_col = "[RECORDING]", _COL_RED
    else:
        status_text, status_col = "[READY]", _COL_GREY
    cv2.putText(frame, status_text, (w - 230, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_col, 2, cv2.LINE_AA)

    # Frame count + progress bar
    count_text = f"Frames: {recorded} / {FRAMES_PER_LETTER}"
    cv2.putText(frame, count_text, (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, _COL_WHITE, 1, cv2.LINE_AA)

    bar_x, bar_y, bar_w, bar_h = 270, 56, 240, 14
    pct = min(1.0, recorded / FRAMES_PER_LETTER)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (80, 80, 80), 1)
    fill_w = int(bar_w * pct)
    bar_col = _COL_GREEN if pct >= 1.0 else (0, 200, 255)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h), bar_col, -1)

    # No-hand warning
    if recording and not hand_detected:
        cv2.putText(frame, "NO HAND DETECTED",
                    (w // 2 - 155, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, _COL_RED, 2, cv2.LINE_AA)

    # Bottom instructions
    instructions = "SPACE: Record | R: Redo | N: Next | Q: Quit & Save"
    cv2.putText(frame, instructions, (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COL_GREY, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _write_csv(
    filepath: Path,
    all_rows: list[list],
) -> None:
    """Write all collected rows to filepath, overwriting any existing file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    headers = [f"f{i}" for i in range(FEATURE_DIM)] + ["label"]
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(all_rows)
    print(f"[collect_alphabet] Saved {len(all_rows)} rows → {filepath}")


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the interactive alphabet data-collection camera loop."""
    extractor = HandLandmarkExtractor()
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[collect_alphabet] Error: Cannot open camera at index {CAMERA_INDEX}.")
        extractor.close()
        return

    cv2.namedWindow("ASL Alphabet Collector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ASL Alphabet Collector", 960, 640)

    # all_rows accumulates (f0..f125, label) rows across all letters
    all_rows: list[list] = []

    # Per-letter state
    letter_idx: int = 0
    letter_rows: list[list] = []   # frames collected for the current letter
    recording: bool = False

    print("\n[collect_alphabet] Started.")
    print("[collect_alphabet] Controls: SPACE=Record, R=Redo, N=Next, Q=Quit\n")

    try:
        while letter_idx < len(ALPHABET):
            letter = ALPHABET[letter_idx]

            ret, frame = cap.read()
            if not ret:
                print("[collect_alphabet] Warning: Failed to read frame — retrying.")
                continue

            # Mirror for natural selfie view
            frame = cv2.flip(frame, 1)

            # Extract landmarks
            features, annotated = extractor.extract(frame)
            hand_detected = features is not None

            # --- Auto-collect frame if recording and hand visible ---
            if recording and hand_detected:
                row = [f"{v:.6f}" for v in features] + [letter]
                letter_rows.append(row)

                if len(letter_rows) >= FRAMES_PER_LETTER:
                    # Target reached — commit and auto-advance
                    recording = False
                    all_rows.extend(letter_rows)
                    print(
                        f"[collect_alphabet] [{letter}] "
                        f"{len(letter_rows)} frames collected — advancing."
                    )
                    letter_rows = []
                    letter_idx += 1
                    continue   # skip drawing, go straight to next letter

            # --- Draw HUD ---
            _draw_hud(
                annotated,
                letter=letter,
                letter_idx=letter_idx,
                recorded=len(letter_rows),
                recording=recording,
                hand_detected=hand_detected,
            )

            cv2.imshow("ASL Alphabet Collector", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):   # Q or ESC → quit
                # Commit any partially-collected frames for the current letter
                if letter_rows:
                    all_rows.extend(letter_rows)
                    print(
                        f"[collect_alphabet] [{letter}] "
                        f"Partial — {len(letter_rows)} frames committed on quit."
                    )
                break

            elif key == ord(" "):                  # SPACE → start recording
                if not recording:
                    letter_rows = []               # fresh buffer
                    recording = True
                    print(f"[collect_alphabet] [{letter}] Recording started…")

            elif key in (ord("r"), ord("R")):      # R → redo
                letter_rows = []
                recording = False
                print(f"[collect_alphabet] [{letter}] Redo — buffer cleared.")

            elif key in (ord("n"), ord("N")):      # N → next (skip)
                recording = False
                if letter_rows:
                    all_rows.extend(letter_rows)
                    print(
                        f"[collect_alphabet] [{letter}] "
                        f"Skipped with {len(letter_rows)} frames saved."
                    )
                else:
                    print(f"[collect_alphabet] [{letter}] Skipped (no frames).")
                letter_rows = []
                letter_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        print("\n[collect_alphabet] Camera released.")

    # --- Save collected data ---
    if all_rows:
        _write_csv(OUTPUT_CSV, all_rows)
        # Summary per letter
        from collections import Counter
        counts = Counter(row[-1] for row in all_rows)
        print("\n[collect_alphabet] Per-letter summary:")
        for letter in ALPHABET:
            n = counts.get(letter, 0)
            bar = "█" * (n // 2)
            print(f"  {letter}: {n:>3} frames  {bar}")
        print(f"\n[collect_alphabet] Total rows: {len(all_rows)}")
    else:
        print("[collect_alphabet] No data collected — CSV not written.")


if __name__ == "__main__":
    main()
