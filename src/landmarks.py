"""
HandLandmarkExtractor — Modern MediaPipe Tasks API wrapper.

Contract (architecture.md § 4-A):
  • Uses mediapipe.tasks.python.vision.HandLandmarker in VIDEO mode.
  • Auto-downloads hand_landmarker.task float16 model if missing.
  • Draws 21 landmarks + bone connections with cv2.circle / cv2.line ONLY
    (mp.solutions.hands and mp.solutions.drawing_utils are strictly forbidden).
  • Returns a (126,) float32 feature vector:
      indices  0..62  → Left  hand, wrist-normalised (x, y, z) × 21
      indices 63..125 → Right hand, wrist-normalised (x, y, z) × 21
      Zero-padded for a missing hand; None if no hands detected.
"""

import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from src.config import (
    HAND_LANDMARKER_PATH,
    HAND_LANDMARKER_URL,
    MODELS_DIR,
    NUM_HANDS,
    FEATURE_DIM,
)

# ---------------------------------------------------------------------------
# MediaPipe hand skeleton: 21-landmark bone connections
# ---------------------------------------------------------------------------
_HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (0, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky
    (5, 9), (9, 13), (13, 17),                # Palm cross-connections
]

# BGR colours per detected hand slot (index 0 = first hand, index 1 = second)
_SLOT_COLOURS: list[tuple[int, int, int]] = [
    (0, 255, 0),     # green  — first hand
    (255, 128, 0),   # orange — second hand
]


def _download_model() -> None:
    """Download the MediaPipe float16 hand-landmarker model if absent."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[landmarks] hand_landmarker.task not found.\n"
        f"[landmarks] Downloading from:\n  {HAND_LANDMARKER_URL}\n"
        f"[landmarks] Saving to: {HAND_LANDMARKER_PATH}"
    )
    urllib.request.urlretrieve(HAND_LANDMARKER_URL, HAND_LANDMARKER_PATH)
    print("[landmarks] Download complete.")


def _normalise_hand(landmarks: list) -> np.ndarray:
    """
    Wrist-centre + max-abs normalisation for a single hand.

    Parameters
    ----------
    landmarks : list of NormalizedLandmark
        21 landmarks from MediaPipe Tasks result.

    Returns
    -------
    np.ndarray, shape (63,), dtype float32
        Flattened (x, y, z) offsets, scale-normalised.
    """
    coords = np.array(
        [(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32
    )  # (21, 3)

    # Step 1 — translate relative to wrist (landmark 0)
    coords -= coords[0]

    # Step 2 — divide by max absolute value for scale invariance
    max_abs = np.max(np.abs(coords))
    coords /= max_abs + 1e-6

    return coords.flatten()  # (63,)


class HandLandmarkExtractor:
    """
    Wraps MediaPipe Tasks HandLandmarker for video-stream (VIDEO mode) inference.

    Usage
    -----
    extractor = HandLandmarkExtractor()
    features, annotated = extractor.extract(bgr_frame)
    extractor.close()
    """

    def __init__(self) -> None:
        # Ensure the model asset exists before creating the landmarker.
        if not HAND_LANDMARKER_PATH.exists():
            _download_model()

        options = HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(HAND_LANDMARKER_PATH)
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._landmarker: HandLandmarker = HandLandmarker.create_from_options(options)
        self._last_timestamp_ms: int = 0
        self._timestamp_ms: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self, frame_bgr: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """
        Run hand-landmark detection on a single BGR frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            OpenCV BGR image, shape (H, W, 3).

        Returns
        -------
        features : np.ndarray of shape (126,) and dtype float32, or None
            Wrist-normalised feature vector.
            • indices  0..62  → Left  hand (or hand-slot 0)
            • indices 63..125 → Right hand (or hand-slot 1)
            Zero-padded if only one hand is detected.
            None if no hands are detected.
        annotated_frame : np.ndarray
            Copy of frame_bgr with hand skeletons drawn via cv2.
        """
        annotated = frame_bgr.copy()

        # Convert BGR → RGB and wrap in a MediaPipe Image.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Monotonically increasing timestamp in milliseconds for VIDEO mode
        now_ms = time.monotonic_ns() // 1_000_000
        self._timestamp_ms = max(now_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = self._timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        hand_landmarks_list = result.hand_landmarks          # list[list[NormalisedLandmark]]
        handedness_list = result.handedness                  # list[list[Category]]

        if not hand_landmarks_list:
            return None, annotated

        h, w = frame_bgr.shape[:2]

        # Build slot map: "Left" → slot 0, "Right" → slot 1
        # (MediaPipe Tasks returns handedness as seen by the camera,
        #  which is mirrored for a webcam feed — matches the architecture contract.)
        slot_features: dict[int, np.ndarray] = {}

        for hand_idx, (landmarks, handedness) in enumerate(
            zip(hand_landmarks_list, handedness_list)
        ):
            # Determine slot: Left=0, Right=1
            label = handedness[0].category_name  # "Left" or "Right"
            slot = 0 if label == "Left" else 1

            # If slot is already taken (e.g. MediaPipe misclassified both hands as same side due to overlap),
            # place the second hand into the remaining empty slot (0 or 1).
            if slot in slot_features:
                slot = 1 - slot

            # --- Normalise features ---
            slot_features[slot] = _normalise_hand(landmarks)

            # --- Draw skeleton with cv2 (NO mp.solutions.drawing_utils) ---
            colour = _SLOT_COLOURS[hand_idx % len(_SLOT_COLOURS)]

            # Pixel coordinates for all 21 landmarks
            pts = [
                (int(lm.x * w), int(lm.y * h)) for lm in landmarks
            ]

            # Draw bone connections
            for start_idx, end_idx in _HAND_CONNECTIONS:
                cv2.line(annotated, pts[start_idx], pts[end_idx], colour, 2, cv2.LINE_AA)

            # Draw landmark dots
            for pt in pts:
                cv2.circle(annotated, pt, 4, colour, -1, cv2.LINE_AA)
                cv2.circle(annotated, pt, 4, (255, 255, 255), 1, cv2.LINE_AA)  # white ring

        # --- Assemble feature vector (126,) ---
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        for slot, vec in slot_features.items():
            start = slot * 63
            features[start : start + 63] = vec

        return features, annotated

    def close(self) -> None:
        """Release the underlying MediaPipe HandLandmarker resources."""
        self._landmarker.close()
