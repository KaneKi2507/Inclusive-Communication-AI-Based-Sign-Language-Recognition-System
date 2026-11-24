"""
SignPredictor — Live inference with hold-timer debouncing.

Architecture contract (§ D):
  predict(features) is called once per frame by the app layer and handles:
    • model.predict_proba  → label + confidence
    • CONFIDENCE_THRESHOLD gating
    • Hold-timer: sign must be held for HOLD_DURATION_SEC before committing
    • One-commit guard: same sign committed only once per continuous hold
    • Label change → candidate/timer/last-committed state reset

Return dict keys (every call):
    label           : str | None — decoded class label (gated by confidence)
    confidence      : float      — max probability from predict_proba, 0.0..1.0
    hold_progress   : float      — 0.0..1.0 fill of the HOLD_DURATION_SEC timer
    committed_word  : str | None — word committed this frame (triggers TTS), else None
"""

import time

import joblib
import numpy as np

from src.config import (
    CLASSIFIER_PATH,
    CONFIDENCE_THRESHOLD,
    HOLD_DURATION_SEC,
    LABEL_ENCODER_PATH,
)

# Shape constants derived from the feature contract
_FEATURE_DIM = 126
_INPUT_SHAPE = (1, _FEATURE_DIM)


class SignPredictor:
    """
    Loads a trained sklearn classifier and label encoder, then performs
    per-frame sign prediction with hold-timer debouncing.

    Usage
    -----
    predictor = SignPredictor()
    result = predictor.predict(features_or_none)
    predictor.reset()
    """

    def __init__(
        self,
        classifier_path=CLASSIFIER_PATH,
        encoder_path=LABEL_ENCODER_PATH,
    ) -> None:
        # -----------------------------------------------------------------------
        # Load model artifacts — both must exist before we can run inference.
        # -----------------------------------------------------------------------
        classifier_path = type(CLASSIFIER_PATH)(classifier_path)
        encoder_path    = type(LABEL_ENCODER_PATH)(encoder_path)

        if not classifier_path.exists():
            raise FileNotFoundError(
                f"Classifier not found at '{classifier_path}'.\n"
                "Please train the model first by running:\n"
                "    python -m src.train_model"
            )
        if not encoder_path.exists():
            raise FileNotFoundError(
                f"Label encoder not found at '{encoder_path}'.\n"
                "Please train the model first by running:\n"
                "    python -m src.train_model"
            )

        self._model   = joblib.load(classifier_path)
        self._encoder = joblib.load(encoder_path)

        print(
            f"[predictor] Loaded classifier from: {classifier_path}\n"
            f"[predictor] Loaded label encoder from: {encoder_path}\n"
            f"[predictor] Classes: {list(self._encoder.classes_)}"
        )

        # Initialise debounce / hold-timer state
        self._candidate_label: str | None = None
        self._candidate_start: float = 0.0
        self._last_committed_label: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray | None) -> dict:
        """
        Run one inference step and return the prediction state dict.

        Parameters
        ----------
        features : np.ndarray of shape (126,) and dtype float32, or None
            Landmark feature vector from HandLandmarkExtractor.extract().
            Pass None when no hand is detected in the current frame.

        Returns
        -------
        dict with keys:
            label          : str | None
            confidence     : float  (0.0 – 1.0)
            hold_progress  : float  (0.0 – 1.0)
            committed_word : str | None
        """
        # --- Case 1: No hand detected ---
        if features is None:
            self.reset()
            return {
                "label": None,
                "confidence": 0.0,
                "hold_progress": 0.0,
                "committed_word": None,
            }

        # --- Run inference ---
        X = features.reshape(_INPUT_SHAPE)          # (1, 126)
        proba = self._model.predict_proba(X)[0]     # (n_classes,)
        top_idx = int(np.argmax(proba))
        confidence = float(proba[top_idx])
        label = str(self._encoder.inverse_transform([top_idx])[0])

        # --- Case 2: Confidence below threshold ---
        if confidence < CONFIDENCE_THRESHOLD:
            self._reset_timer()
            return {
                "label": label,
                "confidence": confidence,
                "hold_progress": 0.0,
                "committed_word": None,
            }

        # --- Case 3: Confident prediction — run hold-timer logic ---
        now = time.monotonic()
        committed_word: str | None = None

        if label == self._candidate_label:
            # Same sign is still being held — advance the timer
            elapsed = now - self._candidate_start
            hold_progress = min(elapsed / HOLD_DURATION_SEC, 1.0)

            if elapsed >= HOLD_DURATION_SEC and label != self._last_committed_label:
                # Timer expired and this sign hasn't been committed yet in
                # this continuous hold — commit it exactly once.
                committed_word = label
                self._last_committed_label = label
        else:
            # Sign changed — reset candidate and timer, clear committed guard
            self._candidate_label = label
            self._candidate_start = now
            self._last_committed_label = None
            hold_progress = 0.0

        return {
            "label": label,
            "confidence": confidence,
            "hold_progress": hold_progress,
            "committed_word": committed_word,
        }

    def reset(self) -> None:
        """
        Clear all internal timer and cooldown state.

        Call when the app switches signs, clears the sentence buffer,
        or when no hand is detected.
        """
        self._candidate_label = None
        self._candidate_start = 0.0
        self._last_committed_label = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset_timer(self) -> None:
        """Reset only the hold-timer without touching the committed-label guard."""
        self._candidate_label = None
        self._candidate_start = 0.0
