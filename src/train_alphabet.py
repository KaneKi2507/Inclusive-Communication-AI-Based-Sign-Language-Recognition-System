"""
Alphabet Classifier Training Pipeline.

Mirrors the logic in src/train_model.py but operates on the ASL alphabet
dataset (data/alphabet_landmarks.csv, 26 classes A–Z) and writes all
artifacts to alphabet-specific paths so they never collide with the
sign-language gesture classifier.

Outputs
-------
models/alphabet_classifier.joblib
models/alphabet_label_encoder.joblib
reports/alphabet_confusion_matrix.png   (16×14 in, 300 DPI)
reports/alphabet_model_comparison.csv
reports/alphabet_classification_report.txt

Usage
-----
    python -m src.train_alphabet
"""

import joblib
import matplotlib
matplotlib.use("Agg")   # headless / non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR

# ---------------------------------------------------------------------------
# Alphabet-specific artifact paths
# ---------------------------------------------------------------------------
ALPHABET_CSV            = DATA_DIR    / "alphabet_landmarks.csv"
ALPHABET_CLASSIFIER     = MODELS_DIR  / "alphabet_classifier.joblib"
ALPHABET_LABEL_ENCODER  = MODELS_DIR  / "alphabet_label_encoder.joblib"
ALPHABET_CM_PNG         = REPORTS_DIR / "alphabet_confusion_matrix.png"
ALPHABET_COMPARISON_CSV = REPORTS_DIR / "alphabet_model_comparison.csv"
ALPHABET_REPORT_TXT     = REPORTS_DIR / "alphabet_classification_report.txt"


def train_and_evaluate() -> None:
    """Execute the alphabet model training, benchmarking, and reporting pipeline."""

    # -----------------------------------------------------------------------
    # 1. Dataset Loading & Validation
    # -----------------------------------------------------------------------
    if not ALPHABET_CSV.exists():
        raise FileNotFoundError(
            f"Alphabet dataset not found at {ALPHABET_CSV}.\n"
            "Please run 'python -m src.extract_alphabet' first."
        )

    df = pd.read_csv(ALPHABET_CSV)
    if df.empty:
        raise ValueError(
            f"Alphabet dataset at {ALPHABET_CSV} is empty. "
            "Re-run src.extract_alphabet to populate it."
        )
    if "label" not in df.columns:
        raise KeyError("Expected 'label' column in alphabet dataset.")

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].astype(str).values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = [str(c) for c in label_encoder.classes_]

    print(
        f"[train_alphabet] Loaded {X.shape[0]} samples across "
        f"{len(class_names)} classes: {', '.join(class_names)}"
    )

    # -----------------------------------------------------------------------
    # 2. Stratified 80/20 Train/Test Split
    # -----------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(
        f"[train_alphabet] Training samples: {X_train.shape[0]}, "
        f"Test samples: {X_test.shape[0]}"
    )

    # -----------------------------------------------------------------------
    # 3. Mirror Augmentation on Training Set Only (prevents data leakage)
    #    • Swap Left-hand slot  (cols  0..62) ↔ Right-hand slot (cols 63..125)
    #    • Negate all x-coordinates (cols 0, 3, 6, …, 123) in the mirrored copy
    #    X_test / y_test are NEVER touched — honest evaluation is preserved.
    # -----------------------------------------------------------------------
    X_mirror = X_train.copy()

    # Swap left ↔ right feature blocks (63 values each: 21 × (x, y, z))
    X_mirror[:, :63] = X_train[:, 63:]
    X_mirror[:, 63:] = X_train[:, :63]

    # Negate x-coords: every 3rd column starting at 0 → 42 indices total
    x_indices = np.arange(0, 126, 3)
    X_mirror[:, x_indices] *= -1.0

    X_train = np.concatenate([X_train, X_mirror], axis=0)
    y_train = np.concatenate([y_train, y_train],  axis=0)

    print(
        f"[train_alphabet] Mirror augmentation applied — "
        f"training set doubled to {X_train.shape[0]} samples "
        f"(test set untouched: {X_test.shape[0]} samples)."
    )

    # -----------------------------------------------------------------------
    # 4. Model Definitions
    # -----------------------------------------------------------------------
    models = {
        "SVC_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=10, probability=True, random_state=42)),
        ]),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = []
    fitted_models: dict = {}
    predictions: dict = {}

    # -----------------------------------------------------------------------
    # 5. Training & Benchmarking
    # -----------------------------------------------------------------------
    for name, model in models.items():
        print(f"[train_alphabet] Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")

        fitted_models[name] = model
        predictions[name]   = y_pred
        results.append({"model": name, "accuracy": acc, "weighted_f1": f1})

    comparison_df = pd.DataFrame(results)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(ALPHABET_COMPARISON_CSV, index=False)

    print("\n" + "=" * 54)
    print("ALPHABET MODEL COMPARISON RESULTS")
    print("=" * 54)
    print(comparison_df.to_string(index=False))
    print("=" * 54)
    print(f"[train_alphabet] Saved comparison to: {ALPHABET_COMPARISON_CSV}")

    # -----------------------------------------------------------------------
    # 6. Best Model Selection & Artifact Export
    # -----------------------------------------------------------------------
    best_result = comparison_df.sort_values(by="weighted_f1", ascending=False).iloc[0]
    best_name   = str(best_result["model"])
    best_f1     = float(best_result["weighted_f1"])
    best_model  = fitted_models[best_name]
    best_preds  = predictions[best_name]

    print(
        f"\n[train_alphabet] Selected best model: {best_name} "
        f"(Weighted F1: {best_f1:.4f})"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model,    ALPHABET_CLASSIFIER)
    joblib.dump(label_encoder, ALPHABET_LABEL_ENCODER)
    print(f"[train_alphabet] Saved classifier    to: {ALPHABET_CLASSIFIER}")
    print(f"[train_alphabet] Saved label encoder to: {ALPHABET_LABEL_ENCODER}")

    # -----------------------------------------------------------------------
    # 7. Classification Report & Confusion Matrix
    # -----------------------------------------------------------------------
    report_text = classification_report(
        y_test, best_preds, target_names=class_names, digits=4
    )
    ALPHABET_REPORT_TXT.write_text(report_text, encoding="utf-8")
    print(f"[train_alphabet] Saved classification report to: {ALPHABET_REPORT_TXT}")

    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(
        f"Alphabet Confusion Matrix — {best_name} (Weighted F1: {best_f1:.4f})",
        fontsize=14,
        pad=14,
    )
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(ALPHABET_CM_PNG, dpi=300)
    plt.close()
    print(f"[train_alphabet] Saved 300 DPI confusion matrix to: {ALPHABET_CM_PNG}")


if __name__ == "__main__":
    train_and_evaluate()
