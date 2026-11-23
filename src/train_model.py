"""
Model Training and Benchmarking Pipeline.

Benchmarks SVC (RBF kernel with StandardScaler) against RandomForestClassifier
on wrist-normalized sign language landmark features. Saves the best model,
label encoder, confusion matrix heatmap, and evaluation reports.
"""

from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
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

from src.config import (
    CLASSIFIER_PATH,
    DATASET_PATH,
    LABEL_ENCODER_PATH,
    MODELS_DIR,
    REPORTS_DIR,
)


def train_and_evaluate() -> None:
    """Execute the model training, benchmarking, and reporting pipeline."""
    # -----------------------------------------------------------------------
    # 1. Dataset Loading & Validation
    # -----------------------------------------------------------------------
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Please run 'src/collect_data.py' first to gather training samples."
        )

    df = pd.read_csv(DATASET_PATH)
    if df.empty or len(df) == 0:
        raise ValueError(
            f"Dataset at {DATASET_PATH} is empty. "
            "Please collect samples before training."
        )

    if "label" not in df.columns:
        raise KeyError("Expected 'label' column in dataset, but it was not found.")

    feature_cols = [col for col in df.columns if col != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].astype(str).values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = [str(cls) for cls in label_encoder.classes_]

    print(f"[train_model] Loaded dataset with {X.shape[0]} samples across {len(class_names)} classes.")
    print(f"[train_model] Classes: {', '.join(class_names)}")

    # -----------------------------------------------------------------------
    # 2. Stratified 80/20 Train/Test Split
    # -----------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[train_model] Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # -----------------------------------------------------------------------
    # 3. Model Definitions
    # -----------------------------------------------------------------------
    models = {
        "SVC_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=10, probability=True, random_state=42)),
        ]),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = []
    fitted_models = {}
    predictions = {}

    # -----------------------------------------------------------------------
    # 4. Training & Benchmarking
    # -----------------------------------------------------------------------
    for name, model in models.items():
        print(f"[train_model] Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        fitted_models[name] = model
        predictions[name] = y_pred
        results.append({"model": name, "accuracy": acc, "weighted_f1": f1})

    comparison_df = pd.DataFrame(results)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = REPORTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("\n" + "=" * 50)
    print("MODEL COMPARISON RESULTS")
    print("=" * 50)
    print(comparison_df.to_string(index=False))
    print("=" * 50)
    print(f"[train_model] Saved comparison to: {comparison_path}")

    # -----------------------------------------------------------------------
    # 5. Best Model Selection & Artifact Export
    # -----------------------------------------------------------------------
    best_result = comparison_df.sort_values(by="weighted_f1", ascending=False).iloc[0]
    best_name = str(best_result["model"])
    best_f1 = float(best_result["weighted_f1"])
    best_model = fitted_models[best_name]
    best_preds = predictions[best_name]

    print(f"\n[train_model] Selected best model: {best_name} (Weighted F1: {best_f1:.4f})")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, CLASSIFIER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"[train_model] Saved classifier to: {CLASSIFIER_PATH}")
    print(f"[train_model] Saved label encoder to: {LABEL_ENCODER_PATH}")

    # -----------------------------------------------------------------------
    # 6. Classification Report & Confusion Matrix
    # -----------------------------------------------------------------------
    report_text = classification_report(
        y_test, best_preds, target_names=class_names, digits=4
    )
    report_path = REPORTS_DIR / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"[train_model] Saved classification report to: {report_path}")

    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(
        f"Confusion Matrix - Best Model: {best_name} (Weighted F1: {best_f1:.4f})",
        fontsize=13,
        pad=12,
    )
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[train_model] Saved 300 DPI confusion matrix to: {cm_path}")


if __name__ == "__main__":
    train_and_evaluate()
