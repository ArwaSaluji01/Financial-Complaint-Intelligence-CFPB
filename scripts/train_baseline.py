import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline import build_baseline_model


CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_split(task_dir, split_name):
    file_path = task_dir / f"{split_name}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {file_path}")

    df = pd.read_csv(file_path)

    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{file_path} must contain columns: {required_columns}"
        )

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    return df


def train_and_evaluate(task_name, config):
    print("\n" + "=" * 65)
    print(f"TRAINING TF-IDF + LINEAR SVM: {task_name.upper()}")
    print("=" * 65)

    task_dir = PROJECT_ROOT / "data" / "processed" / task_name

    train_df = load_split(task_dir, "train")
    validation_df = load_split(task_dir, "validation")

    X_train = train_df["text"]
    y_train = train_df["label"]

    X_validation = validation_df["text"]
    y_validation = validation_df["label"]

    train_classes = set(y_train.unique())
    validation_classes = set(y_validation.unique())

    if train_classes != validation_classes:
        raise ValueError(
            f"{task_name}: training and validation class labels differ."
        )

    print(f"Training samples: {len(train_df):,}")
    print(f"Validation samples: {len(validation_df):,}")
    print(f"Number of classes: {len(train_classes)}")

    model = build_baseline_model(config["baseline"])

    print("\nFitting model...")
    model.fit(X_train, y_train)

    print("Generating validation predictions...")
    predictions = model.predict(X_validation)

    accuracy = accuracy_score(y_validation, predictions)
    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_validation,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_validation,
        predictions,
        labels=sorted(train_classes),
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "task": task_name,
        "model": "TF-IDF + LinearSVC",
        "training_samples": len(train_df),
        "validation_samples": len(validation_df),
        "number_of_classes": len(train_classes),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    # Save model
    model_dir = PROJECT_ROOT / "artifacts" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{task_name}_tfidf_svm.joblib"
    joblib.dump(model, model_path)

    # Save evaluation reports
    report_dir = PROJECT_ROOT / "reports" / "metrics" / "baseline"
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = report_dir / f"{task_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    report_path = report_dir / f"{task_name}_classification_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # Save confusion matrix
    class_labels = sorted(train_classes)

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=class_labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=class_labels,
        columns=class_labels,
    )

    matrix_path = report_dir / f"{task_name}_confusion_matrix.csv"
    matrix_df.to_csv(matrix_path)

    print("\nValidation results:")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print(f"\nModel saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print(f"Classification report saved: {report_path}")
    print(f"Confusion matrix saved: {matrix_path}")

    return metrics


def main():
    config = load_config()

    results = []

    for task_name in ["product", "issue"]:
        results.append(
            train_and_evaluate(task_name, config)
        )

    comparison = pd.DataFrame(results)

    comparison_path = (
        PROJECT_ROOT
        / "reports"
        / "metrics"
        / "baseline"
        / "baseline_comparison.csv"
    )

    comparison.to_csv(comparison_path, index=False)

    print("\n" + "=" * 65)
    print("BASELINE TRAINING COMPLETED")
    print("=" * 65)
    print(comparison.to_string(index=False))
    print(f"\nComparison saved: {comparison_path}")


if __name__ == "__main__":
    main()