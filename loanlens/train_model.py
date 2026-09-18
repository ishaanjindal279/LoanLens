#!/usr/bin/env python3
"""
Train the TF-IDF + LinearSVC scam detection model.

Uses the scam_detection_input.csv with 12 columns:
message,otp_request,upfront_fee,guaranteed_approval,urgency_pressure,
excessive_permissions,fake_rbi_claim,hidden_charges,suspicious_link,flag,risk_score,risk_category

Also compares LinearSVC vs LogisticRegression vs MultinomialNB.

Saves:
- model.pkl: Trained LinearSVC pipeline (primary)
- tfidf.pkl: Fitted TfidfVectorizer
- metrics.json: Evaluation metrics + model comparison
"""

import csv
import json
import re
import sys
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─── Configuration ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
INPUT_CSV = BASE_DIR / "scam_detection_input.csv"
MODEL_OUTPUT = BASE_DIR / "model.pkl"
Tfidf_OUTPUT = BASE_DIR / "tfidf.pkl"
METRICS_OUTPUT = BASE_DIR / "metrics.json"

# ─── Required column names ───────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "message", "otp_request", "upfront_fee", "guaranteed_approval",
    "urgency_pressure", "excessive_permissions", "fake_rbi_claim",
    "hidden_charges", "suspicious_link", "flag", "risk_score", "risk_category"
]


def normalize_text(text: str) -> str:
    """Conservative text normalization that preserves fraud signals."""
    if not text:
        return ""
    s = text.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.,;:!\?\"'()]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_and_prepare_data(csv_path: Path) -> tuple:
    """Load CSV data, validate, and return messages and labels."""
    messages = []
    labels = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        if header != REQUIRED_COLUMNS:
            col_map = {name: idx for idx, name in enumerate(header)}
            required_set = set(REQUIRED_COLUMNS)
            if not required_set.issubset(col_map.keys()):
                raise ValueError(f"Missing required columns. Got: {header}")
            rows = []
            for row_data in reader:
                if len(row_data) < len(REQUIRED_COLUMNS):
                    continue
                mapped = [row_data[col_map[col]] for col in REQUIRED_COLUMNS]
                rows.append(mapped)
        else:
            rows = list(reader)

    for row in rows:
        if len(row) < len(REQUIRED_COLUMNS):
            continue

        message = row[0].strip()
        if not message:
            continue

        try:
            label = int(row[REQUIRED_COLUMNS.index("flag")])
        except (ValueError, IndexError):
            continue

        messages.append(normalize_text(message))
        labels.append(label)

    print(f"Loaded {len(messages)} messages with labels")
    print(f"Class distribution: {np.bincount(labels)}")
    return messages, labels


def build_pipeline(clf_name: str = "LinearSVC") -> Pipeline:
    """Build a TF-IDF + classifier pipeline.
    
    Supported classifiers: LinearSVC, LogisticRegression, MultinomialNB
    """
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )

    if clf_name == "LogisticRegression":
        clf = LogisticRegression(max_iter=2000, random_state=42)
    elif clf_name == "MultinomialNB":
        clf = MultinomialNB(alpha=1.0)
    else:  # Default: LinearSVC
        clf = LinearSVC(max_iter=2000, dual="auto", tol=1e-3)

    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def evaluate_model(pipeline, X_test, y_test, model_name: str) -> dict:
    """Evaluate a trained pipeline and return metrics dict."""
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Extract metrics for suspicious class (class=1)
    suspicious_key = "1"
    precision = report[suspicious_key]["precision"] if suspicious_key in report else 0
    recall = report[suspicious_key]["recall"] if suspicious_key in report else 0
    f1 = report[suspicious_key]["f1-score"] if suspicious_key in report else 0

    return {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
    }


def train_and_evaluate():
    """Train the model and evaluate with proper metrics.
    
    Compares LinearSVC, LogisticRegression, and MultinomialNB.
    Saves LinearSVC as the primary model.
    """
    messages, labels = load_and_prepare_data(INPUT_CSV)

    if len(messages) < 10:
        print("ERROR: Not enough data for training")
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        messages, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    print(f"Train set: {len(X_train)} messages, {np.bincount(y_train)} class distribution")
    print(f"Test set: {len(X_test)} messages, {np.bincount(y_test)} class distribution")

    # ── Compare multiple models ────────────────────────────────────────
    models_to_compare = ["LinearSVC", "LogisticRegression", "MultinomialNB"]
    comparison_results = []

    for clf_name in models_to_compare:
        print(f"\n{'-' * 40}")
        print(f"Training: {clf_name}")
        try:
            pipeline = build_pipeline(clf_name)
            pipeline.fit(X_train, y_train)
            metrics = evaluate_model(pipeline, X_test, y_test, clf_name)
            comparison_results.append(metrics)

            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-score:  {metrics['f1_score']:.4f}")
            print(f"  Confusion: {metrics['confusion_matrix']}")

            # Save the primary model (LinearSVC)
            if clf_name == "LinearSVC":
                primary_pipeline = pipeline
                primary_metrics = metrics
        except Exception as e:
            print(f"  ERROR training {clf_name}: {e}")

    # ── Print comparison table ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'-' * 62}")
    for m in comparison_results:
        marker = " <- PRIMARY" if m["model_name"] == "LinearSVC" else ""
        print(f"{m['model_name']:<22} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f}{marker}")
    print(f"{'-' * 62}")

    # ── Full classification report for primary model ───────────────────
    print(f"\n=== Full Classification Report (LinearSVC — Primary) ===")
    y_pred_primary = primary_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred_primary, zero_division=0))

    # ── Save primary model and artifacts ───────────────────────────────
    joblib.dump(primary_pipeline, MODEL_OUTPUT)
    print(f"Primary model saved to: {MODEL_OUTPUT}")

    tfidf_vectorizer = primary_pipeline.named_steps["tfidf"]
    joblib.dump(tfidf_vectorizer, Tfidf_OUTPUT)
    print(f"TF-IDF vectorizer saved to: {Tfidf_OUTPUT}")

    # Save metrics with comparison
    full_metrics = {
        **primary_metrics,
        "model_name": "TfidfVectorizer + LinearSVC",
        "test_samples": int(len(X_test)),
        "train_samples": int(len(X_train)),
        "features": int(primary_pipeline.named_steps["tfidf"].max_features),
        "model_comparison": comparison_results,
    }
    with open(METRICS_OUTPUT, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"Metrics saved to: {METRICS_OUTPUT}")

    return primary_pipeline, full_metrics, tfidf_vectorizer


if __name__ == "__main__":
    train_and_evaluate()
