
"""
COMP1818 - Audio Deepfake Detection
WavLM embeddings + Logistic Regression baseline.
"""

from pathlib import Path

import joblib
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Configuration
FEATURES_PATH = Path("experiments/wavlm_features_grouped.npz")
MODEL_PATH = Path("models/wavlm_logreg_baseline.joblib")
RANDOM_SEED = 42


def main():
    # 1. Load cached WavLM embeddings
    with np.load(FEATURES_PATH) as data:
        X = data["X"].astype(np.float64)
        y = data["y"]
        splits = data["splits"]

    assert np.isfinite(X).all(), "Invalid feature values detected"

    # 2. Prepare training and validation partitions
    train_mask = splits == "train"
    val_mask = splits == "validation"

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print("\n--- Dataset ---")
    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")

    # 3. Train WavLM + Logistic Regression
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=RANDOM_SEED,
        ),
    )

    classifier.fit(X_train, y_train)
    print("\nClassifier trained successfully.")

    # 4. Evaluate on validation recordings
    y_pred = classifier.predict(X_val)
    y_scores = classifier.predict_proba(X_val)[:, 1]

    print("\n--- WavLM Baseline Results ---")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy_score(y_val, y_pred):.4f}"
    )
    print(f"F1 score: {f1_score(y_val, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, y_scores):.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_val, y_pred))

    # 5. Compare against a majority-class baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_val)

    print("\n--- Majority-Class Baseline ---")
    print(f"Accuracy: {accuracy_score(y_val, dummy_pred):.4f}")
    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy_score(y_val, dummy_pred):.4f}"
    )
    print(
        f"F1 score: "
        f"{f1_score(y_val, dummy_pred, zero_division=0):.4f}"
    )

    # 6. Save the fitted pipeline
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
