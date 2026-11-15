
"""Train a noise-augmented WavLM deepfake classifier."""

from pathlib import Path

import joblib
import numpy as np
import torch

from transformers import WavLMModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
)

from src.preprocessing import load_audio
from src.features import extract_features
from src.augmentation import add_gaussian_noise


FEATURES_PATH = Path("experiments/wavlm_features_grouped.npz")
AUGMENTED_PATH = Path("experiments/wavlm_augmented_train.npz")
MODEL_PATH = Path("models/wavlm_logreg_augmented.joblib")

MODEL_NAME = "microsoft/wavlm-base"
SNR_LEVELS = [20, 10, 5]
SEED = 42


def extract_augmented_features(filepaths, labels):
    """Extract noisy training embeddings, using a local cache."""

    if AUGMENTED_PATH.exists():
        print("Loading cached augmented features...")
        with np.load(AUGMENTED_PATH) as cache:
            assert np.array_equal(
                cache["filepaths"], filepaths
            ), "Augmented cache contains different recordings"

            assert np.array_equal(
                cache["labels"], labels
            ), "Augmented cache labels do not match"

            return cache["X"]

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Loading WavLM on {device}...")

    model = WavLMModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    rng = np.random.default_rng(SEED)
    snr_assignments = np.resize(
        np.array(SNR_LEVELS),
        len(filepaths)
    )
    rng.shuffle(snr_assignments)

    embeddings = []

    for index, filepath in enumerate(filepaths):
        audio, _ = load_audio(filepath)

        noise_rng = np.random.default_rng(SEED + index)

        noisy_audio = add_gaussian_noise(
            audio,
            int(snr_assignments[index]),
            noise_rng
        )

        features = extract_features(
            noisy_audio,
            model,
            device
        )

        embeddings.append(features.reshape(-1))

        if (index + 1) % 25 == 0:
            print(
                f"Extracted {index + 1}/{len(filepaths)} "
                "augmented recordings"
            )

    X_augmented = np.asarray(
        embeddings, dtype=np.float32
    )

    assert np.isfinite(X_augmented).all()

    AUGMENTED_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    np.savez_compressed(
        AUGMENTED_PATH,
        X=X_augmented,
        labels=labels,
        filepaths=filepaths,
        snr_db=snr_assignments
    )

    print(f"Augmented features saved: {AUGMENTED_PATH}")

    return X_augmented


def main():
    with np.load(FEATURES_PATH) as data:
        X = data["X"]
        y = data["y"]
        splits = data["splits"]
        filepaths = data["filepaths"]

    train_mask = splits == "train"
    validation_mask = splits == "validation"

    X_train = X[train_mask]
    y_train = y[train_mask]
    train_files = filepaths[train_mask]

    X_val = X[validation_mask]
    y_val = y[validation_mask]

    print(f"Clean training recordings: {len(y_train)}")
    print(f"Validation recordings: {len(y_val)}")

    # Augment training recordings only
    X_noisy = extract_augmented_features(
        train_files, y_train
    )

    # Combine clean and augmented embeddings
    X_combined = np.vstack([
        X_train,
        X_noisy
    ])

    y_combined = np.concatenate([
        y_train,
        y_train
    ])

    assert np.isfinite(X_combined).all()

    print(f"Total training examples: {len(y_combined)}")

    # Same classifier configuration as clean baseline
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=SEED
        )
    )

    classifier.fit(X_combined, y_combined)

    # Evaluate using clean validation recordings
    predictions = classifier.predict(X_val)
    scores = classifier.predict_proba(X_val)[:, 1]

    print("\n--- Augmented Model: Clean Validation ---")
    print(f"Accuracy: {accuracy_score(y_val, predictions):.4f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y_val, predictions):.4f}"
    )
    print(f"F1: {f1_score(y_val, predictions):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, scores):.4f}")

    MODEL_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    joblib.dump(classifier, MODEL_PATH)

    print(f"\nAugmented model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
