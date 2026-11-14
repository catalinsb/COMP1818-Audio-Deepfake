
"""Evaluate WavLM deepfake detection under Gaussian noise."""

import csv
from pathlib import Path

import joblib
import numpy as np
import torch

from transformers import WavLMModel
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from src.preprocessing import load_audio
from src.features import extract_features
from src.augmentation import add_gaussian_noise


FEATURES_PATH = Path("experiments/wavlm_features_grouped.npz")
MODEL_PATH = Path("models/wavlm_logreg_baseline.joblib")
RESULTS_PATH = Path("experiments/noise_robustness.csv")

WAVLM_NAME = "microsoft/wavlm-base"
SNR_LEVELS = [20, 10, 5]
SEED = 42


def calculate_eer(labels, scores):
    """Calculate equal error rate using linear interpolation."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr

    difference = fpr - fnr
    crossing = np.where(difference >= 0)[0][0]

    if crossing == 0:
        return float(fpr[0])

    previous = crossing - 1

    weight = (
        -difference[previous]
        / (difference[crossing] - difference[previous])
    )

    eer = fpr[previous] + weight * (
        fpr[crossing] - fpr[previous]
    )

    return float(eer)


def evaluate(classifier, features, labels, condition):
    """Calculate classification metrics."""
    predictions = classifier.predict(features)
    scores = classifier.predict_proba(features)[:, 1]

    return {
        "condition": condition,
        "samples": len(labels),
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            labels, predictions
        ),
        "f1": f1_score(labels, predictions),
        "roc_auc": roc_auc_score(labels, scores),
        "eer": calculate_eer(labels, scores),
    }


def plot_robustness(results):
    """Visualise classifier performance across noise levels."""
    import matplotlib.pyplot as plt

    labels = ["Clean", "20 dB", "10 dB", "5 dB"]

    balanced_accuracy = [
        result["balanced_accuracy"] * 100
        for result in results
    ]

    roc_auc = [
        result["roc_auc"] * 100
        for result in results
    ]

    eer = [
        result["eer"] * 100
        for result in results
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        labels,
        balanced_accuracy,
        marker="o",
        linewidth=2,
        label="Balanced Accuracy"
    )

    ax.plot(
        labels,
        roc_auc,
        marker="s",
        linewidth=2,
        label="ROC-AUC"
    )

    ax.plot(
        labels,
        eer,
        marker="^",
        linewidth=2,
        label="Equal Error Rate"
    )

    ax.set_title(
        "WavLM Audio Deepfake Detection: Noise Robustness",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xlabel("Audio Condition")
    ax.set_ylabel("Performance (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_path = Path("experiments/noise_robustness.png")
    fig.savefig(output_path, dpi=300)

    print(f"\nChart saved to: {output_path}")

    plt.show()
    plt.close(fig)

def main():
    # Load existing embeddings and validation partition
    with np.load(FEATURES_PATH) as data:
        X = data["X"]
        y = data["y"]
        splits = data["splits"]
        filepaths = data["filepaths"]

    validation_mask = splits == "validation"

    X_val = X[validation_mask]
    y_val = y[validation_mask]
    validation_files = filepaths[validation_mask]

    print(f"Validation recordings: {len(y_val)}")
    assert len(y_val) == 54
    assert np.isfinite(X_val).all()

    # Load the previously trained classifier
    classifier = joblib.load(MODEL_PATH)

    results = []

    # Evaluate the existing clean embeddings
    clean_result = evaluate(
        classifier, X_val, y_val, "clean"
    )
    results.append(clean_result)

    print("\nClean baseline:", clean_result)

    # Load WavLM once for all noise conditions
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"\nLoading WavLM on {device}...")

    model = WavLMModel.from_pretrained(WAVLM_NAME)
    model = model.to(device)
    model.eval()

    # Check that fresh preprocessing matches cached features
    check_audio, _ = load_audio(validation_files[0])
    check_features = extract_features(
        check_audio, model, device
    ).reshape(-1)

    difference = np.max(
        np.abs(check_features - X_val[0])
    )

    print(f"Clean feature difference: {difference:.6f}")

    if difference > 0.01:
        raise RuntimeError(
            "Fresh and cached clean features differ. "
            "Check preprocessing and WavLM configuration."
        )

    # Generate and evaluate noisy audio
    for snr_db in SNR_LEVELS:
        print(f"\nEvaluating {snr_db} dB noise...")

        noisy_features = []

        for index, filepath in enumerate(validation_files):
            audio, _ = load_audio(filepath)

            # Deterministic independent noise per recording
            rng = np.random.default_rng(
                SEED + index + snr_db * 10000
            )

            noisy_audio = add_gaussian_noise(
                audio, snr_db, rng
            )

            features = extract_features(
                noisy_audio, model, device
            )

            noisy_features.append(features.reshape(-1))

            if (index + 1) % 10 == 0:
                print(
                    f"Processed {index + 1}/{len(validation_files)}"
                )

        X_noisy = np.asarray(
            noisy_features, dtype=np.float64
        )

        assert np.isfinite(X_noisy).all()

        result = evaluate(
            classifier,
            X_noisy,
            y_val,
            f"noise_{snr_db}db"
        )

        results.append(result)
        print("Results:", result)

    # Save results
    RESULTS_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(RESULTS_PATH, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {RESULTS_PATH}")

    print("\n--- Robustness Summary ---")
    for result in results:
        print(
            f"{result['condition']:12} | "
            f"Balanced Acc: "
            f"{result['balanced_accuracy']:.4f} | "
            f"AUC: {result['roc_auc']:.4f} | "
            f"EER: {result['eer']:.4f}"
        )

    plot_robustness(results)

if __name__ == "__main__":
    main()
