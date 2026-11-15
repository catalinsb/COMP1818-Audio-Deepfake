
"""Compare clean and augmented WavLM classifiers under noise."""

import csv
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
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
RESULTS_PATH = Path("experiments/model_comparison.csv")
CHART_PATH = Path("experiments/model_comparison.png")

MODELS = {
    "Baseline": Path("models/wavlm_logreg_baseline.joblib"),
    "Augmented": Path("models/wavlm_logreg_augmented.joblib"),
}

SNR_LEVELS = [20, 10, 5]
SEED = 42


def calculate_eer(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1.0 - tpr

    difference = fpr - fnr
    crossing = np.where(difference >= 0)[0][0]

    if crossing == 0:
        return float(fpr[0])

    previous = crossing - 1
    fraction = (
        -difference[previous]
        / (difference[crossing] - difference[previous])
    )

    return float(
        fpr[previous]
        + fraction * (fpr[crossing] - fpr[previous])
    )


def evaluate(classifier, X, y, model_name, condition):
    predictions = classifier.predict(X)
    scores = classifier.predict_proba(X)[:, 1]

    assert np.isfinite(scores).all()

    return {
        "model": model_name,
        "condition": condition,
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        "f1": float(f1_score(y, predictions)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "eer": calculate_eer(y, scores),
    }


def plot_results(results):
    conditions = ["clean", "20db", "10db", "5db"]
    labels = ["Clean", "20 dB", "10 dB", "5 dB"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name in MODELS:
        subset = {
            result["condition"]: result
            for result in results
            if result["model"] == model_name
        }

        values = [
            subset[condition]["balanced_accuracy"] * 100
            for condition in conditions
        ]

        ax.plot(
            labels,
            values,
            marker="o",
            linewidth=2,
            label=model_name,
        )

    ax.set_title(
        "WavLM Deepfake Detection: Noise Robustness"
    )
    ax.set_xlabel("Audio condition")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(CHART_PATH, dpi=300)
    print(f"\nChart saved: {CHART_PATH}")

    plt.show()
    plt.close(fig)


def main():
    with np.load(FEATURES_PATH) as data:
        mask = data["splits"] == "validation"
        X_clean = data["X"][mask]
        y_val = data["y"][mask]
        filepaths = data["filepaths"][mask]

    assert len(y_val) == 54
    assert np.isfinite(X_clean).all()

    classifiers = {
        name: joblib.load(path)
        for name, path in MODELS.items()
    }

    results = []

    # Clean validation results
    for name, classifier in classifiers.items():
        results.append(
            evaluate(classifier, X_clean, y_val, name, "clean")
        )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Loading WavLM on {device}...")

    wavlm = WavLMModel.from_pretrained(
        "microsoft/wavlm-base"
    ).to(device)
    wavlm.eval()

    # Ensure freshly extracted features match the cached baseline
    audio_check, _ = load_audio(filepaths[0])
    fresh = extract_features(
        audio_check, wavlm, device
    ).reshape(-1)

    difference = np.max(np.abs(fresh - X_clean[0]))
    print(f"Clean feature difference: {difference:.6f}")

    if difference > 0.01:
        raise RuntimeError("Cached WavLM features do not match.")

    for snr_db in SNR_LEVELS:
        print(f"\nProcessing {snr_db} dB...")

        embeddings = []

        for index, filepath in enumerate(filepaths):
            audio, _ = load_audio(filepath)

            rng = np.random.default_rng(
                SEED + index + snr_db * 10000
            )

            noisy_audio = add_gaussian_noise(
                audio, snr_db, rng
            )

            features = extract_features(
                noisy_audio, wavlm, device
            )

            embeddings.append(features.reshape(-1))

            if (index + 1) % 10 == 0:
                print(f"Processed {index + 1}/{len(filepaths)}")

        X_noisy = np.asarray(embeddings, dtype=np.float64)
        assert np.isfinite(X_noisy).all()

        for name, classifier in classifiers.items():
            result = evaluate(
                classifier,
                X_noisy,
                y_val,
                name,
                f"{snr_db}db",
            )
            results.append(result)

            print(
                f"{name}: "
                f"Balanced Acc = "
                f"{result['balanced_accuracy']:.4f}, "
                f"AUC = {result['roc_auc']:.4f}, "
                f"EER = {result['eer']:.4f}"
            )

    RESULTS_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(RESULTS_PATH, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved: {RESULTS_PATH}")

    plot_results(results)


if __name__ == "__main__":
    main()
