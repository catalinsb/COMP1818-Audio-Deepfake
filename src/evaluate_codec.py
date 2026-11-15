
"""Compare WavLM classifiers under Opus compression."""

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
from src.codec import opus_compress


FEATURES_PATH = Path("experiments/wavlm_features_grouped.npz")

MODEL_PATHS = {
    "Baseline": Path("models/wavlm_logreg_baseline.joblib"),
    "Augmented": Path("models/wavlm_logreg_augmented.joblib"),
}

RESULTS_PATH = Path("experiments/codec_comparison.csv")
CHART_PATH = Path("experiments/codec_comparison.png")

BITRATES = [64000, 32000, 16000]


def calculate_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr

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


def evaluate(classifier, X, labels, name, condition):
    predictions = classifier.predict(X)
    scores = classifier.predict_proba(X)[:, 1]

    assert np.isfinite(scores).all()

    return {
        "model": name,
        "condition": condition,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "f1": float(f1_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "eer": calculate_eer(labels, scores),
    }


def plot_results(results):
    conditions = ["clean", "64kbps", "32kbps", "16kbps"]
    labels = ["Clean", "64 kbps", "32 kbps", "16 kbps"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name in MODEL_PATHS:
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
            label=model_name
        )

    ax.set_title(
        "WavLM Deepfake Detection: Opus Codec Robustness"
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
        labels = data["y"][mask]
        filepaths = data["filepaths"][mask]

    assert len(labels) == 54
    assert np.isfinite(X_clean).all()

    classifiers = {
        name: joblib.load(path)
        for name, path in MODEL_PATHS.items()
    }

    results = []

    # Clean reference results
    for name, classifier in classifiers.items():
        results.append(
            evaluate(
                classifier, X_clean, labels, name, "clean"
            )
        )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Loading WavLM on {device}...")

    wavlm = WavLMModel.from_pretrained(
        "microsoft/wavlm-base"
    ).to(device)

    wavlm.eval()

    for bitrate in BITRATES:
        condition = f"{bitrate // 1000}kbps"
        print(f"\nEvaluating Opus {condition}...")

        embeddings = []

        for index, filepath in enumerate(filepaths):
            audio, sr = load_audio(filepath)

            compressed = opus_compress(
                audio,
                sample_rate=sr,
                bitrate=bitrate
            )

            features = extract_features(
                compressed, wavlm, device
            )

            embeddings.append(features.reshape(-1))

            if (index + 1) % 10 == 0:
                print(
                    f"Processed {index + 1}/{len(filepaths)}"
                )

        X_codec = np.asarray(
            embeddings, dtype=np.float64
        )

        assert np.isfinite(X_codec).all()

        for name, classifier in classifiers.items():
            result = evaluate(
                classifier,
                X_codec,
                labels,
                name,
                condition
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
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved: {RESULTS_PATH}")

    plot_results(results)


if __name__ == "__main__":
    main()
