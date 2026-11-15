
"""Final held-out evaluation of WavLM deepfake classifiers."""

import csv
import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from transformers import WavLMModel
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from src.preprocessing import load_audio
from src.features import extract_features
from src.augmentation import add_gaussian_noise
from src.codec import opus_compress


FEATURES_PATH = Path("experiments/wavlm_features_grouped.npz")
OUTPUT = Path("experiments/final")
SEED = 42

MODELS = {
    "Baseline": "models/wavlm_logreg_baseline.joblib",
    "Augmented": "models/wavlm_logreg_augmented.joblib",
}

CONDITIONS = [
    "clean",
    "pcm16",
    "noise_20db",
    "noise_10db",
    "noise_5db",
    "opus_64kbps",
    "opus_32kbps",
    "opus_16kbps",
]


def pcm16_roundtrip(audio, sample_rate=16000):
    """Encode and decode 16-bit PCM WAV without Opus."""

    buffer = io.BytesIO()

    sf.write(
        buffer,
        audio,
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )

    buffer.seek(0)
    decoded, sr = sf.read(buffer, dtype="float32")

    assert sr == sample_rate
    assert len(decoded) == len(audio)
    assert np.isfinite(decoded).all()

    return decoded


def calculate_eer(labels, scores):
    """Calculate approximate EER by ROC interpolation."""

    fpr, tpr, _ = roc_curve(labels, scores)
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


def evaluate(classifier, X, labels, model_name, condition, files):
    """Return aggregate metrics and individual predictions."""

    predictions = classifier.predict(X)
    scores = classifier.predict_proba(X)[:, 1]

    assert np.isfinite(scores).all()

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "model": model_name,
        "condition": condition,
        "samples": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "f1": float(f1_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "eer": calculate_eer(labels, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    individual = []

    for filepath, label, prediction, score in zip(
        files, labels, predictions, scores
    ):
        individual.append({
            "model": model_name,
            "condition": condition,
            "filepath": str(filepath),
            "true_label": int(label),
            "predicted_label": int(prediction),
            "spoof_probability": float(score),
        })

    return metrics, individual


def save_csv(path, records):
    if not records:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=records[0].keys()
        )
        writer.writeheader()
        writer.writerows(records)


def plot_comparison(results):
    """Save noise and codec comparison figures."""

    groups = {
        "noise": [
            "clean", "noise_20db",
            "noise_10db", "noise_5db"
        ],
        "codec": [
            "clean", "pcm16",
            "opus_64kbps", "opus_32kbps",
            "opus_16kbps"
        ],
    }

    labels = {
        "clean": "Clean",
        "pcm16": "PCM-16",
        "noise_20db": "20 dB",
        "noise_10db": "10 dB",
        "noise_5db": "5 dB",
        "opus_64kbps": "64 kbps",
        "opus_32kbps": "32 kbps",
        "opus_16kbps": "16 kbps",
    }

    for group_name, conditions in groups.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        for model_name in MODELS:
            subset = {
                row["condition"]: row
                for row in results
                if row["model"] == model_name
            }

            values = [
                subset[c]["balanced_accuracy"] * 100
                for c in conditions
            ]

            ax.plot(
                [labels[c] for c in conditions],
                values,
                marker="o",
                linewidth=2,
                label=model_name,
            )

        ax.set_title(
            f"WavLM: Final {group_name.title()} Robustness"
        )
        ax.set_xlabel("Audio condition")
        ax.set_ylabel("Balanced accuracy (%)")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend()

        fig.tight_layout()

        path = OUTPUT / f"{group_name}_comparison.png"
        fig.savefig(path, dpi=300)

        print(f"Chart saved: {path}")

        plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    with np.load(FEATURES_PATH) as data:
        mask = data["splits"] == "test"

        X_clean = data["X"][mask]
        labels = data["y"][mask]
        filepaths = data["filepaths"][mask]

    assert len(labels) == 69
    assert len(np.unique(labels)) == 2
    assert np.isfinite(X_clean).all()

    print(f"Final test recordings: {len(labels)}")
    print("Class distribution:", np.unique(
        labels, return_counts=True
    ))

    classifiers = {
        name: joblib.load(path)
        for name, path in MODELS.items()
    }

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Loading WavLM on {device}...")

    wavlm = WavLMModel.from_pretrained(
        "microsoft/wavlm-base"
    ).to(device)

    wavlm.eval()

    # Check consistency with the cached clean features.
    check_audio, _ = load_audio(filepaths[0])
    check_features = extract_features(
        check_audio, wavlm, device
    ).reshape(-1)

    difference = np.max(
        np.abs(check_features - X_clean[0])
    )

    print(f"Clean feature difference: {difference:.6f}")

    if difference > 0.01:
        raise RuntimeError(
            "Cached and freshly extracted features differ."
        )

    results = []
    predictions = []

    for condition in CONDITIONS:
        print(f"\nEvaluating: {condition}")

        if condition == "clean":
            X_condition = X_clean

        else:
            embeddings = []

            for index, filepath in enumerate(filepaths):
                audio, sr = load_audio(filepath)

                if condition == "pcm16":
                    transformed = pcm16_roundtrip(
                        audio, sr
                    )

                elif condition.startswith("noise_"):
                    snr = int(
                        condition.split("_")[1].replace("db", "")
                    )

                    rng = np.random.default_rng(
                        SEED + index + snr * 10000
                    )

                    transformed = add_gaussian_noise(
                        audio, snr, rng
                    )

                elif condition.startswith("opus_"):
                    bitrate = int(
                        condition.split("_")[1].replace("kbps", "")
                    ) * 1000

                    transformed = opus_compress(
                        audio,
                        sample_rate=sr,
                        bitrate=bitrate,
                    )

                else:
                    raise ValueError(condition)

                feature = extract_features(
                    transformed, wavlm, device
                )

                embeddings.append(feature.reshape(-1))

                if (index + 1) % 10 == 0:
                    print(
                        f"Processed {index + 1}/{len(filepaths)}"
                    )

            X_condition = np.asarray(
                embeddings, dtype=np.float64
            )

            assert np.isfinite(X_condition).all()

        for name, classifier in classifiers.items():
            metrics, individual = evaluate(
                classifier,
                X_condition,
                labels,
                name,
                condition,
                filepaths,
            )

            results.append(metrics)
            predictions.extend(individual)

            print(
                f"{name:10} | "
                f"Balanced Acc: "
                f"{metrics['balanced_accuracy']:.4f} | "
                f"AUC: {metrics['roc_auc']:.4f} | "
                f"EER: {metrics['eer']:.4f}"
            )

    save_csv(OUTPUT / "test_metrics.csv", results)
    save_csv(OUTPUT / "test_predictions.csv", predictions)

    plot_comparison(results)

    print("\nFinal evaluation complete.")
    print(f"Results directory: {OUTPUT}")


if __name__ == "__main__":
    main()
