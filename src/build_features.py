"""Rebuild grouped WavLM embeddings from the committed dataset manifest.

Run from the project root: python -m src.build_features
"""

from pathlib import Path
import csv

import numpy as np
import torch
from transformers import WavLMModel

from src.preprocessing import load_audio
from src.features import extract_features

MANIFEST = Path("experiments/dataset_manifest.csv")
OUTPUT = Path("experiments/wavlm_features_grouped.npz")
MODEL_ID = "microsoft/wavlm-base"
VALID_SPLITS = {"train", "validation", "test"}


def main():
    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("Manifest has no recordings")

    filepaths = [row["filepath"] for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    splits = np.asarray([row["split"] for row in rows], dtype=str)

    if len(set(filepaths)) != len(filepaths):
        raise ValueError("Duplicate file paths in manifest")
    if not set(splits).issubset(VALID_SPLITS) or set(splits) != VALID_SPLITS:
        raise ValueError(f"Unexpected partitions: {set(splits)}")
    if not set(labels).issubset({0, 1}):
        raise ValueError("Labels must be 0 (genuine) or 1 (spoof)")

    missing = [path for path in filepaths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} dataset recordings are missing. "
            f"Example: {missing[0]}. Download the dataset into data/mlaad-tiny."
        )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading {MODEL_ID} on {device}")
    model = WavLMModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    features = []
    for index, path in enumerate(filepaths, 1):
        audio, sample_rate = load_audio(path)
        if sample_rate != 16000:
            raise ValueError(f"Expected 16 kHz: {path}")
        embedding = extract_features(audio, model, device).reshape(-1)
        if embedding.shape != (768,) or not np.isfinite(embedding).all():
            raise ValueError(f"Invalid embedding: {path}")
        features.append(embedding.astype(np.float32))
        if index % 25 == 0 or index == len(filepaths):
            print(f"Processed {index}/{len(filepaths)}")

    X = np.stack(features)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        X=X,
        y=labels,
        splits=splits,
        filepaths=np.asarray(filepaths, dtype=str),
    )
    print(f"Saved {OUTPUT}: {X.shape}")
    for split in ("train", "validation", "test"):
        mask = splits == split
        print(f"  {split}: {mask.sum()} recordings, label counts {np.bincount(labels[mask], minlength=2).tolist()}")


if __name__ == "__main__":
    main()
