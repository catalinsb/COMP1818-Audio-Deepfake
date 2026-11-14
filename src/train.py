import numpy as np
from pathlib import Path
import pandas as pd
import torch

from transformers import WavLMModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.preprocessing import load_audio
from src.features import extract_features

MANIFEST = Path("experiments/dataset_manifest.csv")
MODEL_NAME = "microsoft/wavlm-base"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)

manifest = pd.read_csv(MANIFEST)

print("Total recordings:", len(manifest))
print(pd.crosstab(manifest["split"], manifest["label"]))

model = WavLMModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

print("WavLM loaded successfully")


feature_vectors = []
labels = []

for index, row in manifest.iterrows():
    audio, sr = load_audio(row["filepath"])
    features = extract_features(audio, model, device)

    feature_vectors.append(features.squeeze(0))
    labels.append(int(row["label"]))

    if (index + 1) % 25 == 0:
        print(f"Processed {index + 1}/{len(manifest)} recordings")

X = np.vstack(feature_vectors)
y = np.array(labels)

print("Feature matrix shape:", X.shape)
print("Label array shape:", y.shape)


output_path = Path("experiments/wavlm_features.npz")
output_path.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    output_path,
    X=X,
    y=y,
    splits=manifest["split"].to_numpy(dtype=str),
    filepaths=manifest["filepath"].to_numpy(dtype=str)
)

print(f"Saved features to: {output_path}")

