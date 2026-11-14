
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

DATA_ROOT = Path("data/mlaad-tiny")

GENUINE_DIR = DATA_ROOT / "original" / "en"
SPOOF_DIR = DATA_ROOT / "fake" / "en"

genuine_files = sorted(GENUINE_DIR.rglob("*.wav"))
spoof_files = sorted(SPOOF_DIR.rglob("*.wav"))

print("Genuine recordings:", len(genuine_files))
print("Spoofed recordings:", len(spoof_files))

genuine_records = [
    {"filepath": str(path), "label": 0, "source": path.stem}
    for path in genuine_files
]

spoof_records = [
    {"filepath": str(path), "label": 1, "source": path.parent.name}
    for path in spoof_files
]

print("Genuine records prepared:", len(genuine_records))
print("Spoof records prepared:", len(spoof_records))
print("Example genuine:", genuine_records[0])
print("Example spoof:", spoof_records[0])

import random

RANDOM_SEED = 42
rng = random.Random(RANDOM_SEED)

selected_genuine = rng.sample(genuine_records, 200)
selected_spoof = rng.sample(spoof_records, 200)

selected_records = selected_genuine + selected_spoof
rng.shuffle(selected_records)

print("Selected genuine:", len(selected_genuine))
print("Selected spoof:", len(selected_spoof))
print("Total selected:", len(selected_records))


import re
import numpy as np

def get_source_group(filepath):
    filename = Path(filepath).name
    return re.sub(r"_f\d+\.wav$", "", filename)

groups = np.array([
    get_source_group(record["filepath"])
    for record in selected_records
])

labels = np.array([
    record["label"]
    for record in selected_records
])

# First split: training versus remaining groups
first_split = GroupShuffleSplit(
    n_splits=1,
    train_size=0.7,
    random_state=42
)

train_idx, remaining_idx = next(
    first_split.split(selected_records, labels, groups)
)

# Second split: validation versus test groups
second_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.5,
    random_state=42
)

val_local, test_local = next(
    second_split.split(
        remaining_idx,
        labels[remaining_idx],
        groups[remaining_idx]
    )
)

validation_idx = remaining_idx[val_local]
test_idx = remaining_idx[test_local]


import csv

# Assign split labels to each selected recording
for idx in train_idx:
    selected_records[idx]["split"] = "train"

for idx in validation_idx:
    selected_records[idx]["split"] = "validation"

for idx in test_idx:
    selected_records[idx]["split"] = "test"

# Check that every recording was assigned
assert all("split" in record for record in selected_records)

# Save the updated manifest
output_path = Path("experiments/dataset_manifest.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["filepath", "label", "source", "split"]
    )
    writer.writeheader()
    writer.writerows(selected_records)

print(f"Updated manifest saved to: {output_path}")
