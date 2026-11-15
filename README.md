WavLM Audio Deepfake Detection
COMP1818 — Artificial Intelligence Applications

An experimental comparison of audio deepfake detection under background noise and Opus compression. The project uses WavLM to extract speech embeddings and logistic regression to classify recordings as genuine or spoofed.

Two classifiers are compared:
- Baseline: trained on clean speech recordings.
- Noise-augmented: trained on clean recordings and Gaussian-noise-augmented recordings.
Both classifiers use the same WavLM feature extractor and classification architecture. WavLM is used for feature extraction; its parameters are not fine-tuned.
Requirements
- Python 3.9
- Git
- FFmpeg with the libopus encoder
- Python dependencies listed in requirements.txt
- MLAAD-tiny audio dataset (downloaded separately)
- Internet access for the initial dataset and WavLM model downloads
The project was developed on macOS with Apple Silicon. It uses PyTorch MPS when available and falls back to CPU otherwise. CPU execution may be slower.
Installation
Clone the repository and enter the project directory:
git clone https://github.com/catalinsb/COMP1818-Audio-Deepfake.git
cd COMP1818-Audio-Deepfake
Create and activate a virtual environment:
python3.9 -m venv .venv
source .venv/bin/activate
Install the Python dependencies:
python -m pip install -r requirements.txt
Install FFmpeg separately if it is not already available. On macOS with Homebrew:
brew install ffmpeg
Dataset
The project uses the English-language audio files from MLAAD-tiny. The audio dataset is not stored in this repository.

Download the dataset from its source and place the English WAV recordings in this layout:
data/
└── mlaad-tiny/
    ├── original/
    │   └── en/
    └── fake/
        └── en/
The file experiments/dataset_manifest.csv records the 400 selected recordings, their class labels and their assigned train, validation or test partition. The paths in that manifest are relative to the repository root and must match the downloaded audio files.

The fixed split contains 277 training recordings, 54 validation recordings and 69 test recordings. The manifest should be retained unchanged when reproducing the published experiments.

Please refer to the original dataset repository for its licensing and non-commercial use conditions.
Running the experiments
Run all commands from the project root with the virtual environment activated.
1. Extract WavLM features
python -m src.build_features
This reads the fixed dataset manifest, loads microsoft/wavlm-base, and generates experiments/wavlm_features_grouped.npz. On the first run, the WavLM model will be downloaded automatically.
2. Train the classifiers
python -m src.train_classifier
python -m src.train_augmented
These commands train the baseline and augmented logistic regression classifiers and save their fitted pipelines to models/.
3. Run the final evaluation
python -m src.evaluate_final
This evaluates both classifiers on the fixed test partition across clean speech, 16-bit PCM conversion, Gaussian noise (20, 10 and 5 dB SNR), and Opus compression (64, 32 and 16 kbps).

The evaluation creates:
experiments/final/
├── test_metrics.csv
├── test_predictions.csv
├── noise_comparison.png
└── codec_comparison.png
The CSV files contain aggregate metrics and per-recording predictions. The figures compare the models' balanced accuracy across audio conditions.
Optional validation experiments
python -m src.evaluate_robustness
python -m src.evaluate_codec
These scripts evaluate the validation partition used during development. src.evaluate_final is the held-out test evaluation.
Results
The committed results, if available in experiments/final/, can be inspected directly without retraining the classifiers. For the experimental findings, interpretation and limitations, refer to the accompanying coursework report.

The classification metrics include accuracy, balanced accuracy, F1 score, ROC-AUC and equal error rate (EER), together with confusion-matrix counts. Labels are 0 for genuine speech and 1 for spoofed speech.
Project structure
src/
├── preprocessing.py       # Load, convert to mono and resample audio
├── features.py            # WavLM feature extraction
├── augmentation.py        # Gaussian noise augmentation
├── codec.py               # Opus encode/decode using FFmpeg
├── build_features.py      # Rebuild feature cache from the manifest
├── train_classifier.py    # Clean baseline training
├── train_augmented.py     # Noise-augmented classifier training
├── evaluate_robustness.py # Validation noise evaluation
├── evaluate_codec.py      # Validation codec evaluation
└── evaluate_final.py      # Final held-out evaluation

experiments/
├── dataset_manifest.csv
└── final/                 # Results and figures
Generated feature archives (.npz), saved classifier files (.joblib), virtual environments and downloaded audio data are not included in the repository. They are recreated locally using the steps above.
Reproducibility notes
The same recorded splits are used for both classifiers, and random seeds are fixed for augmentation and evaluation. Results may vary slightly with library versions, hardware and numerical backends.

The dataset subset is relatively small, and separation by the defined recording groups does not guarantee speaker-independent or synthesis-system-independent evaluation. The tests measure controlled offline degradation, not live telephone performance. These limitations are discussed in the coursework report.
References
- Microsoft WavLM-Base
- MLAAD-tiny dataset
See the coursework report for academic references and the detailed experimental methodology.