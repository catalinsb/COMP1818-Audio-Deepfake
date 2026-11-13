
import numpy as np
import torch
from transformers import WavLMModel

from src.preprocessing import load_audio
from src.features import extract_features


def main():
    # Select Apple GPU if available
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Device:", device)

    # Load pretrained WavLM Base
    model = WavLMModel.from_pretrained(
        "microsoft/wavlm-base"
    ).to(device)

    model.eval()

    # Load the existing audio test recording
    audio, sr = load_audio(
        "tests/fixtures/test.aiff",
        target_sr=16000
    )

    print("Sample rate:", sr)
    print("Audio shape:", audio.shape)
    print("Duration:", round(len(audio) / sr, 3), "seconds")

    # Test your feature-extraction function
    features = extract_features(audio, model, device)

    print("Feature shape:", features.shape)
    print("All values finite:", np.isfinite(features).all())

    # Verify expected output
    assert sr == 16000
    assert features.shape == (1, 768)
    assert np.isfinite(features).all()

    print("All WavLM feature extraction tests passed!")


if __name__ == "__main__":
    main()
