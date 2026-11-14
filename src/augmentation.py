
import numpy as np


def add_gaussian_noise(audio, snr_db, rng=None):
    """Add Gaussian noise to audio at a specified SNR."""
    if rng is None:
        rng = np.random.default_rng()

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        raise ValueError("Audio cannot be empty")

    signal_power = np.mean(audio.astype(np.float64) ** 2)

    if signal_power == 0:
        raise ValueError("Cannot add SNR-based noise to silent audio")

    # Calculate the target noise power
    noise_power = signal_power / (10 ** (snr_db / 10))

    # Generate Gaussian noise
    noise = rng.standard_normal(audio.shape).astype(np.float32)

    # Scale noise to the requested power
    actual_power = np.mean(noise.astype(np.float64) ** 2)
    noise *= np.sqrt(noise_power / actual_power)

    # Add noise to the original waveform
    return (audio + noise).astype(np.float32)
