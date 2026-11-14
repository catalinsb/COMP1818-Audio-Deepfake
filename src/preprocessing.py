
import numpy as np
import soundfile as sf
import librosa


def load_audio(file_path, target_sr=16000):
    """Load audio, convert to mono and resample for WavLM."""

    audio, sr = sf.read(file_path)

    if audio.size == 0:
        raise ValueError("Loaded audio is empty.")

    # Convert stereo recordings to mono first
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample to the required sampling rate
    if sr != target_sr:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=target_sr
        )

    audio = np.asarray(audio, dtype=np.float32)

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains invalid values.")

    return audio, target_sr
