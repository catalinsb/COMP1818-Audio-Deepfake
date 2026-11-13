import soundfile as sf
import numpy as np
import librosa

def load_audio(file_path, target_sr=16000):
    """
    Load an audio recording from a file and prepare it for WavLM."""

    # 1. Read the audio waveform and sample rate.
    audio, sr = sf.read(file_path)
    if sr != target_sr:
        audio = librosa.resample(
        audio,
        orig_sr=sr,
        target_sr=target_sr
    )
    sr = target_sr

    # 2. Check whether the recording is empty.
    if audio.size == 0:
        raise ValueError("Loaded audio is empty.")

    # 3. Convert multiple channels to mono.
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    

    #Resample if sr != target_sr
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    

    # 5. Check for NaN or infinite values.
    if np.isnan(audio).any() or np.isinf(audio).any():
        raise ValueError("Loaded audio contains NaN or infinite values.")

    # 6. Return the processed audio waveform and sample rate.
    return audio, sr
