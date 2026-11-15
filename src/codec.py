
"""Opus codec compression for audio robustness experiments."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def opus_compress(audio, sample_rate=16000, bitrate=32000):
    """Encode and decode mono audio using Opus via FFmpeg."""

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Expected a non-empty mono waveform.")

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains invalid values.")

    if bitrate not in (16000, 32000, 64000):
        raise ValueError("Unsupported bitrate.")

    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)

        input_file = temp / "input.wav"
        opus_file = temp / "compressed.opus"
        output_file = temp / "decoded.wav"

        # Use a consistent PCM input format
        sf.write(
            input_file,
            audio,
            sample_rate,
            subtype="PCM_16"
        )

        # Encode with Opus
        encode_command = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(input_file),
            "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "libopus",
            "-application", "voip",
            "-b:a", str(bitrate),
            str(opus_file),
        ]

        subprocess.run(
            encode_command,
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )

        # Decode back to WAV
        decode_command = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(opus_file),
            "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(output_file),
        ]

        subprocess.run(
            decode_command,
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )

        decoded, decoded_sr = sf.read(
            output_file, dtype="float32"
        )

    if decoded_sr != sample_rate:
        raise RuntimeError("Unexpected decoded sample rate.")

    if decoded.ndim != 1 or not np.isfinite(decoded).all():
        raise RuntimeError("Invalid decoded audio.")

    if len(decoded) != len(audio):
        raise RuntimeError(
            f"Audio length changed: {len(audio)} -> "
            f"{len(decoded)} samples"
        )

    return decoded
