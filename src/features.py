
import numpy as np
import torch


def extract_features(audio, model, device):
    """
    Extract a fixed-length WavLM embedding from a
    preprocessed 16 kHz mono audio waveform.

    Returns:
        NumPy array with shape (1, 768).
    """

    # Convert NumPy waveform to a float32 PyTorch tensor
    waveform = torch.from_numpy(
        np.asarray(audio, dtype=np.float32)
    )

    # Add batch dimension: [samples] -> [1, samples]
    waveform = waveform.unsqueeze(0)
    

    # Move input to the device used by WavLM
    waveform = waveform.to(device)

    # Enable evaluation mode
    model.eval()

    # Extract representations without computing gradients
    with torch.inference_mode():
        outputs = model(input_values=waveform)

        # Extract the sequence of representations
        hidden_states = outputs.last_hidden_state

        # Average across the time dimension
        pooled = hidden_states.mean(dim=1)
        

    # Convert features to a NumPy array on the CPU
    features = pooled.cpu().numpy()

    # Return the fixed-length embedding
    return features
