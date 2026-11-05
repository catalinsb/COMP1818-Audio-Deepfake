import torch
from transformers import WavLMModel

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = WavLMModel.from_pretrained(
    "microsoft/wavlm-base"
).to(device)

model.eval()

# Synthetic input only tests execution, not detection.
audio = torch.zeros(1, 16000, device=device)

with torch.inference_mode():
    output = model(input_values=audio)

print("Device:", device)
print("Output shape:", output.last_hidden_state.shape)
