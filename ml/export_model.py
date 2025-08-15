import torch
from ml.model import LSTMAutoencoder

# 1. Define hyperparameters (must match training)
input_dim = 4
hidden_dim = 64
latent_dim = 16

# 2. Instantiate model and load weights
model = LSTMAutoencoder(input_dim, hidden_dim, latent_dim)
model.load_state_dict(torch.load("ml/lstm_autoencoder.pt", map_location=torch.device("cpu")))
model.eval()
print("Loaded model weights from ml/lstm_autoencoder.pt")

# 3. Create dummy input for tracing (batch=1, seq_len=1, input_dim=4)
dummy_input = torch.randn(1, 1, input_dim, dtype=torch.float32)
print(f"Dummy input shape for tracing: {dummy_input.shape}")

# 4. Trace the model
traced_model = torch.jit.trace(model, dummy_input)
print("Model successfully traced to TorchScript.")

# 5. Save the TorchScript model
torchscript_path = "ml/lstm_autoencoder.torchscript.pt"
traced_model.save(torchscript_path)
print(f"TorchScript model saved to {torchscript_path}")
