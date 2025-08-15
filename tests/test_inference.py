import torch
import numpy as np
import pytest

from backend.inference import InferenceWrapper
from ml.model import LSTMAutoencoder

def test_model_equivalence():
    # 1. Define hyperparameters
    input_dim = 4
    hidden_dim = 64
    latent_dim = 16

    # 2. Load the original model and weights
    model = LSTMAutoencoder(input_dim, hidden_dim, latent_dim)
    model.load_state_dict(torch.load("ml/lstm_autoencoder.pt", map_location=torch.device("cpu")))
    model.eval()

    # 3. Instantiate the InferenceWrapper
    inference_wrapper = InferenceWrapper(model_dir="ml")

    # 4. Create a sample input
    sample_input = [0.1, -0.2, 0.3, -0.4]

    # 5. Get prediction from the original model
    input_tensor = torch.tensor(sample_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 4)
    with torch.no_grad():
        reconstructed = model(input_tensor)
        sse = torch.nn.functional.mse_loss(reconstructed, input_tensor, reduction="sum").item()

    # 6. Get prediction from the InferenceWrapper
    pred = inference_wrapper.predict(sample_input)
    wrapper_sse = pred["total_reconstruction_error"]

    # 7. Assert that the SSEs are numerically very close
    np.testing.assert_allclose(sse, wrapper_sse, rtol=1e-5, atol=1e-6)
