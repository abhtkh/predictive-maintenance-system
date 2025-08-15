import os
import torch
import joblib
import numpy as np

class InferenceWrapper:
    def __init__(self, model_dir="ml"):
        # Define paths for TorchScript model and metadata
        self.model_path = os.path.join(model_dir, "lstm_autoencoder.torchscript.pt")
        self.metadata_path = os.path.join(model_dir, "model_metadata.json")  # Not required to load, but defined

        # Add scaler instance variable and path
        self.scaler = None
        scaler_path = os.path.join(model_dir, "scaler.joblib")

        # Load TorchScript model
        self.model = torch.jit.load(self.model_path, map_location=torch.device("cpu"))
        self.model.eval()

        # Load StandardScaler
        self.scaler = joblib.load(scaler_path)

        # Model metadata (could be loaded from file, but hardcoded here for simplicity)
        self.model_version = "1.0.0"
        self.training_date = "2024-06-01"
        self.alerting_threshold = 15.0  # Default threshold for anomaly

        # Number of expected features (input_dim)
        self.input_dim = 4

    def predict(self, sensor_values):
        # Validate input
        if not isinstance(sensor_values, (list, tuple)):
            raise ValueError("Input must be a list or tuple of sensor values.")
        if len(sensor_values) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} sensor values, got {len(sensor_values)}.")

        # Convert to numpy array and reshape for scaler
        sensor_values_arr = np.array(sensor_values).reshape(1, -1)
        # Scale input using StandardScaler (expects 2D array)
        scaled_values = self.scaler.transform(sensor_values_arr)

        # Convert to tensor: shape (1, 1, input_dim)
        input_tensor = torch.tensor(scaled_values, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            reconstructed = self.model(input_tensor)
            # reconstructed: (1, 1, input_dim)
            # Compute total MSE (scalar)
            total_mse = torch.nn.functional.mse_loss(reconstructed, input_tensor, reduction="sum").item()
            # Compute per-sensor squared error (list of floats)
            per_sensor_error = ((reconstructed - input_tensor) ** 2).squeeze(0).squeeze(0).tolist()
            # If input_dim == 1, tolist() returns a float, so wrap in list
            if isinstance(per_sensor_error, float):
                per_sensor_error = [per_sensor_error]

        return {
            "model_version": self.model_version,
            "total_reconstruction_error": total_mse,
            "per_sensor_error": per_sensor_error
        }
