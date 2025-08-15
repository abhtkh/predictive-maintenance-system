# The correct, updated ml/train.py

import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Add project root to sys.path
# ... (you can add the path fix logic here if you need it)

from ml.model import LSTMAutoencoder

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    sensor_cols = ["vibration_x", "vibration_y", "temperature", "current"]
    data = df[sensor_cols].values.astype(np.float32)

    # Scale the data and save the scaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    joblib.dump(scaler, "ml/scaler.joblib") # This is the crucial line
    print("Scaler saved to ml/scaler.joblib")
    
    tensor_data = torch.from_numpy(data_scaled).unsqueeze(0)
    return tensor_data

def train():
    # Hyperparameters
    input_dim = 4
    hidden_dim = 64
    latent_dim = 16
    num_epochs = 100
    learning_rate = 0.001

    # Load data from our new realistic dataset
    data = load_and_preprocess_data("data/realistic_normal_data.csv")
    
    model = LSTMAutoencoder(input_dim, hidden_dim, latent_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting model retraining with realistic data...")
    model.train()
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.6f}")

    # Save the newly trained model
    torch.save(model.state_dict(), "ml/lstm_autoencoder.pt")
    print("New model saved to ml/lstm_autoencoder.pt")

if __name__ == "__main__":
    train()