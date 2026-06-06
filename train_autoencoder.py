"""
CraveSense — first-generation Fitbit LSTM autoencoder (reference only).

Trains a lightweight LSTM autoencoder on 45-minute sliding windows of
Fitbit heart rate and step count data. Produces a 5-dimensional latent
embedding via unsupervised reconstruction (MSE loss).

Architecture:
    Encoder: LSTM(input=2, hidden=5, layers=1)
    Decoder: LSTM(input=5, hidden=2, layers=1)
    Sequence length: 45 timesteps (one per minute)
    Embedding dim:   5

Saved weights: crave_autoencoder.pth

DEPRECATION NOTICE:
    This script is superseded by train_multimodal_encoders.py, which uses:
      - A longer sequence window (90 min vs 45 min) for richer context
      - A higher embedding dimension (32 vs 5) for greater expressiveness
      - A separate fMRI autoencoder trained jointly
    Retained for methodological reference and ablation comparison.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=5):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        
        self.encoder = nn.LSTM(
            input_size=n_features, 
            hidden_size=embedding_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        self.decoder = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=n_features, 
            num_layers=1, 
            batch_first=True
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, n_features)
        enc_output, (hidden, cell) = self.encoder(x)
        
        latent_vector = hidden[-1] 
        
        x_decoded = latent_vector.unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_output, _ = self.decoder(x_decoded)
        
        return dec_output, latent_vector


def create_sequences():
    print("Loading Raw Fitbit Data for Neural Network Training...")
    fitbit = pd.read_csv('Crave_Pilot_Fitbit.csv')
    fitbit['DateTime'] = pd.to_datetime(fitbit['DateTime'])
    
    fitbit = fitbit.sort_values(by=['Participant', 'DateTime'])
    
    scaler = StandardScaler()
    fitbit[['HR', 'Steps']] = scaler.fit_transform(fitbit[['HR', 'Steps']].fillna(0))
    
    sequences = []
    seq_length = 45 
    
    print("Chopping data into 45-minute physiological sequences...")
    for user, user_data in fitbit.groupby('Participant'):
        vals = user_data[['HR', 'Steps']].values
        # Create rolling 45-minute chunks
        for i in range(len(vals) - seq_length):
            sequences.append(vals[i:i + seq_length])
            
    X_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
    print(f"Total sequences generated: {X_tensor.shape[0]}")
    return X_tensor, scaler

def train_model():
    X_tensor, scaler = create_sequences()
    
    # Hyperparameters
    epochs = 10
    batch_size = 64
    learning_rate = 0.001
    
    dataset = TensorDataset(X_tensor, X_tensor) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = LSTMAutoencoder(seq_len=45, n_features=2, embedding_dim=5)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("\n--- Training Sequential LSTM Autoencoder ---")
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, _ in dataloader:
            optimizer.zero_grad()
            reconstructed, _ = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Reconstruction Loss: {epoch_loss/len(dataloader):.4f}")
        
    print("\nTraining Complete! Saving the model...")
    torch.save(model.state_dict(), 'crave_autoencoder.pth')
    print("Saved 'crave_autoencoder.pth'. This will be used by our Random Forest pipeline.")

if __name__ == "__main__":
    train_model()