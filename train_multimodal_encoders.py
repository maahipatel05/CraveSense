"""
CraveSense — production multimodal autoencoder training.

Trains two independent neural autoencoders for unsupervised feature extraction
from physiological and neurological data modalities. Saved weights are loaded
at inference time by data_loader.py to produce latent features for the
downstream Random Forest classifier.

--- Fitbit LSTM Autoencoder ---
Architecture:
    Encoder: LSTM(input=2, hidden=32, layers=1, batch_first=True)
    Decoder: LSTM(input=32, hidden=2, layers=1, batch_first=True)
    Sequence: 90 timesteps × [HR, Steps] (1-minute resolution)
    Embedding: 32-dimensional LSTM hidden state
Training:
    Loss: MSE reconstruction  |  Epochs: 5  |  Batch: 128  |  LR: 0.001
Output: crave_fitbit_ae.pth

--- fMRI Feedforward Autoencoder ---
Architecture:
    Encoder: Linear(496→64, ReLU) → Linear(64→16)
    Decoder: Linear(16→64, ReLU) → Linear(64→496)
    Input: 496 unique ROI-to-ROI functional connectivity values
    Embedding: 16-dimensional latent vector (static per participant)
Training:
    Loss: MSE reconstruction  |  Epochs: 20  |  Batch: 32  |  LR: 0.001
Output: crave_fmri_ae.pth, fmri_col_order.csv

Run this script ONCE before main.py. Weights are reused across all pipeline runs.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

class FitbitAutoencoder(nn.Module):
    def __init__(self, seq_len=90, n_features=2, embedding_dim=32):
        super(FitbitAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=embedding_dim, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=embedding_dim, hidden_size=n_features, num_layers=1, batch_first=True)

    def forward(self, x):
        enc_output, (hidden, cell) = self.encoder(x)
        latent_vector = hidden[-1] 
        x_decoded = latent_vector.unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_output, _ = self.decoder(x_decoded)
        return dec_output, latent_vector

class fMRIAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=16):
        super(fMRIAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent_vector = self.encoder(x)
        reconstructed = self.decoder(latent_vector)
        return reconstructed, latent_vector

def train_fitbit_model():
    print("\n--- 1. PREPARING FITBIT DATA (90-Min Window) ---")
    fitbit = pd.read_csv('Crave_Pilot_Fitbit.csv')
    fitbit['DateTime'] = pd.to_datetime(fitbit['DateTime'])
    fitbit = fitbit.sort_values(by=['Participant', 'DateTime'])
    
    scaler = StandardScaler()
    fitbit[['HR', 'Steps']] = scaler.fit_transform(fitbit[['HR', 'Steps']].fillna(0))
    
    sequences = []
    seq_length = 90 
    
    for user, user_data in fitbit.groupby('Participant'):
        vals = user_data[['HR', 'Steps']].values
        for i in range(0, len(vals) - seq_length, 2): 
            sequences.append(vals[i:i + seq_length])
            
    X_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
    print(f"Generated {X_tensor.shape[0]} sequences. Training Fitbit Network...")
    
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = FitbitAutoencoder(seq_len=90, n_features=2, embedding_dim=32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(5): 
        epoch_loss = 0
        for batch_x, _ in dataloader:
            optimizer.zero_grad()
            reconstructed, _ = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Fitbit Epoch [{epoch+1}/5] Loss: {epoch_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), 'crave_fitbit_ae.pth')
    print(" Fitbit Autoencoder saved ('crave_fitbit_ae.pth')")

def train_fmri_model():
    print("\n--- 2. PREPARING fMRI DATA ---")
    fmri = pd.read_csv('All_fMRI_connectivity_features.csv')
    
    nw_cols = [c for c in fmri.columns if "__" in c and c.split("__")[0].startswith("NW") and c.split("__")[1].startswith("NW")]
    X_fmri = fmri[nw_cols].dropna().values
    
    scaler = StandardScaler()
    X_fmri_scaled = scaler.fit_transform(X_fmri)
    
    X_tensor = torch.tensor(X_fmri_scaled, dtype=torch.float32)
    print(f"Training fMRI Network on {X_tensor.shape[1]} input dimensions...")
    
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = fMRIAutoencoder(input_dim=X_tensor.shape[1], embedding_dim=16)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(20): 
        epoch_loss = 0
        for batch_x, _ in dataloader:
            optimizer.zero_grad()
            reconstructed, _ = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    print(f"  fMRI Final Epoch Loss: {epoch_loss/len(dataloader):.4f}")
    
    torch.save(model.state_dict(), 'crave_fmri_ae.pth')
    print(" fMRI Autoencoder saved ('crave_fmri_ae.pth')")
    
    pd.Series(nw_cols).to_csv('fmri_col_order.csv', index=False)

if __name__ == "__main__":
    train_fitbit_model()
    train_fmri_model()