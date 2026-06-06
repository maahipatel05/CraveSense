"""
CraveSense — core data loading, feature engineering, and imputation pipeline.

This is the central data module. It is responsible for transforming five raw
clinical data sources into a single analysis-ready feature matrix.

Pipeline overview:
  1. Load raw EMA, Fitbit, fMRI, demographics, and survey files.
  2. Normalize EMA scales across collection phases (phase-aware).
  3. Compute Fitbit time-window features for three window sizes (15, 30, 45 min)
     and two temporal shifts (shift=0 for detection; shift=90 for forecasting).
  4. Extract 32-dim LSTM latent features from the pre-trained Fitbit autoencoder
     for 45-minute windows (using a 90-minute lookback for richer context).
  5. Extract 16-dim feedforward latent features from the pre-trained fMRI
     autoencoder for each participant's resting-state connectivity matrix.
  6. Merge all modalities on User_ID.
  7. Apply smart imputation: stratified by time-of-day (Morning/Afternoon/
     Evening/Night) and activity level, with a physiological boost (×1.15 HR)
     applied to windows where steps indicate the participant was active but
     heart rate is missing.

Side effects on import:
  Loads PyTorch autoencoder weights from disk (crave_fitbit_ae.pth and
  crave_fmri_ae.pth). If weights are missing, autoencoders fall back to None
  and latent features are filled with NaN (pipeline still runs, but with
  degraded feature set). Run train_multimodal_encoders.py first.

Constants:
  WINDOW_SIZES = [15, 30, 45]  — Fitbit aggregation windows in minutes
  SHIFTS       = [0, 90]       — temporal shifts for detection vs. forecasting
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

WINDOW_SIZES = [15, 30, 45]
SHIFTS = [0, 90] 

class FitbitAutoencoder(nn.Module):
    def __init__(self, seq_len=90, n_features=2, embedding_dim=32):
        super(FitbitAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=embedding_dim, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=embedding_dim, hidden_size=n_features, num_layers=1, batch_first=True)

    def forward(self, x):
        enc_output, (hidden, cell) = self.encoder(x)
        latent_vector = hidden[-1] 
        x_decoded = latent_vector.unsqueeze(1).repeat(1, 90, 1)
        dec_output, _ = self.decoder(x_decoded)
        return dec_output, latent_vector

class fMRIAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=16):
        super(fMRIAutoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, embedding_dim))
        self.decoder = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, input_dim))

    def forward(self, x):
        latent_vector = self.encoder(x)
        reconstructed = self.decoder(latent_vector)
        return reconstructed, latent_vector

try:
    ae_fitbit = FitbitAutoencoder(seq_len=90, n_features=2, embedding_dim=32)
    ae_fitbit.load_state_dict(torch.load('crave_fitbit_ae.pth', map_location=torch.device('cpu')))
    ae_fitbit.eval()
    print("✅ Fitbit Neural Net (MIL Bucket) loaded successfully!")
except Exception as e:
    ae_fitbit = None
    print("⚠️ Warning: Fitbit Autoencoder model not found.")

try:
    col_order = pd.read_csv('fmri_col_order.csv').iloc[:, 0].tolist()
    ae_fmri = fMRIAutoencoder(input_dim=len(col_order), embedding_dim=16)
    ae_fmri.load_state_dict(torch.load('crave_fmri_ae.pth', map_location=torch.device('cpu')))
    ae_fmri.eval()
    print("✅ fMRI Neural Net loaded successfully!")
except Exception as e:
    ae_fmri = None
    print("⚠️ Warning: fMRI Autoencoder model not found.")

def clean_ids(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if col_name in df.columns:
        df[col_name] = df[col_name].astype(str).str.strip().str.upper()
    return df

def get_fitbit_features(ema_row, fitbit_df, window_minutes, shift_minutes=0):
    uid = ema_row['User_ID']
    end_time = ema_row['Start'] - timedelta(minutes=shift_minutes)
    start_time = end_time - timedelta(minutes=window_minutes)
    
    prefix = f"Forecast{shift_minutes}_" if shift_minutes > 0 else ""
    col_names = [f'{prefix}HR_Mean_{window_minutes}', f'{prefix}HR_Std_{window_minutes}', f'{prefix}Steps_{window_minutes}']
    
    if window_minutes == 45:
        col_names += [f'{prefix}Latent_Fitbit_{i}' for i in range(32)]
    
    if uid not in fitbit_df.index:
        return pd.Series([np.nan]*len(col_names), index=col_names)
    
    user_data = fitbit_df.loc[uid] 
    if isinstance(user_data, pd.Series): user_data = user_data.to_frame().T

    mask = (user_data['DateTime'] >= start_time) & (user_data['DateTime'] <= end_time)
    window = user_data[mask].copy()
    
    if window.empty:
        return pd.Series([np.nan]*len(col_names), index=col_names)
    
    # 1. Standard Flat Features
    feats = [window['HR'].mean(), window['HR'].std(), window['Steps'].sum()]
    
    # 2. Neural Network Latent Features (Only for 45 min, using 90 min lookback)
    if window_minutes == 45:
        if ae_fitbit is not None:
            try:
                start_time_90 = end_time - timedelta(minutes=90)
                mask_90 = (user_data['DateTime'] >= start_time_90) & (user_data['DateTime'] <= end_time)
                window_90 = user_data[mask_90].copy()
                
                tmp_window = window_90[['DateTime', 'HR', 'Steps']].drop_duplicates(subset=['DateTime']).set_index('DateTime')
                tmp_resampled = tmp_window.resample('1min').mean().ffill().fillna(0)
                seq = tmp_resampled.values
                
                if len(seq) > 90: seq = seq[-90:]
                elif len(seq) < 90: seq = np.vstack([np.zeros((90 - len(seq), 2)), seq])
                
                scaler = StandardScaler()
                seq_scaled = scaler.fit_transform(seq)
                tensor_in = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    _, latent = ae_fitbit(tensor_in)
                    latent = latent.numpy().flatten()
                feats.extend(latent.tolist())
            except Exception as e:
                feats.extend([np.nan]*32)
        else:
            feats.extend([np.nan]*32)
            
    return pd.Series(feats, index=col_names)

def process_fmri_latent(fmri_df):
    if ae_fmri is None: return fmri_df
    try:
        valid_users = fmri_df.dropna(subset=col_order).copy()
        if valid_users.empty: return fmri_df
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(valid_users[col_order].values)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            _, latent = ae_fmri(X_tensor)
            
        latent_cols = [f'Latent_fMRI_{i}' for i in range(16)]
        latent_df = pd.DataFrame(latent.numpy(), index=valid_users.index, columns=latent_cols)
        latent_df['User_ID'] = valid_users['User_ID'].values
        
        return pd.merge(fmri_df, latent_df, on='User_ID', how='left')
    except Exception as e:
        print(f"Warning: fMRI latent extraction failed {e}")
        return fmri_df

def get_time_of_day(hour: int) -> str:
    if 5 <= hour < 12: return 'Morning'
    elif 12 <= hour < 17: return 'Afternoon'
    elif 17 <= hour < 22: return 'Evening'
    else: return 'Night'

def smart_impute(df):
    print("--- 5. RUNNING SMART IMPUTATION (Time-of-Day + Activity) ---")
    df['Time_of_Day'] = df['Hour'].apply(get_time_of_day)
    tod_counts = df['Time_of_Day'].value_counts()
    print(f"  Time-of-Day distribution: {tod_counts.to_dict()}")

    ignore_impute = ['Craving_Intensity', 'Craving_Binary', 'Target_Now', 'Hour', 'User_ID', 'Prev_Time', 'Hours_Since_Prev']
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    generic_cols = [c for c in numeric_cols if c not in ignore_impute and 'HR_' not in c and 'Steps_' not in c and 'NW' not in c and 'Latent_' not in c]
    df[generic_cols] = df.groupby('User_ID')[generic_cols].transform(lambda x: x.fillna(x.median()))
    df[generic_cols] = df[generic_cols].fillna(df[generic_cols].median()) 

    for shift in SHIFTS:
        prefix = f"Forecast{shift}_" if shift > 0 else ""
        for w in WINDOW_SIZES:
            hr_col = f'{prefix}HR_Mean_{w}'
            steps_col = f'{prefix}Steps_{w}'
            latent_cols = [f'{prefix}Latent_Fitbit_{i}' for i in range(32)]
            
            if hr_col not in df.columns or steps_col not in df.columns: continue

            missing_hr_mask = df[hr_col].isna()

            df[steps_col] = df.groupby(['User_ID', 'Time_of_Day'])[steps_col].transform(lambda x: x.fillna(x.median()))
            df[steps_col] = df.groupby('User_ID')[steps_col].transform(lambda x: x.fillna(x.median()))
            df[steps_col] = df[steps_col].fillna(0) 

            df[hr_col] = df.groupby(['User_ID', 'Time_of_Day'])[hr_col].transform(lambda x: x.fillna(x.median()))
            df[hr_col] = df.groupby('User_ID')[hr_col].transform(lambda x: x.fillna(x.median()))
            df[hr_col] = df[hr_col].fillna(df[hr_col].median())

            if w == 45:
                for l_col in latent_cols:
                    if l_col in df.columns:
                        df[l_col] = df.groupby(['User_ID', 'Time_of_Day'])[l_col].transform(lambda x: x.fillna(x.median()))
                        df[l_col] = df.groupby('User_ID')[l_col].transform(lambda x: x.fillna(x.median()))
                        df[l_col] = df[l_col].fillna(df[l_col].median())

            active_mask = missing_hr_mask & (df[steps_col] > 50)
            df.loc[active_mask, hr_col] = df.loc[active_mask, hr_col] * 1.15

    df = df.drop(columns=['Time_of_Day'])
    return df

def generate_master_dataset(apply_imputation=True):
    print("--- 1. LOADING RAW FILES ---")
    ema = pd.read_csv('Updated_CombineEMA-2.csv')
    fitbit = pd.read_csv('Crave_Pilot_Fitbit.csv')
    fmri = pd.read_csv('All_fMRI_connectivity_features.csv')
    demo = pd.read_csv('Crave_Demographics.csv')
    surveys = pd.read_csv('Crave_Surveys.csv')
    
    ema = clean_ids(ema, 'User_ID')
    fitbit = clean_ids(fitbit, 'Participant')
    fmri = clean_ids(fmri, 'User_ID')
    demo = clean_ids(demo, 'Crave-ID')
    surveys = clean_ids(surveys, 'User_ID')
    print(f"  EMA: {ema.shape}, Fitbit: {fitbit.shape}, fMRI: {fmri.shape}")
    print(f"  Demographics: {demo.shape}, Surveys: {surveys.shape}")

    print("--- 2. PREPROCESSING EMA ---")
    ema['Start'] = pd.to_datetime(ema['Start'])
    ema = ema.sort_values(by=['User_ID', 'Start'])
    ema['Target_Now'] = ema['Craving_Binary']
    ema['Hour'] = ema['Start'].dt.hour
    
    def normalize_stress(row):
        val = row['Stress']
        return val / 6.0 if row['Data Collection (Phase)'] == 'Phase 1' else val / 12.0
    ema['Stress_Norm'] = ema.apply(normalize_stress, axis=1).clip(0, 1)
    ema['Mood_Norm'] = ema['Mood'] / 12.0 
    
    ema['Prev_Time'] = ema.groupby('User_ID')['Start'].shift(1)
    ema['Hours_Since_Prev'] = (ema['Start'] - ema['Prev_Time']).dt.total_seconds() / 3600.0
    
    ema['Stress_Prev'] = ema.groupby('User_ID')['Stress_Norm'].shift(1)
    ema['Mood_Prev'] = ema.groupby('User_ID')['Mood_Norm'].shift(1)
    ema['Craving_Prev'] = ema.groupby('User_ID')['Craving_Binary'].shift(1)
    
    ema['Delta_Stress'] = ema['Stress_Norm'] - ema['Stress_Prev']
    ema['Delta_Mood'] = ema['Mood_Norm'] - ema['Mood_Prev']
    
    ema['Delta_Stress_Prev'] = ema.groupby('User_ID')['Delta_Stress'].shift(1)
    ema['Delta_Mood_Prev'] = ema.groupby('User_ID')['Delta_Mood'].shift(1)
    
    ema = ema.dropna(subset=['Stress_Prev'])
    
    print(f"--- 3. MERGING FITBIT & LSTM FEATURES (Shifts: {SHIFTS} mins) ---")
    fitbit['DateTime'] = pd.to_datetime(fitbit['DateTime'])
    fitbit = fitbit.set_index('Participant')
    
    for shift in SHIFTS:
        for w in WINDOW_SIZES:
            feats = ema.apply(lambda row: get_fitbit_features(row, fitbit, w, shift_minutes=shift), axis=1)
            ema = pd.concat([ema, feats], axis=1)
    
    print("--- 4. MERGING STATIC FEATURES & fMRI DENSE FEATURES ---")
    demo = demo.drop_duplicates(subset=['Crave-ID'])
    surveys = surveys.drop_duplicates(subset=['User_ID'])
    fmri = fmri.drop_duplicates(subset=['User_ID'])
    
    fmri = process_fmri_latent(fmri)
    
    static = pd.merge(demo, surveys, left_on='Crave-ID', right_on='User_ID', how='outer')
    static['User_ID'] = static['Crave-ID'].fillna(static['User_ID'])
    static = static.drop(columns=['Crave-ID'])
    static = pd.merge(static, fmri, on='User_ID', how='outer')
    final_df = pd.merge(ema, static, on='User_ID', how='left')
    
    if apply_imputation:
        final_df = smart_impute(final_df)
        
    return final_df