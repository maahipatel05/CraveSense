"""
CraveSense — Fitbit data density diagnostic tool.

Quantifies how many Fitbit sensor readings are available within each 45-minute
survey window across the full EMA dataset. This diagnostic directly informs
the imputation strategy: if most windows have zero Fitbit readings, imputation
is unavoidable; if most have dense coverage, stricter thresholds are justified.

Outputs:
  - Imputation_Histogram.png: log-scale histogram of Fitbit readings per window
  - Console table: dataset size N as a function of minimum reading threshold
    (0, 1, 5, 15, 30 readings), shown separately for sensor-only and full
    multimodal (requires fMRI) subsets

Design rationale:
  The 45-minute window was chosen to match the survey response latency: a
  participant typically completes a survey within ~45 minutes of a craving
  event. Windows with fewer than 1 reading are fully imputed. The diagnostic
  helps the team decide whether to tighten this threshold at the cost of N.

Usage:
    python diagnostic_imputation.py

Requires: Updated_CombineEMA-2.csv, Crave_Pilot_Fitbit.csv,
          All_fMRI_connectivity_features.csv in the working directory.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import os

def clean_ids(df, col_name):
    if col_name in df.columns:
        df[col_name] = df[col_name].astype(str).str.strip().str.upper()
    return df

def run_diagnostic() -> None:
    print("--- 1. LOADING RAW FILES FOR DIAGNOSTIC ---")
    ema = pd.read_csv('Updated_CombineEMA-2.csv')
    fitbit = pd.read_csv('Crave_Pilot_Fitbit.csv')
    fmri = pd.read_csv('All_fMRI_connectivity_features.csv')
    
    ema = clean_ids(ema, 'User_ID')
    fitbit = clean_ids(fitbit, 'Participant')
    fmri = clean_ids(fmri, 'User_ID')
    
    ema['Start'] = pd.to_datetime(ema['Start'])
    fitbit['DateTime'] = pd.to_datetime(fitbit['DateTime'])
    fitbit = fitbit.set_index('Participant')
    
    print("--- 2. CALCULATING FITBIT DATA DENSITY (45-Min Window) ---")
    # We are going to count EXACTLY how many Fitbit readings exist in every 45-min window
    counts = []
    for _, row in ema.iterrows():
        uid = row['User_ID']
        end_time = row['Start']
        start_time = end_time - timedelta(minutes=45)
        
        if uid in fitbit.index:
            user_data = fitbit.loc[uid]
            if isinstance(user_data, pd.Series): user_data = user_data.to_frame().T
            mask = (user_data['DateTime'] >= start_time) & (user_data['DateTime'] <= end_time)
            valid_readings = mask.sum()
        else:
            valid_readings = 0
            
        counts.append(valid_readings)
        
    ema['Fitbit_Reading_Count'] = counts
    
    print("--- 3. GENERATING HISTOGRAM ---")
    plt.figure(figsize=(12, 6))
    
    sns.histplot(ema['Fitbit_Reading_Count'], bins=50, kde=False, color='coral')
    plt.yscale('log')
    plt.title("Histogram of Fitbit Readings per 45-Min Survey Window (Log Scale)")
    plt.xlabel("Number of Fitbit Data Points in Window (0 = Fully Missing)")
    plt.ylabel("Frequency (Log Scale)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Imputation_Histogram.png')
    print("Saved histogram as 'Imputation_Histogram.png'.")
    zero_pct = (ema['Fitbit_Reading_Count'] == 0).mean() * 100
    print(f"  Fully missing windows (0 readings): {zero_pct:.1f}% of all EMA surveys")
    
    print("\n" + "="*80)
    print("--- 4. DATA POINT COUNTS (N) BY IMPUTATION THRESHOLD ---")
    print("="*80)
    total_ema = len(ema)
    fmri_users = fmri['User_ID'].unique()
    ema['Has_fMRI'] = ema['User_ID'].isin(fmri_users)
    
    print(f"Total EMA Surveys (Maximum Possible N): {total_ema}")
    print(f"Total fMRI Surveys (Maximum Possible Multimodal N): {ema['Has_fMRI'].sum()}\n")
    
    print("How N drops based on minimum required Fitbit readings:")
    for threshold in [0, 1, 5, 15, 30]:
        valid_sensor = ema[ema['Fitbit_Reading_Count'] >= threshold]
        valid_multi = valid_sensor[valid_sensor['Has_fMRI']]
        
        if threshold == 0:
            print(f"Threshold >= {threshold:2} readings (Allow 100% Imputation) -> Sensors N: {len(valid_sensor):4} | Multimodal N: {len(valid_multi):4}")
        else:
            print(f"Threshold >= {threshold:2} readings (Limit Imputation)    -> Sensors N: {len(valid_sensor):4} | Multimodal N: {len(valid_multi):4}")
    print("="*80)

if __name__ == "__main__":
    run_diagnostic()