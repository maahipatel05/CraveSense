"""
CraveSense — phase-aware EMA preprocessing utilities.

Handles normalization of self-report scales across two data collection phases
that used incompatible rating ranges:

  Phase 1:  Stress 0–6   |  Mood 1–5
  Phase 2:  Stress 0–12  |  Mood 0–12

Both scales are mapped to [0, 1] before any downstream feature engineering
or modeling. Without this step, Phase 1 and Phase 2 data are not comparable
and any learned model would be confounded by collection protocol rather than
true psychological state.

Also handles deduplication: survey entries within 60 minutes of a previous
entry for the same participant are dropped (keeping the first), as they likely
represent accidental re-submissions or protocol violations.

Note: normalization logic is also re-implemented inline in data_loader.py
for the full pipeline. This module is retained as a standalone preprocessing
utility and for methodological transparency.
"""
import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    """
    Loads data, normalizes scales between Phase 1/2, and removes conflicting duplicates.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # 1. Convert Timestamps
    df['Start'] = pd.to_datetime(df['Start'])
    df['End'] = pd.to_datetime(df['End'])
    
    # 2. Feature Engineering: Extract Hour
    df['Hour'] = df['Start'].dt.hour

    # 3. Normalize Stress & Mood (Handling Phase changes)
    # Phase 1: Stress 0-6, Mood 1-5
    # Phase 2: Stress 0-12, Mood 0-12
    # We normalize everything to 0-1 range
    
    def normalize_stress(row):
        val = row['Stress']
        if row['Data Collection (Phase)'] == 'Phase 1':
            return val / 6.0
        else:
            return val / 12.0

    def normalize_mood(row):
        val = row['Mood']
        if row['Data Collection (Phase)'] == 'Phase 1':
            # Mood 1-5 -> min 1, range 4
            return (val - 1) / 4.0 
        else:
            return val / 12.0

    df['Stress_Norm'] = df.apply(normalize_stress, axis=1).clip(0, 1)
    df['Mood_Norm'] = df.apply(normalize_mood, axis=1).clip(0, 1)

    # 4. Deduplication / Conflict Resolution
    # Sort by User and Time
    df = df.sort_values(by=['User_ID', 'Start'])
    
    # Calculate time difference between consecutive entries for the same user
    df['Time_Diff'] = df.groupby('User_ID')['Start'].diff().dt.total_seconds() / 60.0 # in minutes
    
    # Logic: If a user has two entries within 60 minutes, the second one is likely a duplicate or conflict.
    # We keep only the first entry (where Time_Diff is NaN or >= 60)
    original_count = len(df)
    df_clean = df[(df['Time_Diff'].isna()) | (df['Time_Diff'] >= 60)].copy()
    dropped_count = original_count - len(df_clean)
    
    print(f"Data Cleaning Complete. Dropped {dropped_count} conflicting/duplicate entries.")
    print(f"Final dataset shape: {df_clean.shape}")
    
    return df_clean