import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, roc_curve, accuracy_score
import warnings
warnings.filterwarnings("ignore")

os.makedirs("visualizations", exist_ok=True)

def get_modality_columns(df, window_size, task_mode):
    fmri = [f'Latent_fMRI_{i}' for i in range(16)] 
    surveys = [c for c in df.columns if any(x in c for x in ['BIS', 'UPPS', 'GAD', 'PHQ', 'CAMS', 'QIDS'])]
    ignore = ['Craving_Intensity', 'Craving_Binary', 'Target_Now', 'User_ID', 'Start', 'End', 'Date', 'Participant', 'Prev_Time', 'Hours_Since_Prev']
    
    if task_mode == 'detection':
        base = ['Stress_Norm', 'Mood_Norm', 'Delta_Stress', 'Delta_Mood', 'Hour']
        sensors = [f'HR_Mean_{window_size}', f'HR_Std_{window_size}', f'Steps_{window_size}'] + [f'Latent_Fitbit_{i}' for i in range(32)]
    else:
        base = ['Stress_Prev', 'Mood_Prev', 'Craving_Prev', 'Delta_Stress_Prev', 'Delta_Mood_Prev', 'Hour']
        sensors = [f'Forecast90_HR_Mean_{window_size}', f'Forecast90_HR_Std_{window_size}', f'Forecast90_Steps_{window_size}'] + [f'Forecast90_Latent_Fitbit_{i}' for i in range(32)]
    
    return [c for c in base if c not in ignore], [c for c in sensors if c not in ignore], [c for c in surveys if c not in ignore], [c for c in fmri if c not in ignore]

def augment_with_gaussian_noise(X_minority, y_minority, target_count, noise_level=0.1):
    num_to_generate = target_count - len(X_minority)
    if num_to_generate <= 0: return X_minority, y_minority
    synthetic_base = X_minority.sample(n=num_to_generate, replace=True, random_state=42)
    stds = X_minority.std().fillna(0) 
    noise = np.random.normal(0, 1, size=synthetic_base.shape) * (noise_level * stds.values)
    synthetic_X = synthetic_base + noise
    synthetic_y = pd.Series([1]*num_to_generate, name=y_minority.name, index=synthetic_X.index)
    return pd.concat([X_minority, synthetic_X]), pd.concat([y_minority, synthetic_y])

def plot_histogram(df, mode_str):
    plt.figure(figsize=(12, 6))
    counts = df['User_ID'].value_counts()
    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.title(f'Participant Data Distribution ({mode_str})', fontsize=14)
    plt.xlabel('Participant ID')
    plt.ylabel('Number of Data Points (Rows)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'visualizations/histogram_{mode_str.lower().replace(" ", "_")}.png')
    plt.close()
    print(f"  📊 Saved Histogram to visualizations/")

def run_analysis(df, window_size=45, task_mode='detection'):
    mode_str = "DETECTION" if task_mode == 'detection' else "STANDARDIZED FORECASTING"
    print(f"\n=========================================================")
    print(f" ABLATION STUDY: MULTIMODAL FEATURE FUSION ({mode_str})")
    print(f"=========================================================")
    
    base, sensors, surveys, fmri = get_modality_columns(df, window_size, task_mode)
    
    df_ablation = df.dropna(subset=['Target_Now', 'User_ID'] + fmri).copy()
    
    if task_mode == 'forecasting' and 'Hours_Since_Prev' in df_ablation.columns:
        df_ablation = df_ablation[df_ablation['Hours_Since_Prev'] <= 24.0]
        
    print(f"  Total Valid Participants: {df_ablation['User_ID'].nunique()}")
    plot_histogram(df_ablation, mode_str)
    
    experiments = {
        'Baseline (Majority)': {'cols': base},
        'EMA Only (Psychological)': {'cols': base},
        'Sensors Only (Physical)': {'cols': sensors},
        'fMRI Only (Neural)': {'cols': fmri},
        'Sensors + fMRI (Physo-Neural)': {'cols': sensors + fmri},
        'EMA + Sensors + fMRI (Full Multimodal)': {'cols': base + sensors + fmri}
    }
    
    results = []
    
    # Setup ROC Plot
    plt.figure(figsize=(10, 8))
    
    for name, config in experiments.items():
        cols = [c for c in config['cols'] if c in df_ablation.columns]
        if not cols: continue
        
        df_exp = df_ablation.dropna(subset=cols) 
        clean_rows = len(df_exp)
        
        groups = df_exp['User_ID']
        if len(groups.unique()) < 2: continue
                
        X = df_exp[cols]
        y = df_exp['Target_Now']
        
        logo = LeaveOneGroupOut()
        f1s, recs, precs, aucs, accs = [], [], [], [], []
        
        all_y_test = []
        all_y_prob = []
        
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [2, 3, 4, 5], 'min_samples_split': [2, 5]}
        
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            if len(y_test.unique()) < 2: continue
            
            try:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                if 'Baseline' in name:
                    model = DummyClassifier(strategy='most_frequent')
                    model.fit(X_train_scaled, y_train)
                    test_probs = model.predict_proba(X_test_scaled)[:, 1]
                    pred = model.predict(X_test_scaled)
                else:
                    maj_X = pd.DataFrame(X_train_scaled[y_train == 0], columns=cols)
                    min_X = pd.DataFrame(X_train_scaled[y_train == 1], columns=cols)
                    min_y = y_train[y_train == 1]
                    
                    aug_X, aug_y = augment_with_gaussian_noise(min_X, min_y, target_count=len(maj_X), noise_level=0.1)
                    X_train_final = pd.concat([maj_X, aug_X])
                    y_train_final = pd.concat([y_train[y_train == 0], aug_y])
                    
                    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
                    search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=5, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42)
                    search.fit(X_train_final, y_train_final)
                    
                    best_model = search.best_estimator_
                    
                    # Youden's J Statistic
                    train_probs = best_model.predict_proba(X_train_final)[:, 1]
                    fpr_t, tpr_t, thresholds_t = roc_curve(y_train_final, train_probs)
                    best_threshold = thresholds_t[np.argmax(tpr_t - fpr_t)]
                    
                    test_probs = best_model.predict_proba(X_test_scaled)[:, 1]
                    pred = (test_probs >= best_threshold).astype(int)
                
                all_y_test.extend(y_test)
                all_y_prob.extend(test_probs)
                
                auc = roc_auc_score(y_test, test_probs)
                accs.append(accuracy_score(y_test, pred))
                f1s.append(f1_score(y_test, pred, zero_division=0))
                recs.append(recall_score(y_test, pred, zero_division=0))
                precs.append(precision_score(y_test, pred, zero_division=0))
                aucs.append(auc)
                
            except Exception as e:
                continue
                
        avg_auc = np.mean(aucs) if aucs else 0.5
        
        # Plot ROC curve for this modality
        if all_y_test and all_y_prob and 'Baseline' not in name:
            fpr, tpr, _ = roc_curve(all_y_test, all_y_prob)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {avg_auc:.3f})')
        
        results.append({
            'Modality': name,
            'N_Rows': clean_rows,
            'Accuracy': np.mean(accs) if accs else 0.0,
            'AUC_ROC': avg_auc,
            'Recall': np.mean(recs) if recs else 0.0,
            'Precision': np.mean(precs) if precs else 0.0,
            'F1': np.mean(f1s) if f1s else 0.0
        })
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'Receiver Operating Characteristic - {mode_str}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'visualizations/roc_{mode_str.lower().replace(" ", "_")}.png')
    plt.close()
    print(f" Saved ROC Curves to visualizations/")
    
    df_res = pd.DataFrame(results).sort_values(by='AUC_ROC', ascending=False)
    if 'Ablation_Model' in df_res.columns:
         df_res = df_res.rename(columns={'Ablation_Model': 'Modality'})
         
    return df_res