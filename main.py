import data_loader
import modeling_advanced
import pandas as pd
import os

def main():
    print("\n" + "="*95)
    print("FINAL PIPELINE: BASELINES, TUNING & EMA+SENSORS")
    print("="*95)
    
    dataset_file = 'Final_Master_Dataset_Imputed.csv'
    
    if not os.path.exists(dataset_file):
        print("Generating Fresh Smart Imputed Dataset (Fixing fMRI bug)...")
        df_imputed = data_loader.generate_master_dataset(apply_imputation=True)
        df_imputed.to_csv(dataset_file, index=False)
    else:
        print(f"Loading existing {dataset_file}...")
        print("⚠️ NOTE: If you haven't deleted this file since the fMRI bug fix, stop this script, delete the CSV, and run again!")
        df_imputed = pd.read_csv(dataset_file)
        
    print("\n" + "="*95)
    print("EXPERIMENT 1: REAL-TIME DETECTION (Current Window Data)")
    print("="*95)
    
    res_detect = modeling_advanced.run_analysis(df_imputed, window_size=45, task_mode='detection')
    
    print("\nFINAL DETECTION METRICS (Target=Target_Now)")
    print(res_detect[['Modality', 'N_Rows', 'Accuracy', 'AUC_ROC', 'Recall', 'Precision', 'F1']]
          .sort_values(by='AUC_ROC', ascending=False)
          .to_string(index=False))

    print("\n" + "="*95)
    print("EXPERIMENT 2: STANDARDIZED FORECASTING (Predicting Target_Now using exactly 90-Min Old Data)")
    print("="*95)
    
    res_forecast = modeling_advanced.run_analysis(df_imputed, window_size=45, task_mode='forecasting')
    
    print("\nFINAL FORECASTING METRICS (Target=Target_Now)")
    print(res_forecast[['Modality', 'N_Rows', 'Accuracy', 'AUC_ROC', 'Recall', 'Precision', 'F1']]
          .sort_values(by='AUC_ROC', ascending=False)
          .to_string(index=False))

if __name__ == "__main__":
    main()