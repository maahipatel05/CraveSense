import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import resample, compute_class_weight

# --- MODEL DEFINITIONS ---
def get_models(random_state=42):
    return {
        'LogReg': LogisticRegression(max_iter=1000, random_state=random_state),
        'RandForest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'GradBoost': GradientBoostingClassifier(random_state=random_state),
        'SVM': SVC(kernel='rbf', probability=True, random_state=random_state)
    }

# --- EXPERIMENT ENGINE ---
def run_experiment(target_name, df_data):
    print(f"\n>>> STARTING EXPERIMENT FOR TARGET: {target_name} <<<")
    results = []
    feature_cols = ['Stress_Norm', 'Mood_Norm', 'Trauma', 'Hour']
    
    # Filter NaNs
    df_target = df_data.dropna(subset=[target_name] + feature_cols)
    X = df_target[feature_cols]
    y = df_target[target_name]
    
    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    models = get_models()
    
    # --- STRATEGY 1: BASELINE ---
    for name, model in models.items():
        m = get_models()[name]
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        results.append({
            'Target': target_name, 'Strategy': 'Baseline', 'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'Precision': precision_score(y_test, y_pred, zero_division=0)
        })

    # --- STRATEGY 2: OVERSAMPLING ---
    train_data = pd.concat([X_train, y_train], axis=1)
    majority = train_data[train_data[target_name] == 0]
    minority = train_data[train_data[target_name] == 1]
    
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    upsampled_data = pd.concat([majority, minority_upsampled])
    X_train_up = upsampled_data[feature_cols]
    y_train_up = upsampled_data[target_name]
    
    for name, model in models.items():
        m = get_models()[name]
        m.fit(X_train_up, y_train_up)
        y_pred = m.predict(X_test)
        results.append({
            'Target': target_name, 'Strategy': 'Oversampling', 'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'Precision': precision_score(y_test, y_pred, zero_division=0)
        })

    # --- STRATEGY 3: CLASS WEIGHTS ---
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    weight_dict = dict(zip(classes, weights))
    sample_weights = y_train.map(weight_dict)
    
    for name, model in models.items():
        m = get_models()[name]
        if name in ['LogReg', 'RandForest', 'SVM']:
            m.set_params(class_weight='balanced')
            m.fit(X_train, y_train)
        elif name == 'GradBoost':
            m.fit(X_train, y_train, sample_weight=sample_weights)
            
        y_pred = m.predict(X_test)
        results.append({
            'Target': target_name, 'Strategy': 'Weighted', 'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'Precision': precision_score(y_test, y_pred, zero_division=0)
        })
        
    return pd.DataFrame(results)

# --- VISUALIZATION FUNCTIONS ---

def plot_champion_comparison(results_df):
    best_idx = results_df.groupby('Target')['F1'].idxmax()
    champions = results_df.loc[best_idx]
    melted = champions.melt(id_vars=['Target', 'Strategy', 'Model'], 
                            value_vars=['F1', 'Recall', 'Precision'], 
                            var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x='Target', y='Score', hue='Metric', palette='viridis')
    plt.title("Champion Showdown: Craving vs. Proxy (Negative Emotion)", fontsize=14)
    plt.ylim(0, 1.1)
    plt.ylabel("Score (0-1)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Metric')
    
    for p in plt.gca().patches:
        if p.get_height() > 0:
            plt.gca().annotate(f'{p.get_height():.2f}', 
                               (p.get_x() + p.get_width() / 2., p.get_height()), 
                               ha = 'center', va = 'center', xytext = (0, 8), 
                               textcoords = 'offset points', fontsize=9)
    plt.tight_layout()
    plt.show()

def plot_scenario_metrics(results_df):
    long_df = results_df.melt(id_vars=['Target', 'Strategy', 'Model'], 
                              value_vars=['F1', 'Recall', 'Precision'], 
                              var_name='Metric', value_name='Score')
    g = sns.catplot(data=long_df, x='Strategy', y='Score', hue='Model', 
                    col='Metric', row='Target', kind='bar', 
                    height=4, aspect=1.2, palette='magma', sharey=True)
    g.fig.suptitle("Detailed Performance: F1, Recall, & Precision across all Scenarios", y=1.02, fontsize=16)
    g.set_axis_labels("Strategy", "Score")
    for ax in g.axes.flat:
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_ylim(0, 1.1)
    plt.show()

def plot_detailed_eda(df):
    """
    Generates 3 key plots: Phase Distribution, Time-of-Day Analysis, and Subject Variability.
    """
    print("Generating Detailed EDA Plots...")
    
    # 1. THE PHASE PROBLEM (Why we normalize)
    # We use the ORIGINAL un-normalized columns if available, or reconstruct for visualization
    # Ideally, preprocessing should keep raw columns, but we can imply from logic or use Norm columns
    # Let's visualize the NORMALIZED distribution to show it's now consistent, 
    # OR separate by Phase to show if they look different.
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(data=df, x='Stress_Norm', hue='Data Collection (Phase)', kde=True, element="step", ax=axes[0])
    axes[0].set_title('Normalized Stress Distribution by Phase')
    
    sns.histplot(data=df, x='Mood_Norm', hue='Data Collection (Phase)', kde=True, element="step", ax=axes[1])
    axes[1].set_title('Normalized Mood Distribution by Phase')
    
    plt.suptitle("Check: Is Data Consistent Across Phases After Normalization?")
    plt.tight_layout()
    plt.show()

    # 2. TIME OF DAY ANALYSIS (When do cravings happen?)
    # Create a pivot table of Craving Rate by Hour
    hourly_crave = df.groupby('Hour')['Craving_Binary'].mean().reset_index()
    
    plt.figure(figsize=(10, 5))
    
    sns.lineplot(data=hourly_crave, x='Hour', y='Craving_Binary', marker='o', color='crimson', linewidth=2)
    plt.title("Circadian Rhythm of Cravings: Probability by Hour of Day")
    plt.ylabel("Probability of Craving")
    plt.xlabel("Hour (0-23)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(range(0, 24))
    plt.show()

    # 3. SUBJECT VARIABILITY (Individual Differences)
    # Filter for top 20 users by data volume to keep plot readable
    top_users = df['User_ID'].value_counts().nlargest(20).index
    df_top = df[df['User_ID'].isin(top_users)]
    
    plt.figure(figsize=(14, 6))
    
    sns.boxplot(data=df_top, x='User_ID', y='Stress_Norm', palette='coolwarm')
    plt.title("Subject Variability: Stress Levels for Top 20 Users")
    plt.xticks(rotation=45)
    plt.ylabel("Normalized Stress")
    plt.tight_layout()
    plt.show()

    # 4. Correlation Heatmap (Existing)
    cols_eda = ['Stress_Norm', 'Mood_Norm', 'Trauma', 'Hour', 'Negative', 'Craving_Binary']
    corr_mat = df[cols_eda].corr()
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(corr_mat[['Negative', 'Craving_Binary']].sort_values(by='Craving_Binary', ascending=False), 
                annot=True, cmap='coolwarm', center=0)
    plt.title("Feature Correlation: Proxy (Negative) vs. Craving")
    plt.tight_layout()
    plt.show()