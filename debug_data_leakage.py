#!/usr/bin/env python3
"""
Debug script to analyze potential data leakage issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load the climate data"""
    batch_dir = Path('outputs') / 'climate_batch_import'
    parquet_files = list(batch_dir.glob('*.parquet'))
    
    if parquet_files:
        print(f"Found {len(parquet_files)} parquet files")
        dfs = []
        for file in parquet_files:
            df = pd.read_parquet(file)
            dfs.append(df)
        data = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(data)} records from parquet files")
        return data
    return None

def analyze_feature_distributions(data):
    """Analyze the distribution of features"""
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    feature_cols = ['frequency', 'maximum_duration', 'minimum_duration', 'total_duration', 
                   'average_duration', 'peak_severity', 'average_severity']
    
    for col in feature_cols:
        if col in data.columns:
            print(f"\n{col}:")
            print(f"  Mean: {data[col].mean():.6f}")
            print(f"  Std: {data[col].std():.6f}")
            print(f"  Min: {data[col].min():.6f}")
            print(f"  Max: {data[col].max():.6f}")
            print(f"  Unique values: {data[col].nunique()}")
            print(f"  85th percentile: {np.percentile(data[col], 85):.6f}")
            print(f"  90th percentile: {np.percentile(data[col], 90):.6f}")
            
            # Check if values are suspiciously regular
            if data[col].nunique() < 100:
                print(f"  WARNING: Only {data[col].nunique()} unique values - might be synthetic!")

def test_label_creation_consistency(data):
    """Test if label creation is consistent across different splits"""
    print("\n" + "="*60)
    print("LABEL CREATION CONSISTENCY TEST")
    print("="*60)
    
    feature_cols = ['frequency', 'maximum_duration', 'minimum_duration', 'total_duration', 
                   'average_duration', 'peak_severity', 'average_severity']
    X = data[feature_cols]
    
    # Test multiple random splits
    results = []
    for seed in [42, 123, 456, 789, 999]:
        train_indices, test_indices = train_test_split(
            data.index, test_size=0.3, random_state=seed
        )
        
        # Calculate thresholds from training data
        train_data = data.loc[train_indices]
        freq_threshold = np.percentile(train_data['frequency'], 85)
        duration_threshold = np.percentile(train_data['maximum_duration'], 85)
        severity_threshold = np.percentile(train_data['peak_severity'], 85)
        
        # Create labels
        extreme_events = (
            (data['frequency'] > freq_threshold) |
            (data['maximum_duration'] > duration_threshold) |
            (data['peak_severity'] > severity_threshold)
        ).astype(int)
        
        y_train = extreme_events.loc[train_indices]
        y_test = extreme_events.loc[test_indices]
        
        # Train Random Forest
        X_train = X.loc[train_indices]
        X_test = X.loc[test_indices]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_train_scaled, y_train)
        y_pred = rf.predict(X_test_scaled)
        
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            'seed': seed,
            'train_extreme_rate': y_train.mean(),
            'test_extreme_rate': y_test.mean(),
            'recall': recall,
            'precision': precision,
            'accuracy': accuracy,
            'freq_threshold': freq_threshold,
            'duration_threshold': duration_threshold,
            'severity_threshold': severity_threshold
        })
        
        print(f"\nSeed {seed}:")
        print(f"  Thresholds: freq={freq_threshold:.3f}, dur={duration_threshold:.3f}, sev={severity_threshold:.3f}")
        print(f"  Train extreme rate: {y_train.mean():.3f}")
        print(f"  Test extreme rate: {y_test.mean():.3f}")
        print(f"  RF Performance: Recall={recall:.3f}, Precision={precision:.3f}, Accuracy={accuracy:.3f}")
    
    return results

def analyze_feature_label_correlation(data):
    """Analyze correlation between features and labels"""
    print("\n" + "="*60)
    print("FEATURE-LABEL CORRELATION ANALYSIS")
    print("="*60)
    
    # Create labels using the same logic as in the main script
    train_indices, test_indices = train_test_split(
        data.index, test_size=0.3, random_state=42
    )
    
    train_data = data.loc[train_indices]
    freq_threshold = np.percentile(train_data['frequency'], 85)
    duration_threshold = np.percentile(train_data['maximum_duration'], 85)
    severity_threshold = np.percentile(train_data['peak_severity'], 85)
    
    extreme_events = (
        (data['frequency'] > freq_threshold) |
        (data['maximum_duration'] > duration_threshold) |
        (data['peak_severity'] > severity_threshold)
    ).astype(int)
    
    # Analyze correlations
    feature_cols = ['frequency', 'maximum_duration', 'minimum_duration', 'total_duration', 
                   'average_duration', 'peak_severity', 'average_severity']
    
    print("\nCorrelation between features and extreme_event label:")
    for col in feature_cols:
        if col in data.columns:
            corr = np.corrcoef(data[col], extreme_events)[0, 1]
            print(f"  {col}: {corr:.6f}")
    
    # Check if any feature perfectly predicts the label
    test_data = data.loc[test_indices]
    y_test = extreme_events.loc[test_indices]
    
    print("\nTesting if individual features can perfectly predict labels on test set:")
    for col in ['frequency', 'maximum_duration', 'peak_severity']:
        if col in data.columns:
            if col == 'frequency':
                threshold = freq_threshold
            elif col == 'maximum_duration':
                threshold = duration_threshold
            else:  # peak_severity
                threshold = severity_threshold
            
            pred = (test_data[col] > threshold).astype(int)
            # Check how many test samples this single feature rule captures
            overlap = (pred & y_test).sum()
            total_extreme = y_test.sum()
            print(f"  {col} > {threshold:.3f}: captures {overlap}/{total_extreme} extreme events ({overlap/total_extreme*100:.1f}%)")

def main():
    print("Data Leakage Debug Analysis")
    print("="*50)
    
    # Load data
    data = load_data()
    if data is None:
        print("No data found!")
        return
    
    # Analyze feature distributions
    analyze_feature_distributions(data)
    
    # Test label creation consistency
    results = test_label_creation_consistency(data)
    
    # Analyze feature-label correlations
    analyze_feature_label_correlation(data)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Average recall across different seeds: {np.mean([r['recall'] for r in results]):.3f} ± {np.std([r['recall'] for r in results]):.3f}")
    print(f"Average precision across different seeds: {np.mean([r['precision'] for r in results]):.3f} ± {np.std([r['precision'] for r in results]):.3f}")
    
    if all(r['recall'] > 0.99 for r in results):
        print("\n🚨 WARNING: Random Forest consistently achieves near-perfect recall!")
        print("This suggests there might still be data leakage or the problem is too easy.")
    else:
        print("\n✅ Random Forest performance varies across seeds, suggesting proper evaluation.")

if __name__ == "__main__":
    main()