# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scikit-learn",
#     "xgboost",
#     "pandas",
#     "numpy",
# ]
# ///

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score
import xgboost as xgb
import json
import time

def generate_fraud_data(n_samples=50000, weights=[0.99, 0.01]):
    print(f"Generating synthetic fraud dataset: {n_samples} samples, {weights[1]*100}% fraud rate...")
    X, y = make_classification(n_samples=n_samples, n_features=20, n_informative=10, 
                               n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, 
                               weights=weights, flip_y=0.01, class_sep=0.8, random_state=42)
    feature_names = [f"V{i}" for i in range(1, 21)]
    df = pd.DataFrame(X, columns=feature_names)
    df['Class'] = y
    return df

def main():
    df = generate_fraud_data()
    
    # Feature engineering
    feature_cols = [c for c in df.columns if c != "Class"]
    df['V_mean'] = df[feature_cols].mean(axis=1)
    df['V_std'] = df[feature_cols].std(axis=1)
    df['V_max'] = df[feature_cols].max(axis=1)
    df['V_min'] = df[feature_cols].min(axis=1)

    df.to_csv("fraud_data.csv", index=False)
    
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Save dataset splits for LLM fine-tuning
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv("train_data.csv", index=False)
    test_df.to_csv("test_data.csv", index=False)
    
    # Train XGBoost
    print("Training XGBoost...")
    start_time = time.time()
    
    scale_pos_weight = 7
    
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc'
    )
    
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    infer_time = time.time() - start_time
    
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(classification_report(y_test, y_pred))
    
    results = {
        "model": "XGBoost",
        "train_time_sec": train_time,
        "infer_time_sec_per_10k": (infer_time / len(X_test)) * 10000,
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
    
    with open("xgboost_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"XGBoost Results: AUC={auc:.4f}, F1={f1:.4f}")

if __name__ == "__main__":
    main()
