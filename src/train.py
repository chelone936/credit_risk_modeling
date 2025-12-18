"""
Model Training and Tracking for Credit Risk Modeling

This module handles:
- Data splitting (Train/Test)
- Training multiple models (Logistic Regression, Random Forest)
- Hyperparameter tuning using RandomizedSearchCV
- Experiment tracking with MLflow
- Model evaluation and logging
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)
import os

def prepare_data(filepath, target_col='is_high_risk', test_size=0.2, random_state=42):
    """
    Load processed data and split into training and testing sets.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
        
    df = pd.read_csv(filepath)
    
    # Drop CustomerId if it exists as it's an ID, not a feature
    X = df.drop(columns=[target_col, 'CustomerId'], errors='ignore')
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """
    Calculate evaluation metrics for the model.
    """
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_probs)
    }
    
    return metrics

def train_and_track():
    """
    Execute the full training and tracking pipeline.
    """
    data_path = "data/processed/final_train_data.csv"
    
    # MLflow Setup
    mlflow.set_experiment("Credit_Risk_Modeling")
    
    print("="*60)
    print("STEP 1: Data Preparation")
    print("="*60)
    try:
        X_train, X_test, y_train, y_test = prepare_data(data_path)
        print(f"Train/Test split completed. Features: {X_train.shape[1]}")
    except Exception as e:
        print(f"Error loading data: {e}. Ensure Task 4 was run.")
        return

    # --- MODEL 1: Logistic Regression (Baseline) ---
    print("\n" + "="*60)
    print("STEP 2: Training Logistic Regression (Baseline)")
    print("="*60)
    with mlflow.start_run(run_name="Logistic_Regression_Baseline"):
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        
        metrics = evaluate_model(lr, X_test, y_test)
        
        # Log to MLflow
        mlflow.log_params({"model_type": "LogisticRegression", "max_iter": 1000})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(lr, "model")
        
        print("Logistic Regression Metrics:")
        for k, v in metrics.items():
            print(f"- {k}: {v:.4f}")

    # --- MODEL 2: Random Forest (Optimized) ---
    print("\n" + "="*60)
    print("STEP 3: Training Random Forest (w/ Hyperparameter Tuning)")
    print("="*60)
    with mlflow.start_run(run_name="Random_Forest_Optimized"):
        rf = RandomForestClassifier(random_state=42)
        
        param_dist = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        # Search for best params
        search = RandomizedSearchCV(
            rf, param_distributions=param_dist, n_iter=4, 
            cv=3, random_state=42, n_jobs=-1, scoring='roc_auc'
        )
        search.fit(X_train, y_train)
        
        best_rf = search.best_estimator_
        metrics = evaluate_model(best_rf, X_test, y_test)
        
        # Log to MLflow
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_rf, "model", registered_model_name="CreditRiskRandomForest")
        
        print("Best Random Forest Params:", search.best_params_)
        print("Random Forest Metrics:")
        for k, v in metrics.items():
            print(f"- {k}: {v:.4f}")

if __name__ == "__main__":
    train_and_track()
