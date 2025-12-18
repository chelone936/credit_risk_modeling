"""
Feature Engineering Pipeline for Credit Risk Modeling

This module provides feature engineering transformations using sklearn pipelines.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


class AggregateFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Creates customer-level aggregate features from transaction data.
    
    Features created:
    - Total_Transaction_Amount: Sum of amounts per customer
    - Average_Transaction_Amount: Mean amount per customer
    - Transaction_Count: Number of transactions per customer
    - Std_Transaction_Amount: Std deviation of amounts per customer
    """
    
    def fit(self, X, y=None):
        """Compute aggregate statistics per customer."""
        self.customer_aggregates_ = X.groupby('CustomerId')['Amount'].agg([
            ('Total_Transaction_Amount', 'sum'),
            ('Average_Transaction_Amount', 'mean'),
            ('Transaction_Count', 'count'),
            ('Std_Transaction_Amount', 'std')
        ]).reset_index()
        
        # Fill NaN std (customers with 1 transaction) with 0
        self.customer_aggregates_['Std_Transaction_Amount'].fillna(0, inplace=True)
        return self
    
    def transform(self, X):
        """Add aggregate features to dataframe."""
        return X.merge(self.customer_aggregates_, on='CustomerId', how='left')


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts temporal features from TransactionStartTime.
    
    Features created:
    - Transaction_Hour, Transaction_Day, Transaction_Month, Transaction_Year
    - Transaction_DayOfWeek, Transaction_IsWeekend
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract temporal features."""
        X_copy = X.copy()
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(X_copy['TransactionStartTime']):
            X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])
        
        # Extract components
        X_copy['Transaction_Hour'] = X_copy['TransactionStartTime'].dt.hour
        X_copy['Transaction_Day'] = X_copy['TransactionStartTime'].dt.day
        X_copy['Transaction_Month'] = X_copy['TransactionStartTime'].dt.month
        X_copy['Transaction_Year'] = X_copy['TransactionStartTime'].dt.year
        X_copy['Transaction_DayOfWeek'] = X_copy['TransactionStartTime'].dt.dayofweek
        X_copy['Transaction_IsWeekend'] = (X_copy['Transaction_DayOfWeek'] >= 5).astype(int)
        
        return X_copy


def build_feature_pipeline():
    """
    Build the complete feature engineering pipeline.
    
    Returns:
    --------
    Pipeline with aggregate and temporal transformers
    """
    return Pipeline([
        ('aggregates', AggregateFeatureTransformer()),
        ('temporal', TemporalFeatureExtractor())
    ])


def get_preprocessing_pipeline(numerical_cols, categorical_cols):
    """
    Build preprocessing pipeline for encoding and scaling.
    
    Parameters:
    -----------
    numerical_cols : list
        List of numerical column names
    categorical_cols : list
        List of categorical column names
        
    Returns:
    --------
    ColumnTransformer for preprocessing
    """
    transformers = []
    
    # Numerical pipeline: impute then scale
    if numerical_cols:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', num_pipeline, numerical_cols))
    
    # Categorical pipeline: impute then one-hot encode
    if categorical_cols:
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipeline, categorical_cols))
    
    return ColumnTransformer(transformers=transformers, remainder='drop')


def load_and_process_data(input_path, output_path=None):
    """
    Load raw data, apply feature engineering, and optionally save.
    
    Parameters:
    -----------
    input_path : str
        Path to raw CSV file
    output_path : str, optional
        Path to save processed data
        
    Returns:
    --------
    tuple: (processed_dataframe, feature_pipeline, preprocessing_pipeline)
    """
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Apply custom feature engineering
    print("\nApplying feature engineering...")
    feature_pipeline = build_feature_pipeline()
    df_features = feature_pipeline.fit_transform(df)
    print(f"After feature engineering: {len(df_features.columns)} columns")
    
    # Define column groups for preprocessing
    numerical_cols = [
        'Amount', 'Value',
        'Total_Transaction_Amount', 'Average_Transaction_Amount',
        'Transaction_Count', 'Std_Transaction_Amount',
        'Transaction_Hour', 'Transaction_Day', 'Transaction_Month',
        'Transaction_Year', 'Transaction_DayOfWeek', 'Transaction_IsWeekend'
    ]
    
    categorical_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']
    
    # Filter to existing columns
    numerical_cols = [col for col in numerical_cols if col in df_features.columns]
    categorical_cols = [col for col in categorical_cols if col in df_features.columns]
    
    # Apply preprocessing
    print("\nApplying preprocessing (imputation, encoding, scaling)...")
    preprocessing_pipeline = get_preprocessing_pipeline(numerical_cols, categorical_cols)
    df_processed = preprocessing_pipeline.fit_transform(df_features)
    
    # Get feature names
    feature_names = []
    for name, transformer, columns in preprocessing_pipeline.transformers_:
        if name == 'num':
            feature_names.extend(columns)
        elif name == 'cat':
            encoder = transformer.named_steps['encoder']
            cat_features = encoder.get_feature_names_out(columns)
            feature_names.extend(cat_features)
    
    # Convert to DataFrame
    df_final = pd.DataFrame(df_processed, columns=feature_names)
    print(f"Final shape: {df_final.shape}")
    
    # Save if output path provided
    if output_path:
        df_final.to_csv(output_path, index=False)
        print(f"\nSaved processed data to {output_path}")
    
    return df_final, feature_pipeline, preprocessing_pipeline


if __name__ == "__main__":
    # Example usage
    processed_data, feat_pipe, prep_pipe = load_and_process_data(
        "data/raw/data.csv",
        "data/processed/train_processed.csv"
    )
    
    print("\n" + "="*60)
    print("SAMPLE OF PROCESSED DATA")
    print("="*60)
    print(processed_data.head())
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(processed_data.describe())
