"""
Proxy Target Variable Engineering for Credit Risk Modeling

This module calculates RFM metrics, performs K-Means clustering, 
and assigns a proxy 'is_high_risk' label to customers.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class TargetEngineer:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.rfm_data = None
        self.high_risk_cluster = None

    def calculate_rfm(self, df):
        """
        Calculate Recency, Frequency, and Monetary metrics per CustomerId.
        """
        # Ensure TransactionStartTime is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['TransactionStartTime']):
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

        # Use the day after the last transaction as the snapshot date
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

        rfm = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Amount': 'sum'
        }).reset_index()

        rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
        self.rfm_data = rfm
        return rfm

    def cluster_customers(self, rfm_df):
        """
        Cluster customers based on RFM profiles using K-Means.
        """
        features = ['Recency', 'Frequency', 'Monetary']
        scaled_features = self.scaler.fit_transform(rfm_df[features])
        
        # Perform clustering
        rfm_df['Cluster'] = self.kmeans.fit_predict(scaled_features)
        
        return rfm_df

    def identify_high_risk_cluster(self, rfm_df):
        """
        Identify the high-risk cluster (lowest engagement).
        High risk typically has Low Frequency and Low Monetary.
        """
        cluster_stats = rfm_df.groupby('Cluster').agg({
            'Frequency': 'mean',
            'Monetary': 'mean',
            'Recency': 'mean'
        })
        
        # High risk is defined as the cluster with relative low frequency and low monetary
        # We can use a simple heuristic: the cluster with the lowest (Frequency + normalized Monetary)
        # or just the one with the lowest Frequency.
        self.high_risk_cluster = cluster_stats['Frequency'].idxmin()
        return self.high_risk_cluster

    def assign_labels(self, rfm_df):
        """
        Create the 'is_high_risk' target variable.
        """
        if self.high_risk_cluster is None:
            self.identify_high_risk_cluster(rfm_df)
            
        rfm_df['is_high_risk'] = (rfm_df['Cluster'] == self.high_risk_cluster).astype(int)
        return rfm_df

def run_target_engineering(raw_data_path, processed_data_path, output_path):
    """
    Main execution function for Task 4.
    """
    # Import here to avoid circular imports if any
    from src.data_processing import load_and_process_data, apply_woe_iv
    
    print("="*60)
    print("STEP 1: Load and Process Raw Data (Task 3)")
    print("="*60)
    processed_features, _, _ = load_and_process_data(raw_data_path, keep_ids=True)
    
    print("\n" + "="*60)
    print("STEP 2: Calculate RFM and Cluster Customers (Task 4)")
    print("="*60)
    raw_df = pd.read_csv(raw_data_path)
    engineer = TargetEngineer()
    rfm_df = engineer.calculate_rfm(raw_df)
    rfm_df = engineer.cluster_customers(rfm_df)
    rfm_df = engineer.assign_labels(rfm_df)
    
    # Show cluster stats
    print("\nCluster Statistics:")
    print(rfm_df.groupby('Cluster').agg({
        'Frequency': 'mean', 
        'Monetary': 'mean', 
        'Recency': 'mean',
        'CustomerId': 'count'
    }))
    print(f"\nIdentified high-risk cluster: {engineer.high_risk_cluster}")

    print("\n" + "="*60)
    print("STEP 3: Merge Target with Engineered Features")
    print("="*60)
    # Merge on CustomerId
    final_df = processed_features.merge(
        rfm_df[['CustomerId', 'is_high_risk']], 
        on='CustomerId', 
        how='left'
    )
    
    # CustomerId is no longer needed for training unless we want to keep it
    # But often we drop it before feeding to ML
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Target distribution (is_high_risk):\n{final_df['is_high_risk'].value_counts(normalize=True)}")

    print("\n" + "="*60)
    print("STEP 4: Apply WoE Transformation (Extension)")
    print("="*60)
    final_df_woe, iv_values = apply_woe_iv(final_df, 'is_high_risk')
    
    if iv_values is not None:
        print("\nTop 5 Features by Information Value (IV):")
        print(iv_values.head(5))

    print("\n" + "="*60)
    print("STEP 5: Save Final Training Data")
    print("="*60)
    final_df.to_csv(output_path, index=False)
    print(f"Saved final training data to: {output_path}")
    
    if iv_values is not None:
        woe_path = output_path.replace('.csv', '_woe.csv')
        final_df_woe.to_csv(woe_path, index=False)
        print(f"Saved WoE transformed data to: {woe_path}")

    return final_df


if __name__ == "__main__":
    # Main entry point for Task 4
    import os
    
    raw_data = "data/raw/data.csv"
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    final_output = os.path.join(output_dir, "final_train_data.csv")
    
    run_target_engineering(raw_data, None, final_output)
