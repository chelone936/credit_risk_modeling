"""
Unit tests for feature engineering pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import (
    AggregateFeatureTransformer,
    TemporalFeatureExtractor,
    build_feature_pipeline
)


def test_aggregate_features_transformer():
    """Test that aggregate features are created correctly."""
    data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C1', 'C2', 'C2'],
        'Amount': [100, 200, 300, 500, 1000]
    })
    
    transformer = AggregateFeatureTransformer()
    result = transformer.fit_transform(data)
    
    # Check new columns exist
    assert 'Total_Transaction_Amount' in result.columns
    assert 'Average_Transaction_Amount' in result.columns
    assert 'Transaction_Count' in result.columns
    assert 'Std_Transaction_Amount' in result.columns
    
    # Check C1 values
    c1_row = result[result['CustomerId'] == 'C1'].iloc[0]
    assert c1_row['Total_Transaction_Amount'] == 600
    assert c1_row['Average_Transaction_Amount'] == 200
    assert c1_row['Transaction_Count'] == 3


def test_temporal_features_extractor():
    """Test that temporal features are extracted correctly."""
    data = pd.DataFrame({
        'TransactionStartTime': [
            datetime(2018, 11, 15, 14, 30, 0),  # Thursday
            datetime(2018, 12, 25, 9, 15, 0),   # Tuesday
            datetime(2019, 1, 6, 22, 45, 0)     # Sunday
        ]
    })
    
    extractor = TemporalFeatureExtractor()
    result = extractor.fit_transform(data)
    
    # Check columns exist
    assert 'Transaction_Hour' in result.columns
    assert 'Transaction_Day' in result.columns
    assert 'Transaction_Month' in result.columns
    assert 'Transaction_Year' in result.columns
    assert 'Transaction_DayOfWeek' in result.columns
    assert 'Transaction_IsWeekend' in result.columns
    
    # Check first row values
    assert result['Transaction_Hour'].iloc[0] == 14
    assert result['Transaction_Month'].iloc[0] == 11
    assert result['Transaction_Year'].iloc[0] == 2018
    assert result['Transaction_IsWeekend'].iloc[0] == 0
    
    # Check Sunday is weekend
    assert result['Transaction_IsWeekend'].iloc[2] == 1


def test_pipeline_output_shape():
    """Test that pipeline produces expected output."""
    data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [100, 200, 300],
        'TransactionStartTime': [
            datetime(2018, 11, 15, 10, 0),
            datetime(2018, 11, 15, 14, 0),
            datetime(2018, 11, 16, 9, 0)
        ]
    })
    
    pipeline = build_feature_pipeline()
    result = pipeline.fit_transform(data)
    
    # Should have more columns than input
    assert result.shape[1] > data.shape[1]
    # Should have same number of rows
    assert result.shape[0] == data.shape[0]


def test_pipeline_reproducibility():
    """Test that pipeline produces consistent results."""
    data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [100, 200, 300],
        'TransactionStartTime': [datetime(2018, 11, 15)] * 3
    })
    
    pipeline1 = build_feature_pipeline()
    result1 = pipeline1.fit_transform(data)
    
    pipeline2 = build_feature_pipeline()
    result2 = pipeline2.fit_transform(data)
    
    pd.testing.assert_frame_equal(result1, result2)


def test_no_data_leakage():
    """Test that fit and transform are properly separated."""
    train_data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [100, 200, 300],
        'TransactionStartTime': [datetime(2018, 11, 15)] * 3
    })
    
    test_data = pd.DataFrame({
        'CustomerId': ['C1', 'C3'],
        'Amount': [150, 250],
        'TransactionStartTime': [datetime(2018, 11, 16)] * 2
    })
    
    pipeline = build_feature_pipeline()
    pipeline.fit(train_data)
    result = pipeline.transform(test_data)
    
    assert result.shape[0] == test_data.shape[0]


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
