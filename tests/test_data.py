import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent.parent))

import pytest
import pandas as pd
from src.data.data_loader import DataLoader
from src.data.data_validation import DataValidator

def test_data_loader():
    """testing data loading functionality"""
    loader = DataLoader()
    df = loader.load_sample_data()

    assert len(df) >0
    assert 'price' in df.columns
    assert df.isnull().sum() == 0

def test_data_validator():
    """test data validator"""
    validator = DataValidator()
    loader = DataLoader()
    df = loader.load_sample_data()

    expected_cols = ['square_feet', 'bedrooms', 'bathrooms', 'age', 'location_score', 'price']
    assert validator.validate_schema(df, expected_cols)
    results = validator.validate_data_quality(df)
    assert all(results.values())


