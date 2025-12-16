import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pytest
import joblib
import numpy as np

def test_model_exists():
    """test if model is in directory"""
    assert Path("models/model.pkl").exists()
    assert Path("models/scaler.pkl").exists()

def test_model_prediction():
    """test model can make predictions"""
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')

    # create sample input
    sample_input = np.array([[1500,2,3,10,7.5]])
    sample_scaled = scaler.transform(sample_input)

    prediction = model.predict(sample_input)

    assert len(prediction) == 1
    assert prediction[0] > 0

    