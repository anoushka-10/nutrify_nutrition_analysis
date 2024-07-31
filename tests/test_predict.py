# tests/test_predict.py
import pytest
from src.predict import load_model, make_prediction
import numpy as np

def test_load_model():
    model = load_model('models/model.pkl')
    assert model is not None

def test_make_prediction():
    model = load_model('models/model.pkl')
    data = np.array([[1, 2, 3, 4]])
    predictions = make_prediction(model, data)
    assert predictions is not None
