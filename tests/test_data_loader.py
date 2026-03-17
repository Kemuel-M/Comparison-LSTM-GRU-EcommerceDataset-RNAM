import numpy as np
import torch
import pytest
import sys
from pathlib import Path

# Add src to path if not already there
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_loader import DemandDataset, create_sequences, get_data_loaders

def test_create_sequences():
    data = np.arange(20).reshape(-1, 1)
    window_size = 5
    forecast_horizon = 3
    
    # data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    # Expected number of sequences: len(data) - window_size - forecast_horizon + 1
    # 20 - 5 - 3 + 1 = 13
    
    X, y = create_sequences(data, window_size, forecast_horizon)
    
    assert X.shape == (13, 5, 1)
    assert y.shape == (13, 3, 1)
    
    # Check first sequence
    np.testing.assert_array_equal(X[0], data[0:5])
    np.testing.assert_array_equal(y[0], data[5:8])
    
    # Check last sequence
    # 13th sequence starts at index 12 (12:17 for X, 17:20 for y)
    np.testing.assert_array_equal(X[-1], data[12:17])
    np.testing.assert_array_equal(y[-1], data[17:20])

def test_demand_dataset():
    X = np.random.randn(10, 5, 1)
    y = np.random.randn(10, 3)
    
    dataset = DemandDataset(X, y)
    
    assert len(dataset) == 10
    
    X_sample, y_sample = dataset[0]
    
    assert isinstance(X_sample, torch.FloatTensor)
    assert isinstance(y_sample, torch.FloatTensor)
    assert X_sample.shape == (5, 1)
    assert y_sample.shape == (3,)

def test_get_data_loaders():
    X_train = np.random.randn(50, 5, 1)
    y_train = np.random.randn(50, 3)
    X_val = np.random.randn(10, 5, 1)
    y_val = np.random.randn(10, 3)
    batch_size = 8
    
    train_loader, val_loader = get_data_loaders(X_train, y_train, X_val, y_val, batch_size)
    
    assert len(train_loader) == int(np.ceil(50 / batch_size))
    assert len(val_loader) == int(np.ceil(10 / batch_size))
    
    # Check first batch
    X_batch, y_batch = next(iter(train_loader))
    assert X_batch.shape == (batch_size, 5, 1)
    assert y_batch.shape == (batch_size, 3)
