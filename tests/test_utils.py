import numpy as np
import torch
import pytest
import sys
from pathlib import Path

# Add src to path if not already there
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.utils import set_seed, calculate_metrics, count_parameters, get_device
from src.models.models import LSTMModel

def test_set_seed():
    seed = 42
    set_seed(seed)
    
    a1 = np.random.rand(5)
    t1 = torch.rand(5)
    
    set_seed(seed)
    a2 = np.random.rand(5)
    t2 = torch.rand(5)
    
    np.testing.assert_array_equal(a1, a2)
    torch.testing.assert_close(t1, t2)

def test_calculate_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    # RMSE = sqrt(mean((0.1^2 + (-0.1)^2 + 0.2^2)/3)) = sqrt((0.01 + 0.01 + 0.04)/3) = sqrt(0.06/3) = sqrt(0.02)
    expected_rmse = np.sqrt(0.02)
    # MAE = (0.1 + 0.1 + 0.2)/3 = 0.4/3
    expected_mae = 0.4/3
    
    assert pytest.approx(metrics['RMSE'], rel=1e-5) == expected_rmse
    assert pytest.approx(metrics['MAE'], rel=1e-5) == expected_mae

def test_count_parameters():
    model = LSTMModel(input_size=1, hidden_size=10, num_layers=1, forecast_horizon=1)
    
    # LSTM layers: 
    # input_size=1, hidden_size=10
    # Weights: 4 * (10 * 1 + 10 * 10) = 4 * 110 = 440
    # Bias: 4 * (10 + 10) = 80
    # Total LSTM = 520
    
    # FC Sequential:
    # Linear(10, 5): 10*5 + 5 = 55
    # Linear(5, 1): 5*1 + 1 = 6
    # Total FC = 61
    
    # Total = 520 + 61 = 581
    
    params = count_parameters(model)
    assert params == 581

def test_get_device():
    device = get_device()
    assert isinstance(device, torch.device)
    # Should be 'cpu' unless cuda is available and config allows it
