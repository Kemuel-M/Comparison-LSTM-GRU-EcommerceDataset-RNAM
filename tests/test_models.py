import torch
import pytest
import sys
from pathlib import Path

# Add src to path if not already there
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.models import LSTMModel, GRUModel, BiLSTMModel, BiGRUModel, get_model

def test_lstm_model_initialization():
    input_size = 1
    hidden_size = 64
    num_layers = 2
    dropout = 0.2
    forecast_horizon = 7
    
    model = LSTMModel(input_size, hidden_size, num_layers, dropout, forecast_horizon)
    
    assert model.hidden_size == hidden_size
    assert model.num_layers == num_layers
    assert isinstance(model.lstm, torch.nn.LSTM)
    assert model.lstm.input_size == input_size
    assert model.lstm.hidden_size == hidden_size
    assert model.lstm.num_layers == num_layers

def test_lstm_model_forward():
    batch_size = 8
    seq_len = 30
    input_size = 1
    forecast_horizon = 7
    
    model = LSTMModel(input_size=input_size, forecast_horizon=forecast_horizon)
    x = torch.randn(batch_size, seq_len, input_size)
    
    output = model(x)
    
    assert output.shape == (batch_size, forecast_horizon)

def test_gru_model_forward():
    batch_size = 8
    seq_len = 30
    input_size = 1
    forecast_horizon = 7
    
    model = GRUModel(input_size=input_size, forecast_horizon=forecast_horizon)
    x = torch.randn(batch_size, seq_len, input_size)
    
    output = model(x)
    
    assert output.shape == (batch_size, forecast_horizon)

def test_bilstm_model_forward():
    batch_size = 8
    seq_len = 30
    input_size = 1
    forecast_horizon = 7
    
    model = BiLSTMModel(input_size=input_size, forecast_horizon=forecast_horizon)
    x = torch.randn(batch_size, seq_len, input_size)
    
    output = model(x)
    
    assert output.shape == (batch_size, forecast_horizon)

def test_bigru_model_forward():
    batch_size = 8
    seq_len = 30
    input_size = 1
    forecast_horizon = 7
    
    model = BiGRUModel(input_size=input_size, forecast_horizon=forecast_horizon)
    x = torch.randn(batch_size, seq_len, input_size)
    
    output = model(x)
    
    assert output.shape == (batch_size, forecast_horizon)

def test_get_model_factory():
    lstm = get_model('lstm')
    assert isinstance(lstm, LSTMModel)
    
    gru = get_model('gru')
    assert isinstance(gru, GRUModel)
    
    bilstm = get_model('bilstm')
    assert isinstance(bilstm, BiLSTMModel)
    
    bigru = get_model('bigru')
    assert isinstance(bigru, BiGRUModel)
    
    with pytest.raises(ValueError):
        get_model('invalid_model_type')
