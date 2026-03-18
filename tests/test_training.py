import pytest
import sys
from pathlib import Path

# Add src to path if not already there
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.training.train import EarlyStopping

def test_early_stopping_initialization():
    patience = 5
    min_delta = 0.01
    
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    
    assert early_stopping.patience == patience
    assert early_stopping.min_delta == min_delta
    assert early_stopping.counter == 0
    assert early_stopping.best_loss is None
    assert early_stopping.early_stop is False

def test_early_stopping_improvement():
    early_stopping = EarlyStopping(patience=3)
    
    early_stopping(0.5)
    assert early_stopping.best_loss == 0.5
    assert early_stopping.counter == 0
    
    early_stopping(0.4)
    assert early_stopping.best_loss == 0.4
    assert early_stopping.counter == 0
    
    early_stopping(0.3)
    assert early_stopping.best_loss == 0.3
    assert early_stopping.counter == 0

def test_early_stopping_no_improvement():
    early_stopping = EarlyStopping(patience=3, min_delta=0.0)
    
    early_stopping(0.5) # best_loss = 0.5
    
    early_stopping(0.51)
    assert early_stopping.counter == 1
    assert early_stopping.early_stop is False
    
    early_stopping(0.52)
    assert early_stopping.counter == 2
    assert early_stopping.early_stop is False
    
    early_stopping(0.53)
    assert early_stopping.counter == 3
    assert early_stopping.early_stop is True

def test_early_stopping_reset_after_improvement():
    early_stopping = EarlyStopping(patience=3, min_delta=0.0)
    
    early_stopping(0.5)
    early_stopping(0.51)
    assert early_stopping.counter == 1
    
    early_stopping(0.49)
    assert early_stopping.counter == 0
    assert early_stopping.best_loss == 0.49

def test_early_stopping_min_delta():
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)
    
    early_stopping(0.5) # best_loss = 0.5
    
    # 0.495 is NOT improvement if delta=0.01 (needs 0.5 - 0.01 = 0.49)
    early_stopping(0.495)
    assert early_stopping.counter == 1
    
    early_stopping(0.485) # Is improvement (0.485 < 0.5 - 0.01)
    assert early_stopping.counter == 0
    assert early_stopping.best_loss == 0.485
