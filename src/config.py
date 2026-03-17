"""
Configurações do projeto de previsão de demanda
"""
import os
from pathlib import Path

# Caminhos
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
# O DATA_DIR específico para cada dataset será definido nos configs de cada um
# mas mantemos aqui por compatibilidade ou padrão
DEMAND_FORECASTING_DIR = DATA_DIR / "demand-forecasting-kernels-only"
PREDICT_FUTURE_SALES_DIR = DATA_DIR / "competitive-data-science-predict-future-sales"

RESULTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models_saved" # Alterado para não conflitar com src/models

# Criar diretórios se não existirem
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Parâmetros de dados
WINDOW_SIZES = [30, 90, 180, 365]  # Janelas de 1 mês, 3 meses, 6 meses e 1 ano
WINDOW_SIZE = 30  # Janela padrão
FORECAST_HORIZON = 7  # Prever os próximos 7 dias
VALIDATION_SPLIT = 0.2  # 20% dos dados para validação

# Parâmetros de treinamento
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15

# Definições de Arquiteturas para Teste
ARCHITECTURES = {
    'small': {
        'hidden_size': 64,
        'num_layers': 1,
        'dropout': 0.1
    },
    'medium': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2
    },
    'large': {
        'hidden_size': 256,
        'num_layers': 3,
        'dropout': 0.3
    },
    'extra_large': {
        'hidden_size': 256,
        'num_layers': 4,
        'dropout': 0.3
    },
    'huge': {
        'hidden_size': 256,
        'num_layers': 5,
        'dropout': 0.4
    }
}

# MLflow
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")
EXPERIMENT_NAME = "LSTM_vs_GRU_Demand_Forecasting"

# Device
DEVICE = "cuda"  # Usar GPU se disponível, senão CPU

# Random seed para reprodutibilidade
RANDOM_SEED = 42
