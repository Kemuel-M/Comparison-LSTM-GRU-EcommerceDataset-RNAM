"""
Configurações específicas para cada dataset
"""
from pathlib import Path
from src import config

# Configurações base para datasets
DATASET_CONFIGS = {
    'demand_forecasting': {
        'name': 'Demand Forecasting',
        'data_dir': config.DEMAND_FORECASTING_DIR,
        'train_file': 'train.csv',
        'test_file': 'test.csv',
        'date_column': 'date',
        'target_column': 'sales',
        'group_columns': ['store', 'item'],
        'use_subset': True,  # Para teste rápido
        'subset_filter': {'store': 1, 'item': 1},
        'forecast_horizon': 7,
        'validation_split': 0.2,
        'batch_size': 128,
        'epochs': 100,
        'early_stopping_patience': 15,
        'learning_rate': 0.0005,
    },
    'predict_future_sales': {
        'name': 'Predict Future Sales',
        'data_dir': config.PREDICT_FUTURE_SALES_DIR,
        'train_file': 'sales_train.csv',
        'test_file': 'test.csv',
        'date_column': 'date',
        'target_column': 'item_cnt_day',
        'group_columns': ['shop_id', 'item_id'],
        'use_subset': True,
        'subset_filter': {'shop_id': 2, 'item_id': 32},
        'forecast_horizon': 7,
        'validation_split': 0.2,
        'batch_size': 128,
        'epochs': 100,
        'early_stopping_patience': 15,
        'learning_rate': 0.0005,
        'date_format': '%d.%m.%Y'
    }
}


def get_dataset_config(dataset_name):
    """
    Retorna a configuração para um dataset específico

    Args:
        dataset_name: Nome do dataset (chave em DATASET_CONFIGS)

    Returns:
        dict com configurações
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{dataset_name}' não encontrado. "
                        f"Opções: {list(DATASET_CONFIGS.keys())}")

    return DATASET_CONFIGS[dataset_name]


def list_available_datasets():
    """Retorna lista de datasets disponíveis"""
    return list(DATASET_CONFIGS.keys())
