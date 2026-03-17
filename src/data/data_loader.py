"""
Carregamento e preparação dos dados
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from src import config


class DemandDataset(Dataset):
    """Dataset personalizado para previsão de demanda"""

    def __init__(self, X, y):
        """
        Args:
            X: Features (janelas de tempo)
            y: Targets (valores futuros)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(file_path, date_column='date', target_column='sales', group_columns=None, date_format=None):
    """
    Carrega os dados do CSV

    Args:
        file_path: Caminho para o arquivo CSV
        date_column: Nome da coluna de data
        target_column: Nome da coluna alvo
        group_columns: Lista de colunas de agrupamento
        date_format: Formato da data (ex: '%d.%m.%Y')

    Returns:
        DataFrame com os dados
    """
    print(f"Carregando dados de {file_path}...")
    df = pd.read_csv(file_path)

    # Converter coluna de data
    if date_column in df.columns:
        if date_format:
            df[date_column] = pd.to_datetime(df[date_column], format=date_format)
        else:
            df[date_column] = pd.to_datetime(df[date_column])

    print(f"Dados carregados: {len(df)} registros")

    if date_column in df.columns:
        print(f"Período: {df[date_column].min()} até {df[date_column].max()}")

    # Informações sobre grupos
    if group_columns:
        for col in group_columns:
            if col in df.columns:
                print(f"{col}: {df[col].nunique()} únicos")

    return df


def create_sequences(data, window_size, forecast_horizon):
    """
    Cria sequências de janelas deslizantes

    Args:
        data: Array com os dados de vendas
        window_size: Tamanho da janela de entrada
        forecast_horizon: Quantidade de dias a prever

    Returns:
        X (features), y (targets)
    """
    X, y = [], []

    for i in range(len(data) - window_size - forecast_horizon + 1):
        # Janela de entrada
        X.append(data[i:i + window_size])
        # Valores futuros a prever
        y.append(data[i + window_size:i + window_size + forecast_horizon])

    return np.array(X), np.array(y)


def prepare_data(df, window_size=config.WINDOW_SIZE,
                 forecast_horizon=config.FORECAST_HORIZON,
                 validation_split=config.VALIDATION_SPLIT,
                 use_subset=True,
                 subset_filter=None,
                 date_column='date',
                 target_column='sales'):
    """
    Prepara os dados para treinamento

    Args:
        df: DataFrame com os dados
        window_size: Tamanho da janela de entrada
        forecast_horizon: Quantidade de dias a prever
        validation_split: Proporção para validação
        use_subset: Se True, usa apenas um subset para teste mais rápido
        subset_filter: Dict com filtros para subset (ex: {'store': 1, 'item': 1})
        date_column: Nome da coluna de data
        target_column: Nome da coluna alvo

    Returns:
        X_train, y_train, X_val, y_val, scaler
    """
    print("\nPreparando dados...")

    # Para acelerar testes, usar subset
    if use_subset and subset_filter:
        print(f"Usando subset dos dados com filtro: {subset_filter}")
        # Aplicar filtros usando máscaras booleanas
        mask = True
        for col, val in subset_filter.items():
            if col in df.columns:
                mask = mask & (df[col] == val)
        df = df[mask].copy()
    else:
        print("Usando todos os dados")

    # Ordenar por data se coluna existir
    if date_column in df.columns:
        df = df.sort_values(date_column)

    # Extrair valores alvo
    if target_column not in df.columns:
        raise ValueError(f"Coluna '{target_column}' não encontrada no DataFrame")

    sales = df[target_column].values.reshape(-1, 1)

    # Normalizar
    scaler = StandardScaler()
    sales_scaled = scaler.fit_transform(sales).flatten()

    # Criar sequências
    X, y = create_sequences(sales_scaled, window_size, forecast_horizon)

    print(f"Sequências criadas: {len(X)}")
    print(f"Shape X: {X.shape}, Shape y: {y.shape}")

    # Dividir em treino e validação
    split_idx = int(len(X) * (1 - validation_split))

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]

    # Adicionar dimensão de features (para RNN: [batch, seq_len, features])
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    print(f"Treino: {len(X_train)} amostras")
    print(f"Validação: {len(X_val)} amostras")

    return X_train, y_train, X_val, y_val, scaler


def get_data_loaders(X_train, y_train, X_val, y_val, batch_size=config.BATCH_SIZE):
    """
    Cria DataLoaders para treino e validação

    Args:
        X_train, y_train: Dados de treino
        X_val, y_val: Dados de validação
        batch_size: Tamanho do batch

    Returns:
        train_loader, val_loader
    """
    train_dataset = DemandDataset(X_train, y_train)
    val_dataset = DemandDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader
