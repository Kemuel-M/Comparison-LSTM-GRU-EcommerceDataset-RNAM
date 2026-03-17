"""
Módulo de carregamento e preparação de dados para modelos de séries temporais.
Contém classes e funções para manipulação de datasets, escalonamento e criação de janelas.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict, List, Any, Union
from src import config


class DemandDataset(Dataset):
    """Dataset personalizado para previsão de demanda com PyTorch.

    Esta classe herda de `torch.utils.data.Dataset` e é responsável por
    converter arrays NumPy em tensores FloatTensor do PyTorch para treinamento.

    Attributes:
        X (torch.Tensor): Tensores de entrada (janelas de tempo deslizantes).
        y (torch.Tensor): Tensores de saída (valores futuros a prever).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """Inicializa o dataset com features e targets.

        Args:
            X (np.ndaray): Array NumPy com as janelas de tempo.
            y (np.ndarray): Array NumPy com os valores alvo.
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        """Retorna o número total de amostras no dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Obtém um par (entrada, alvo) dado um índice.

        Args:
            idx (int): Índice da amostra.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Par contendo (features, target).
        """
        return self.X[idx], self.y[idx]


def load_data(
    file_path: Union[str, Any], 
    date_column: str = 'date', 
    target_column: str = 'sales', 
    group_columns: Optional[List[str]] = None, 
    date_format: Optional[str] = None
) -> pd.DataFrame:
    """Carrega dados brutos de um arquivo CSV e realiza pré-processamento básico de datas.

    Args:
        file_path (Union[str, Path]): Caminho para o arquivo CSV.
        date_column (str): Nome da coluna contendo as datas. Padrão: 'date'.
        target_column (str): Nome da coluna alvo (ex: vendas). Padrão: 'sales'.
        group_columns (Optional[List[str]]): Lista de colunas usadas para agrupamento/ID.
        date_format (Optional[str]): Formato de data específico para o parser do Pandas.

    Returns:
        pd.DataFrame: DataFrame carregado e com colunas de data tipadas.
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


def create_sequences(
    data: np.ndarray, 
    window_size: int, 
    forecast_horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Cria sequências de janelas deslizantes (Sliding Window) para modelos autorregressivos.

    Args:
        data (np.ndarray): Array 1D ou 2D contendo a série temporal.
        window_size (int): Tamanho da janela de entrada (passado).
        forecast_horizon (int): Quantidade de passos à frente a prever (futuro).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Par (X, y) contendo as janelas de entrada
            e seus respectivos alvos.
    """
    X, y = [], []

    for i in range(len(data) - window_size - forecast_horizon + 1):
        # Janela de entrada
        X.append(data[i:i + window_size])
        # Valores futuros a prever
        y.append(data[i + window_size:i + window_size + forecast_horizon])

    return np.array(X), np.array(y)


def prepare_data(
    df: pd.DataFrame, 
    window_size: int = config.WINDOW_SIZE,
    forecast_horizon: int = config.FORECAST_HORIZON,
    validation_split: float = config.VALIDATION_SPLIT,
    use_subset: bool = True,
    subset_filter: Optional[Dict[str, Any]] = None,
    date_column: str = 'date',
    target_column: str = 'sales'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Realiza o pipeline completo de preparação de dados para treinamento.

    Inclui filtragem (subset), ordenação temporal, normalização (StandardScaler),
    criação de sequências e divisão em treino/validação.

    Args:
        df (pd.DataFrame): DataFrame original.
        window_size (int): Tamanho da janela de entrada.
        forecast_horizon (int): Horizonte de previsão.
        validation_split (float): Proporção dos dados finais para validação.
        use_subset (bool): Se deve usar apenas uma fatia dos dados para agilizar o treino.
        subset_filter (Optional[Dict[str, Any]]): Filtro específico (ex: {'store': 1}).
        date_column (str): Nome da coluna de data.
        target_column (str): Nome da coluna alvo.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]: 
            Contém (X_train, y_train, X_val, y_val, scaler).
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


def get_data_loaders(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray, 
    batch_size: int = config.BATCH_SIZE
) -> Tuple[DataLoader, DataLoader]:
    """Encapsula arrays NumPy em DataLoaders do PyTorch.

    Args:
        X_train (np.ndarray): Features de treino.
        y_train (np.ndarray): Targets de treino.
        X_val (np.ndarray): Features de validação.
        y_val (np.ndarray): Targets de validação.
        batch_size (int): Tamanho do lote para o carregador.

    Returns:
        Tuple[DataLoader, DataLoader]: Iteradores (train_loader, val_loader).
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
