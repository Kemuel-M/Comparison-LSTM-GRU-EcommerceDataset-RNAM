"""
Módulo de treinamento para modelos de Deep Learning com PyTorch.
Inclui loops de treino, validação, lógica de Early Stopping e gerenciamento de métricas.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from typing import Tuple, Dict, List, Any, Optional
from torch.utils.data import DataLoader
from src import config
from src.utils.utils import get_device


class EarlyStopping:
    """Implementação de Early Stopping para evitar overfitting.

    Interrompe o treinamento se a perda de validação não melhorar após um
    número especificado de épocas (patience), opcionalmente considerando uma
    melhoria mínima (min_delta).

    Attributes:
        patience (int): Número de épocas para aguardar melhora antes de parar.
        min_delta (float): Melhoria mínima necessária para resetar o contador.
        counter (int): Contador interno de épocas sem melhora.
        best_loss (Optional[float]): A menor perda de validação encontrada até agora.
        early_stop (bool): Flag indicando se o treinamento deve ser interrompido.
    """

    def __init__(self, patience: int = config.EARLY_STOPPING_PATIENCE, min_delta: float = 0) -> None:
        """Inicializa o monitor de parada precoce.

        Args:
            patience (int): Épocas de tolerância. Padrão definido em config.
            min_delta (float): Diferença mínima para considerar progresso.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        """Verifica se a perda de validação atual melhorou.

        Args:
            val_loss (float): Perda de validação da época atual.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(
    model: nn.Module, 
    train_loader: DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    device: torch.device
) -> float:
    """Executa o treinamento do modelo por uma única época.

    Args:
        model (nn.Module): O modelo a ser treinado.
        train_loader (DataLoader): Carregador de dados de treinamento.
        criterion (nn.Module): Função de perda (loss function).
        optimizer (optim.Optimizer): Algoritmo de otimização (ex: Adam).
        device (torch.device): Dispositivo de hardware (CPU/GPU).

    Returns:
        float: Média da perda (loss) para todos os batches da época.
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(
    model: nn.Module, 
    val_loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> float:
    """Avalia o desempenho do modelo em um conjunto de validação.

    Args:
        model (nn.Module): Modelo em modo de avaliação.
        val_loader (DataLoader): Carregador de dados de validação.
        criterion (nn.Module): Função de perda usada para o cálculo.
        device (torch.device): Dispositivo de hardware.

    Returns:
        float: Média da perda (loss) de validação da época.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            total_loss += loss.item()

    return total_loss / len(val_loader)


def train_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    model_name: str,
    epochs: int = config.EPOCHS, 
    learning_rate: float = config.LEARNING_RATE
) -> Tuple[nn.Module, Dict[str, List[float]], float]:
    """Orquestra o ciclo completo de treinamento de um modelo.

    Inclui inicialização do dispositivo, otimizador, scheduler de taxa de 
    aprendizado e monitoramento via Early Stopping.

    Args:
        model (nn.Module): Instância da arquitetura a treinar.
        train_loader (DataLoader): Dados de treino.
        val_loader (DataLoader): Dados de validação.
        model_name (str): Nome identificador para logs.
        epochs (int): Número máximo de épocas.
        learning_rate (float): Taxa de aprendizado inicial.

    Returns:
        Tuple[nn.Module, Dict[str, List[float]], float]: Retorna o modelo treinado,
            um dicionário com o histórico de perdas/LR e o tempo total em segundos.
    """
    print(f"\n{'='*60}")
    print(f"Treinando {model_name}")
    print(f"{'='*60}")

    device = get_device()
    print(f"Device: {device}")

    model = model.to(device)

    # Loss e optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Scheduler: Reduz LR se a perda estagnar
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

    # Histórico de métricas
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    # Monitor de tempo
    start_time = time.time()

    # Loop de épocas
    for epoch in range(epochs):
        # Treinar
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validar
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update scheduler baseado na perda de validação
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Salvar histórico
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        # Print progress
        print(f"Época {epoch+1:03d}/{epochs} - "
              f"Train Loss: {train_loss:.6f} - "
              f"Val Loss: {val_loss:.6f} - "
              f"LR: {current_lr:.6f}")

        # Verificação de Early Stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping na época {epoch+1}")
            break

    training_time = time.time() - start_time
    print(f"\nTempo de treinamento: {training_time:.2f} segundos")

    return model, history, training_time


def predict(
    model: nn.Module, 
    data_loader: DataLoader, 
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Gera previsões para um conjunto de dados utilizando o modelo treinado.

    Args:
        model (nn.Module): O modelo já treinado.
        data_loader (DataLoader): Dados para os quais gerar as previsões.
        device (torch.device): Hardware a ser utilizado.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Par de arrays contendo 
            (previsões concatenadas, alvos reais concatenados).
    """
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)

            pred = model(X_batch)

            predictions.append(pred.cpu().numpy())
            targets.append(y_batch.cpu().numpy())

    # Concatenar todos os batches em arrays únicos
    predictions_arr = np.concatenate(predictions, axis=0)
    targets_arr = np.concatenate(targets, axis=0)

    return predictions_arr, targets_arr
