"""
Treinamento dos modelos
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm
import config
from utils import get_device


class EarlyStopping:
    """Early stopping para parar o treinamento quando a validação não melhora"""

    def __init__(self, patience=config.EARLY_STOPPING_PATIENCE, min_delta=0):
        """
        Args:
            patience: Número de épocas sem melhoria antes de parar
            min_delta: Melhoria mínima para considerar progresso
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Treina por uma época

    Args:
        model: Modelo a treinar
        train_loader: DataLoader de treino
        criterion: Função de loss
        optimizer: Otimizador
        device: Device (cuda ou cpu)

    Returns:
        Loss médio da época
    """
    model.train()
    total_loss = 0

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


def validate(model, val_loader, criterion, device):
    """
    Valida o modelo

    Args:
        model: Modelo a validar
        val_loader: DataLoader de validação
        criterion: Função de loss
        device: Device (cuda ou cpu)

    Returns:
        Loss médio da validação
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            total_loss += loss.item()

    return total_loss / len(val_loader)


def train_model(model, train_loader, val_loader, model_name,
                epochs=config.EPOCHS, learning_rate=config.LEARNING_RATE):
    """
    Treina o modelo completo

    Args:
        model: Modelo a treinar
        train_loader: DataLoader de treino
        val_loader: DataLoader de validação
        model_name: Nome do modelo
        epochs: Número de épocas
        learning_rate: Taxa de aprendizado

    Returns:
        model, history, training_time
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
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

    # Histórico
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    # Tempo de treinamento
    start_time = time.time()

    # Treinamento
    for epoch in range(epochs):
        # Treinar
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validar
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Salvar histórico
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        # Print progress
        print(f"Época {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.6f} - "
              f"Val Loss: {val_loss:.6f} - "
              f"LR: {current_lr:.6f}")

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping na época {epoch+1}")
            break

    training_time = time.time() - start_time
    print(f"\nTempo de treinamento: {training_time:.2f} segundos")

    return model, history, training_time


def predict(model, data_loader, device):
    """
    Faz previsões

    Args:
        model: Modelo treinado
        data_loader: DataLoader com dados
        device: Device

    Returns:
        predictions, targets (arrays numpy)
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

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    return predictions, targets
