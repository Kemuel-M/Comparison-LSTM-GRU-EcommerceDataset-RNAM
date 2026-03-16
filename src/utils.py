"""
Funções utilitárias
"""
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import config


def set_seed(seed=config.RANDOM_SEED):
    """Define seed para reprodutibilidade"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_metrics(y_true, y_pred):
    """
    Calcula métricas de avaliação

    Args:
        y_true: Valores reais
        y_pred: Valores previstos

    Returns:
        dict com RMSE e MAE
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return {
        'RMSE': rmse,
        'MAE': mae
    }


def plot_predictions(y_true, y_pred, model_name, save_path=None):
    """
    Plota previsões vs valores reais

    Args:
        y_true: Valores reais
        y_pred: Valores previstos
        model_name: Nome do modelo
        save_path: Caminho para salvar a figura
    """
    plt.figure(figsize=(15, 6))

    # Pegar apenas uma amostra para visualização (primeiros 200 pontos)
    sample_size = min(200, len(y_true))

    plt.plot(y_true[:sample_size], label='Real', linewidth=2)
    plt.plot(y_pred[:sample_size], label='Previsto', linewidth=2, alpha=0.7)
    plt.xlabel('Amostras')
    plt.ylabel('Vendas')
    plt.title(f'{model_name} - Previsões vs Valores Reais')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_history(history, model_name, save_path=None):
    """
    Plota histórico de treinamento

    Args:
        history: Dicionário com histórico de loss
        model_name: Nome do modelo
        save_path: Caminho para salvar a figura
    """
    plt.figure(figsize=(10, 6))

    plt.plot(history['train_loss'], label='Treino', linewidth=2)
    plt.plot(history['val_loss'], label='Validação', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('Loss (MSE)')
    plt.title(f'{model_name} - Histórico de Treinamento')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def get_device():
    """Retorna o device disponível (cuda ou cpu)"""
    if torch.cuda.is_available() and config.DEVICE == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model):
    """Conta o número de parâmetros treináveis do modelo"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
