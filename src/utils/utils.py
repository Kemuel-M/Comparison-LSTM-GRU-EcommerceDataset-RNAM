"""
Funções utilitárias
"""
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src import config


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


import sys

class LoggerWrapper:
    """Redireciona stdout para console e arquivo"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_device():
    """Retorna o device disponível (cuda ou cpu)"""
    if torch.cuda.is_available() and config.DEVICE == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model):
    """Conta o número de parâmetros treináveis do modelo"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ExperimentLogger:
    """Logger para salvar informações dos experimentos em arquivo TXT"""

    def __init__(self, output_dir, dataset_name):
        from pathlib import Path
        from datetime import datetime
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"experiment_log_{dataset_name}_{timestamp}.txt"

        self.log_lines = []
        self.start_time = datetime.now()
        
        self.log_section(f"EXPERIMENTO: {dataset_name.upper()}")
        self.log(f"Início: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def log(self, message, print_console=True):
        """Adiciona mensagem ao log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        self.log_lines.append(line)

        if print_console:
            print(message)

    def log_metrics(self, model_name, metrics):
        """Loga as métricas de um modelo específico de forma organizada"""
        self.log(f"\n--- Resultados: {model_name} ---")
        self.log(f"RMSE: {metrics.get('RMSE', 'N/A'):.6f}")
        self.log(f"MAE:  {metrics.get('MAE', 'N/A'):.6f}")
        self.log(f"Tempo de Treino: {metrics.get('training_time', 'N/A'):.2f}s")
        self.log(f"Parâmetros: {metrics.get('num_params', 'N/A'):,}")
        self.log("-" * 30)

    def save(self):
        """Salva o log em arquivo"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.log_lines))
        print(f"\nLog finalizado e salvo em: {self.log_file}")

    def log_section(self, title):
        """Adiciona uma seção ao log"""
        separator = "=" * 80
        self.log(f"\n{separator}")
        self.log(f" {title} ")
        self.log(separator)
