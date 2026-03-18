"""
Módulo de utilitários gerais para o projeto.
Inclui funções para reprodutibilidade, cálculo de métricas, visualização de dados e log de experimentos.
"""
import numpy as np
import torch
import random
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src import config


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Define a semente aleatória para todas as bibliotecas relevantes.

    Garante que os resultados dos experimentos sejam reprodutíveis ao fixar
    as sementes do Python, NumPy e PyTorch (incluindo backends CUDA).

    Args:
        seed (int): Valor da semente aleatória. Padrão definido em config.RANDOM_SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula as principais métricas de erro para regressão de séries temporais.

    Args:
        y_true (np.ndarray): Valores reais observados.
        y_pred (np.ndarray): Valores previstos pelo modelo.

    Returns:
        Dict[str, float]: Dicionário contendo as métricas RMSE e MAE.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return {
        'RMSE': float(rmse),
        'MAE': float(mae)
    }


def plot_predictions(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    model_name: str, 
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """Gera um gráfico comparativo entre os valores reais e as previsões.

    Exibe uma amostra (fatia temporal) para facilitar a visualização da aderência do modelo.

    Args:
        y_true (np.ndarray): Valores reais.
        y_pred (np.ndarray): Valores previstos.
        model_name (str): Nome do modelo para o título do gráfico.
        save_path (Optional[Union[str, Path]]): Caminho para salvar a imagem. 
            Se None, exibe o gráfico interativamente.
    """
    plt.figure(figsize=(15, 6))

    # Limitar o tamanho da amostra para visualização clara (ex: últimos 200 pontos)
    sample_size = min(200, len(y_true))

    plt.plot(y_true[:sample_size], label='Real', linewidth=2, color='#2ecc71')
    plt.plot(y_pred[:sample_size], label='Previsto', linewidth=2, alpha=0.8, color='#e74c3c', linestyle='--')
    
    plt.xlabel('Passos Temporais (Amostras)')
    plt.ylabel('Valor da Demanda (Escalonado)')
    plt.title(f'Comparação: Real vs Previsto - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_history(
    history: Dict[str, List[float]], 
    model_name: str, 
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """Plota as curvas de perda (loss) de treino e validação ao longo das épocas.

    Args:
        history (Dict[str, List[float]]): Dicionário contendo listas de perdas por época.
        model_name (str): Nome do modelo para identificação.
        save_path (Optional[Union[str, Path]]): Local para salvar a figura.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(history['train_loss'], label='Treino', linewidth=2, color='#3498db')
    plt.plot(history['val_loss'], label='Validação', linewidth=2, color='#f39c12')
    
    plt.xlabel('Época')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Curva de Aprendizado - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


class LoggerWrapper:
    """Wrapper para redirecionar o fluxo de saída do sistema (stdout).
    
    Permite imprimir mensagens simultaneamente no console e em um arquivo de texto.

    Attributes:
        terminal (TextIO): Referência ao sys.stdout original.
        log (File): Referência ao objeto de arquivo aberto para escrita.
    """
    def __init__(self, filename: Union[str, Path]) -> None:
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


def get_device() -> torch.device:
    """Identifica e retorna o hardware de processamento disponível.

    Returns:
        torch.device: Retorna 'cuda' se houver GPU e estiver configurado, caso contrário 'cpu'.
    """
    if torch.cuda.is_available() and config.DEVICE == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Calcula o número total de parâmetros treináveis em um modelo PyTorch.

    Args:
        model (torch.nn.Module): O modelo a ser analisado.

    Returns:
        int: Total de elementos (pesos e bias) com gradiente ativo.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ExperimentLogger:
    """Gerencia logs detalhados de sessões de experimentos.
    
    Cria arquivos de log individuais por dataset e timestamp, mantendo um registro
    organizado de métricas e eventos durante a orquestração.

    Attributes:
        output_dir (Path): Diretório base para salvamento dos logs.
        log_file (Path): Caminho completo do arquivo de log da sessão atual.
        log_lines (List[str]): Buffer de memória para as linhas de log.
    """

    def __init__(self, output_dir: Union[str, Path], dataset_name: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"experiment_log_{dataset_name}_{timestamp}.txt"

        self.log_lines: List[str] = []
        self.start_time = datetime.now()
        
        self.log_section(f"EXPERIMENTO: {dataset_name.upper()}")
        self.log(f"Início: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def log(self, message: str, print_console: bool = True) -> None:
        """Adiciona uma mensagem ao log com carimbo de tempo (timestamp).

        Args:
            message (str): Texto a ser registrado.
            print_console (bool): Se True, também imprime no console via print().
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        self.log_lines.append(line)

        if print_console:
            print(message)

    def log_metrics(self, model_name: str, metrics: Dict[str, Any]) -> None:
        """Registra métricas de desempenho de um modelo de forma formatada.

        Args:
            model_name (str): Nome do modelo.
            metrics (Dict[str, Any]): Dicionário com métricas calculadas.
        """
        self.log(f"\n--- Resultados: {model_name} ---")
        self.log(f"RMSE: {metrics.get('RMSE', 0.0):.6f}")
        self.log(f"MAE:  {metrics.get('MAE', 0.0):.6f}")
        self.log(f"Tempo de Treino: {metrics.get('training_time', 0.0):.2f}s")
        self.log(f"Parâmetros: {metrics.get('num_parameters', 0):,}")
        self.log("-" * 30)

    def save(self) -> None:
        """Persiste todas as linhas de log acumuladas em buffer no disco."""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.log_lines))
        print(f"\nLog finalizado e salvo em: {self.log_file}")

    def log_section(self, title: str) -> None:
        """Adiciona um cabeçalho visual de seção ao log.

        Args:
            title (str): Título da seção.
        """
        separator = "=" * 80
        self.log(f"\n{separator}")
        self.log(f" {title} ")
        self.log(separator)
