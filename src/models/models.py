"""
Definição de arquiteturas de Redes Neurais Recorrentes (RNN) para Séries Temporais.
Inclui implementações flexíveis de LSTM, GRU e suas variantes bidirecionais.
"""
import torch
import torch.nn as nn
from typing import Any, Union


class LSTMModel(nn.Module):
    """Modelo LSTM customizável para previsão de séries temporais.

    Esta arquitetura utiliza uma ou mais camadas Long Short-Term Memory (LSTM)
    para capturar dependências de longo prazo, seguidas por uma cabeça de
    saída Multi-Layer Perceptron (MLP) com Dropout e ReLU.

    Attributes:
        hidden_size (int): Dimensão do estado oculto da LSTM.
        num_layers (int): Número de camadas recorrentes empilhadas.
        lstm (nn.LSTM): O módulo recorrente principal.
        fc (nn.Sequential): Camadas densas para mapeamento do horizonte de saída.
    """

    def __init__(
        self, 
        input_size: int = 1, 
        hidden_size: int = 128,
        num_layers: int = 2, 
        dropout: float = 0.2,
        forecast_horizon: int = 7
    ) -> None:
        """Inicializa as camadas do modelo LSTM.

        Args:
            input_size (int): Número de features de entrada por passo temporal.
            hidden_size (int): Tamanho da camada oculta.
            num_layers (int): Número de camadas LSTM.
            dropout (float): Taxa de abandono (dropout) entre camadas.
            forecast_horizon (int): Número de passos futuros a prever.
        """
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Camadas LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Camada de saída (MLP para mais "poder" de representação)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa a passagem para frente (forward pass) do modelo.

        Args:
            x (torch.Tensor): Tensor de entrada com shape [batch, seq_len, features].

        Returns:
            torch.Tensor: Previsões com shape [batch, forecast_horizon].
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Usar apenas o último estado oculto da sequência
        out = self.fc(lstm_out[:, -1, :])

        return out


class GRUModel(nn.Module):
    """Modelo GRU (Gated Recurrent Unit) para previsão de séries temporais.

    Similar à LSTM, mas utiliza uma arquitetura de portões mais simplificada,
    o que geralmente resulta em um treinamento mais rápido e menor custo computacional.

    Attributes:
        hidden_size (int): Dimensão do estado oculto.
        num_layers (int): Número de camadas recorrentes.
        gru (nn.GRU): O módulo recorrente GRU.
        fc (nn.Sequential): Camadas densas de saída.
    """

    def __init__(
        self, 
        input_size: int = 1, 
        hidden_size: int = 128,
        num_layers: int = 2, 
        dropout: float = 0.2,
        forecast_horizon: int = 7
    ) -> None:
        """Inicializa as camadas do modelo GRU.

        Args:
            input_size (int): Features por passo temporal.
            hidden_size (int): Tamanho da camada oculta.
            num_layers (int): Número de camadas GRU.
            dropout (float): Taxa de dropout.
            forecast_horizon (int): Passos futuros a prever.
        """
        super(GRUModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Camadas GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Camada de saída (MLP)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa o forward pass do modelo GRU.

        Args:
            x (torch.Tensor): Tensor [batch, seq_len, features].

        Returns:
            torch.Tensor: Tensor [batch, forecast_horizon].
        """
        # GRU
        gru_out, h_n = self.gru(x)

        # Usar apenas o último estado oculto
        out = self.fc(gru_out[:, -1, :])

        return out


class BiLSTMModel(nn.Module):
    """Modelo LSTM Bidirecional. Captura padrões tanto do passado para o presente
    quanto em uma representação reversa da sequência.
    """

    def __init__(
        self, 
        input_size: int = 1, 
        hidden_size: int = 128,
        num_layers: int = 2, 
        dropout: float = 0.2,
        forecast_horizon: int = 7
    ) -> None:
        super(BiLSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, forecast_horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        # O estado final contém informações das duas direções concatenadas
        out = self.fc(lstm_out[:, -1, :])
        return out


class BiGRUModel(nn.Module):
    """Modelo GRU Bidirecional."""

    def __init__(
        self, 
        input_size: int = 1, 
        hidden_size: int = 128,
        num_layers: int = 2, 
        dropout: float = 0.2,
        forecast_horizon: int = 7
    ) -> None:
        super(BiGRUModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, forecast_horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out


def get_model(model_type: str = 'lstm', **kwargs: Any) -> nn.Module:
    """Factory Pattern: Cria e retorna uma instância do modelo solicitado.

    Args:
        model_type (str): Tipo do modelo ('lstm', 'gru', 'bilstm', 'bigru').
        **kwargs (Any): Argumentos extras para o construtor do modelo (hidden_size, etc).

    Returns:
        nn.Module: Instância da arquitetura solicitada.

    Raises:
        ValueError: Caso o model_type não seja suportado.
    """
    model_type = model_type.lower()
    if model_type == 'lstm':
        return LSTMModel(**kwargs)
    elif model_type == 'gru':
        return GRUModel(**kwargs)
    elif model_type == 'bilstm':
        return BiLSTMModel(**kwargs)
    elif model_type == 'bigru':
        return BiGRUModel(**kwargs)
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")
