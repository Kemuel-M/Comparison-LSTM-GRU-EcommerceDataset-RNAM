"""
Definição dos modelos LSTM e GRU com arquiteturas flexíveis
"""
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    Modelo LSTM para previsão de séries temporais
    """

    def __init__(self, input_size=1, hidden_size=128,
                 num_layers=2, dropout=0.2,
                 forecast_horizon=7):
        """
        Args:
            input_size: Número de features de entrada
            hidden_size: Tamanho da camada oculta
            num_layers: Número de camadas LSTM
            dropout: Taxa de dropout
            forecast_horizon: Número de passos a prever
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

    def forward(self, x):
        """
        Forward pass
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Usar apenas o último estado oculto
        out = self.fc(lstm_out[:, -1, :])

        return out


class GRUModel(nn.Module):
    """
    Modelo GRU para previsão de séries temporais
    """

    def __init__(self, input_size=1, hidden_size=128,
                 num_layers=2, dropout=0.2,
                 forecast_horizon=7):
        """
        Args:
            input_size: Número de features de entrada
            hidden_size: Tamanho da camada oculta
            num_layers: Número de camadas GRU
            dropout: Taxa de dropout
            forecast_horizon: Número de passos a prever
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

        # Camada de saída (MLP para mais "poder" de representação)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon)
        )

    def forward(self, x):
        """
        Forward pass
        """
        # GRU
        gru_out, h_n = self.gru(x)

        # Usar apenas o último estado oculto
        out = self.fc(gru_out[:, -1, :])

        return out


class BiLSTMModel(nn.Module):
    """
    Modelo LSTM Bidirecional para previsão de séries temporais
    """

    def __init__(self, input_size=1, hidden_size=128,
                 num_layers=2, dropout=0.2,
                 forecast_horizon=7):
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

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Concatenar os estados finais das duas direções ou usar o último output
        out = self.fc(lstm_out[:, -1, :])
        return out


class BiGRUModel(nn.Module):
    """
    Modelo GRU Bidirecional para previsão de séries temporais
    """

    def __init__(self, input_size=1, hidden_size=128,
                 num_layers=2, dropout=0.2,
                 forecast_horizon=7):
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

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out


def get_model(model_type='lstm', **kwargs):
    """
    Factory para criar modelos
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
