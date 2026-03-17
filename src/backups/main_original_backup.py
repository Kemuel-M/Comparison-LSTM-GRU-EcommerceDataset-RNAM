"""
Script principal para comparação LSTM vs GRU
"""
import warnings
warnings.filterwarnings('ignore')

import torch
import mlflow
import mlflow.pytorch
import sys

import config
from utils import set_seed, get_device
from data_loader import load_data, prepare_data, get_data_loaders
from models import get_model
from train import train_model, predict
from evaluate import evaluate_model, compare_models


def run_experiment(model_type, train_loader, val_loader):
    """
    Executa experimento para um tipo de modelo

    Args:
        model_type: 'lstm' ou 'gru'
        train_loader: DataLoader de treino
        val_loader: DataLoader de validação

    Returns:
        model, history, training_time, predictions, targets, metrics
    """
    # Criar modelo
    model = get_model(model_type)
    model_name = model_type.upper()

    # Treinar
    model, history, training_time = train_model(
        model, train_loader, val_loader, model_name
    )

    # Fazer previsões
    device = get_device()
    predictions, targets = predict(model, val_loader, device)

    # Avaliar
    metrics = evaluate_model(
        model, predictions, targets, history, training_time, model_name
    )

    return model, history, training_time, predictions, targets, metrics


def main():
    """Função principal"""
    print("="*60)
    print("PREVISÃO DE DEMANDA: LSTM vs GRU")
    print("="*60)

    # Definir seed
    set_seed()

    # Configurar MLflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    print(f"\nMLflow tracking URI: {config.MLFLOW_TRACKING_URI}")
    print(f"Experimento: {config.EXPERIMENT_NAME}")

    # Carregar dados
    df = load_data(config.TRAIN_FILE)

    # Preparar dados
    X_train, y_train, X_val, y_val, scaler = prepare_data(
        df,
        window_size=config.WINDOW_SIZE,
        forecast_horizon=config.FORECAST_HORIZON,
        validation_split=config.VALIDATION_SPLIT,
        use_subset=True  # True para teste rápido, False para usar todos os dados
    )

    # Criar data loaders
    train_loader, val_loader = get_data_loaders(
        X_train, y_train, X_val, y_val,
        batch_size=config.BATCH_SIZE
    )

    # Resultados
    results = {}

    # Experimento 1: LSTM
    print("\n" + "="*60)
    print("EXPERIMENTO 1: LSTM")
    print("="*60)

    with mlflow.start_run(run_name="LSTM"):
        # Log parâmetros
        mlflow.log_params({
            "model_type": "LSTM",
            "window_size": config.WINDOW_SIZE,
            "forecast_horizon": config.FORECAST_HORIZON,
            "hidden_size": config.HIDDEN_SIZE,
            "num_layers": config.NUM_LAYERS,
            "dropout": config.DROPOUT,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.EPOCHS
        })

        # Executar experimento
        lstm_model, lstm_history, lstm_time, lstm_pred, lstm_target, lstm_metrics = \
            run_experiment('lstm', train_loader, val_loader)

        # Log métricas
        mlflow.log_metrics(lstm_metrics)

        # Log modelo
        mlflow.pytorch.log_model(lstm_model, "model")

        # Log artefatos (gráficos)
        mlflow.log_artifact(str(config.RESULTS_DIR / "LSTM_predictions.png"))
        mlflow.log_artifact(str(config.RESULTS_DIR / "LSTM_history.png"))

        results['LSTM'] = lstm_metrics

    # Experimento 2: GRU
    print("\n" + "="*60)
    print("EXPERIMENTO 2: GRU")
    print("="*60)

    with mlflow.start_run(run_name="GRU"):
        # Log parâmetros
        mlflow.log_params({
            "model_type": "GRU",
            "window_size": config.WINDOW_SIZE,
            "forecast_horizon": config.FORECAST_HORIZON,
            "hidden_size": config.HIDDEN_SIZE,
            "num_layers": config.NUM_LAYERS,
            "dropout": config.DROPOUT,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.EPOCHS
        })

        # Executar experimento
        gru_model, gru_history, gru_time, gru_pred, gru_target, gru_metrics = \
            run_experiment('gru', train_loader, val_loader)

        # Log métricas
        mlflow.log_metrics(gru_metrics)

        # Log modelo
        mlflow.pytorch.log_model(gru_model, "model")

        # Log artefatos (gráficos)
        mlflow.log_artifact(str(config.RESULTS_DIR / "GRU_predictions.png"))
        mlflow.log_artifact(str(config.RESULTS_DIR / "GRU_history.png"))

        results['GRU'] = gru_metrics

    # Comparar modelos
    comparison = compare_models(results)

    # Log comparação no MLflow
    with mlflow.start_run(run_name="Comparison"):
        mlflow.log_artifact(str(config.RESULTS_DIR / "comparison.csv"))

    print("\n" + "="*60)
    print("EXPERIMENTOS CONCLUÍDOS!")
    print("="*60)
    print(f"\nResultados salvos em: {config.RESULTS_DIR}")
    print(f"MLflow UI: Execute 'mlflow ui --backend-store-uri {config.MLFLOW_TRACKING_URI}'")
    print("="*60)


if __name__ == "__main__":
    main()
