"""
Avaliação dos modelos
"""
import numpy as np
import pandas as pd
from src.utils.utils import calculate_metrics, plot_predictions, plot_training_history, count_parameters
from src import config


def evaluate_model(model, predictions, targets, history, training_time, model_name):
    """
    Avalia o modelo e retorna métricas

    Args:
        model: Modelo treinado
        predictions: Previsões do modelo
        targets: Valores reais
        history: Histórico de treinamento
        training_time: Tempo de treinamento
        model_name: Nome do modelo

    Returns:
        dict com métricas
    """
    print(f"\n{'='*60}")
    print(f"Avaliando {model_name}")
    print(f"{'='*60}")

    # Flatten predictions e targets (pegar apenas o primeiro passo de previsão)
    # ou fazer média de todos os passos
    pred_flat = predictions[:, 0]  # Primeiro passo de previsão
    target_flat = targets[:, 0]

    # Calcular métricas
    metrics = calculate_metrics(target_flat, pred_flat)

    # Adicionar métricas extras
    metrics['training_time'] = training_time
    metrics['num_parameters'] = count_parameters(model)
    metrics['final_train_loss'] = history['train_loss'][-1]
    metrics['final_val_loss'] = history['val_loss'][-1]
    metrics['best_val_loss'] = min(history['val_loss'])

    # Print métricas
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAE: {metrics['MAE']:.6f}")
    print(f"Tempo de treinamento: {metrics['training_time']:.2f}s")
    print(f"Número de parâmetros: {metrics['num_parameters']:,}")
    print(f"Melhor Val Loss: {metrics['best_val_loss']:.6f}")

    # Plotar previsões (com prefixo de dataset se disponível)
    try:
        save_path = config.FIGURES_DIR / f"{model_name}_predictions.png"
        plot_predictions(target_flat, pred_flat, model_name, save_path)
        print(f"Gráfico de previsões salvo em: {save_path}")

        # Plotar histórico
        save_path = config.FIGURES_DIR / f"{model_name}_history.png"
        plot_training_history(history, model_name, save_path)
        print(f"Gráfico de histórico salvo em: {save_path}")
    except Exception as e:
        print(f"Aviso: Erro ao salvar gráficos: {e}")

    return metrics


def compare_models(results):
    """
    Compara os resultados dos modelos

    Args:
        results: Dict com resultados de cada modelo
    """
    print(f"\n{'='*60}")
    print("COMPARAÇÃO DOS MODELOS")
    print(f"{'='*60}\n")

    # Criar DataFrame para comparação
    comparison = pd.DataFrame(results).T

    # Ordenar por RMSE (menor é melhor)
    comparison = comparison.sort_values('RMSE')

    print(comparison.to_string())

    # Salvar em CSV
    save_path = config.RESULTS_DIR / "comparison.csv"
    comparison.to_csv(save_path)
    print(f"\nComparação salva em: {save_path}")

    # Análise
    print("\n" + "="*60)
    print("ANÁLISE")
    print("="*60)

    best_rmse = comparison['RMSE'].idxmin()
    best_mae = comparison['MAE'].idxmin()
    fastest = comparison['training_time'].idxmin()

    print(f"\nMelhor RMSE: {best_rmse} ({comparison.loc[best_rmse, 'RMSE']:.6f})")
    print(f"Melhor MAE: {best_mae} ({comparison.loc[best_mae, 'MAE']:.6f})")
    print(f"Mais rápido: {fastest} ({comparison.loc[fastest, 'training_time']:.2f}s)")

    # Análise de trade-off
    print("\n" + "-"*60)
    print("TRADE-OFF PRECISÃO vs TEMPO")
    print("-"*60)

    for model_name in comparison.index:
        rmse = comparison.loc[model_name, 'RMSE']
        time = comparison.loc[model_name, 'training_time']
        print(f"{model_name}: RMSE={rmse:.6f}, Tempo={time:.2f}s")

    return comparison
