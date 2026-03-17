"""
Módulo de avaliação de performance e visualização de resultados.
Responsável por calcular métricas estatísticas, comparar modelos e gerar gráficos.
"""
import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union
from src.utils.utils import calculate_metrics, plot_predictions, plot_training_history, count_parameters
from src import config


def evaluate_model(
    model: nn.Module, 
    predictions: np.ndarray, 
    targets: np.ndarray, 
    history: Dict[str, List[float]], 
    training_time: float, 
    model_name: str
) -> Dict[str, Any]:
    """Avalia o desempenho de um modelo treinado e gera visualizações de resultados.

    Calcula métricas de erro (RMSE, MAE, etc.), extrai estatísticas do modelo 
    (parâmetros) e do histórico de treino. Além disso, gera e salva gráficos 
    de previsões vs real e curvas de aprendizado.

    Args:
        model (nn.Module): O modelo PyTorch avaliado.
        predictions (np.ndarray): Matriz de previsões [amostras, horizonte].
        targets (np.ndarray): Matriz de valores reais [amostras, horizonte].
        history (Dict[str, List[float]]): Histórico de perdas durante as épocas.
        training_time (float): Tempo total gasto no treinamento em segundos.
        model_name (str): Nome identificador para os logs e arquivos de imagem.

    Returns:
        Dict[str, Any]: Um dicionário consolidado com todas as métricas calculadas.
    """
    print(f"\n{'='*60}")
    print(f"Avaliando {model_name}")
    print(f"{'='*60}")

    # Flatten predictions e targets (pegar apenas o primeiro passo de previsão para métricas simples)
    # Em séries temporais multi-step, é comum avaliar o primeiro passo ou a média do horizonte.
    pred_flat = predictions[:, 0]  
    target_flat = targets[:, 0]

    # Calcular métricas básicas (RMSE, MAE, R2)
    metrics = calculate_metrics(target_flat, pred_flat)

    # Adicionar metadados e métricas de treinamento
    metrics['training_time'] = training_time
    metrics['num_parameters'] = count_parameters(model)
    metrics['final_train_loss'] = history['train_loss'][-1]
    metrics['final_val_loss'] = history['val_loss'][-1]
    metrics['best_val_loss'] = min(history['val_loss'])

    # Print métricas no console
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"MAE: {metrics['MAE']:.6f}")
    print(f"Tempo de treinamento: {metrics['training_time']:.2f}s")
    print(f"Número de parâmetros: {metrics['num_parameters']:,}")
    print(f"Melhor Val Loss: {metrics['best_val_loss']:.6f}")

    # Geração automática de gráficos
    try:
        # Gráfico de previsões (Real vs Predito)
        pred_save_path = config.FIGURES_DIR / f"{model_name}_predictions.png"
        plot_predictions(target_flat, pred_flat, model_name, pred_save_path)
        print(f"Gráfico de previsões salvo em: {pred_save_path}")

        # Gráfico de histórico (Curvas de Loss e LR)
        hist_save_path = config.FIGURES_DIR / f"{model_name}_history.png"
        plot_training_history(history, model_name, hist_save_path)
        print(f"Gráfico de histórico salvo em: {hist_save_path}")
    except Exception as e:
        print(f"Aviso: Erro ao salvar gráficos: {e}")

    return metrics


def compare_models(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Compara estatisticamente os resultados de múltiplos modelos e gera um ranking.

    Args:
        results (Dict[str, Dict[str, Any]]): Dicionário onde as chaves são nomes 
            dos modelos e os valores são seus respectivos dicionários de métricas.

    Returns:
        pd.DataFrame: DataFrame comparativo ordenado por RMSE, salvo em CSV.
    """
    print(f"\n{'='*60}")
    print("COMPARAÇÃO DOS MODELOS")
    print(f"{'='*60}\n")

    # Converter o dicionário aninhado em um DataFrame
    comparison = pd.DataFrame(results).T

    # Ordenar por RMSE (menor erro é melhor)
    comparison = comparison.sort_values('RMSE')

    print(comparison.to_string())

    # Salvar resultados consolidados para persistência
    save_path = config.RESULTS_DIR / "comparison.csv"
    comparison.to_csv(save_path)
    print(f"\nComparação salva em: {save_path}")

    # Análise sumária dos "vencedores" por categoria
    print("\n" + "="*60)
    print("INSIGHTS DA COMPARAÇÃO")
    print("="*60)

    best_rmse = comparison['RMSE'].idxmin()
    best_mae = comparison['MAE'].idxmin()
    fastest = comparison['training_time'].idxmin()

    print(f"\n🏆 Melhor RMSE: {best_rmse} ({comparison.loc[best_rmse, 'RMSE']:.6f})")
    print(f"🏆 Melhor MAE: {best_mae} ({comparison.loc[best_mae, 'MAE']:.6f})")
    print(f"⚡ Mais rápido: {fastest} ({comparison.loc[fastest, 'training_time']:.2f}s)")

    # Exibição do trade-off Precisão vs Eficiência
    print("\n" + "-"*60)
    print("TRADE-OFF: PRECISÃO vs CUSTO COMPUTACIONAL")
    print("-"*60)

    for model_name in comparison.index:
        rmse = comparison.loc[model_name, 'RMSE']
        time = comparison.loc[model_name, 'training_time']
        print(f"-> {model_name}: RMSE={rmse:.6f} | Tempo={time:.2f}s")

    return comparison
