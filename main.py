"""
Script principal para comparação exaustiva LSTM vs GRU
Refatorado para melhor organização, modularização e documentação profissional.
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import mlflow
import argparse
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

from src import config
from src.data.dataset_configs import list_available_datasets
from src.utils.utils import set_seed, ExperimentLogger, LoggerWrapper
from src.experiments.orchestrator import run_dataset_experiments, run_best_comparison


def save_consolidated_results(all_metrics: List[Dict[str, Any]], model_type: str) -> None:
    """Salva e consolida os resultados dos experimentos em CSV e gera um relatório TXT.

    Esta função pega a lista de métricas coletadas durante os experimentos, 
    converte em um DataFrame do Pandas, concatena com resultados existentes (se houver),
    remove duplicatas baseadas na arquitetura e janela, e persiste os dados em disco.
    Também gera um relatório TXT com o Top 5 modelos e médias por categoria.

    Args:
        all_metrics (List[Dict[str, Any]]): Lista de dicionários contendo métricas e 
            parâmetros de cada modelo treinado.
        model_type (str): Identificador do tipo de execução ('lstm', 'gru' ou 'all')
            para nomeação do arquivo de resumo.

    Returns:
        None
    """
    df_results = pd.DataFrame(all_metrics)
    results_path = config.RESULTS_DIR / "comprehensive_comparison.csv"
    
    if results_path.exists() and model_type != 'all':
        df_old = pd.read_csv(results_path)
        # Combinar e remover duplicatas baseadas no nome do modelo
        df_results = pd.concat([df_old, df_results]).drop_duplicates(
            subset=['model_type', 'architecture', 'window_size'], keep='last'
        )
        
    df_results.to_csv(results_path, index=False)
    print(f"\nResultados salvos/atualizados em: {results_path}")
    
    # Gerar relatório resumo em TXT
    summary_path = config.RESULTS_DIR / f"summary_report_{model_type}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"RELATÓRIO DE COMPARAÇÃO: {model_type.upper()}\n")
        f.write("="*50 + "\n\n")
        
        f.write("TOP 5 MODELOS POR RMSE:\n")
        f.write(df_results.sort_values('RMSE').head(5)[['model_type', 'architecture', 'window_size', 'RMSE', 'MAE']].to_string() + "\n\n")
        
        f.write("MÉDIAS POR TIPO DE MODELO:\n")
        if not df_results.empty:
            f.write(df_results.groupby('model_type')[['RMSE', 'MAE', 'training_time']].mean().to_string() + "\n\n")
        
    print(f"Relatório resumo salvo em: {summary_path}")


def main() -> None:
    """Ponto de entrada principal para a execução da bateria de experimentos.

    Configura o ambiente (seeds, logs, MLflow), processa argumentos da linha de comando
    e orquestra a execução dos experimentos para cada dataset disponível ou selecionado.
    Ao final, consolida os resultados em relatórios.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='Comparação LSTM vs GRU para Previsão de Demanda')
    parser.add_argument('--model_type', type=str, choices=['lstm', 'gru', 'all'], default='all',
                        help='Tipo de modelo para executar (padrão: all)')
    parser.add_argument('--compare_best', action='store_true',
                        help='Executar apenas a comparação entre os melhores de cada tipo já treinados')
    parser.add_argument('--dataset', type=str, choices=list_available_datasets(),
                        help='Dataset específico para rodar (padrão: todos os disponíveis)')
    parser.add_argument('--save_models', action='store_true',
                        help='Salvar os modelos treinados em disco (.pt)')
    
    args = parser.parse_args()
    
    set_seed(config.RANDOM_SEED)
    
    # Configurar log de console
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"console_{timestamp}_{args.model_type}"
    if args.compare_best:
        log_name = f"console_{timestamp}_comparison"
    
    console_log_path = config.RESULTS_DIR / f"{log_name}.txt"
    sys.stdout = LoggerWrapper(console_log_path)
    
    # Configurar MLflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    
    available_datasets = [args.dataset] if args.dataset else list_available_datasets()
    all_metrics: List[Dict[str, Any]] = []
    model_types = ['lstm', 'gru'] if args.model_type == 'all' else [args.model_type]
    
    for dataset_name in available_datasets:
        logger = ExperimentLogger(config.RESULTS_DIR, dataset_name)
        
        try:
            if args.compare_best:
                results = run_best_comparison(dataset_name, logger, model_types=model_types, save_models=args.save_models)
                if results:
                    all_metrics.extend(results)
            else:
                metrics_list = run_dataset_experiments(dataset_name, model_types, logger, save_models=args.save_models)
                if metrics_list:
                    all_metrics.extend(metrics_list)
        except Exception as e:
            logger.log(f"Erro fatal durante execução: {str(e)}")
            import traceback
            logger.log(traceback.format_exc())
        finally:
            logger.save()
            
    # Salvar resultados consolidados
    if all_metrics:
        # Define o sufixo do relatório baseado no modo
        report_suffix = args.model_type if not args.compare_best else f"best_{args.model_type}"
        save_consolidated_results(all_metrics, report_suffix)
        
        if args.compare_best:
            print("\n" + "="*50)
            print(f"RESULTADOS FINAIS: MELHOR(ES) {args.model_type.upper()}")
            print("="*50)
            df_final = pd.DataFrame(all_metrics)
            cols = [c for c in ['model_type', 'architecture', 'window_size', 'RMSE', 'MAE', 'training_time'] if c in df_final.columns]
            print(df_final[cols])


if __name__ == "__main__":
    main()
