"""
Script principal para comparação exaustiva LSTM vs GRU
"""
import warnings
warnings.filterwarnings('ignore')

import torch
import mlflow
import mlflow.pytorch
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

import config
from dataset_configs import get_dataset_config, list_available_datasets
from utils import set_seed, get_device
from data_loader import load_data, prepare_data, get_data_loaders
from models import get_model
from train import train_model, predict
from evaluate import evaluate_model, compare_models


class ExperimentLogger:
    """Logger para salvar informações dos experimentos em arquivo TXT"""

    def __init__(self, output_dir, dataset_name):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"experiment_log_{dataset_name}_{timestamp}.txt"

        self.log_lines = []
        self.start_time = datetime.now()

    def log(self, message, print_console=True):
        """Adiciona mensagem ao log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        self.log_lines.append(line)

        if print_console:
            print(message)

    def save(self):
        """Salva o log em arquivo"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.log_lines))
        print(f"\nLog salvo em: {self.log_file}")

    def log_section(self, title):
        """Adiciona uma seção ao log"""
        separator = "=" * 80
        self.log(f"\n{separator}")
        self.log(title)
        self.log(separator)


def run_single_experiment(model_type, arch_name, window_size, train_loader, val_loader, dataset_config, logger):
    """Executa um único experimento com parâmetros específicos"""
    
    arch_params = config.ARCHITECTURES[arch_name]
    model_full_name = f"{model_type.upper()}_{arch_name}_W{window_size}"
    
    logger.log(f"Iniciando: {model_full_name}")
    
    # Criar modelo
    model = get_model(
        model_type,
        hidden_size=arch_params['hidden_size'],
        num_layers=arch_params['num_layers'],
        dropout=arch_params['dropout'],
        forecast_horizon=dataset_config['forecast_horizon']
    )

    # Treinar
    model, history, training_time = train_model(
        model, train_loader, val_loader, model_full_name,
        epochs=dataset_config['epochs'],
        learning_rate=dataset_config['learning_rate']
    )

    # Fazer previsões
    device = get_device()
    predictions, targets = predict(model, val_loader, device)

    # Avaliar
    metrics = evaluate_model(
        model, predictions, targets, history, training_time, model_full_name
    )
    
    # Adicionar metadados
    metrics['model_type'] = model_type.upper()
    metrics['architecture'] = arch_name
    metrics['window_size'] = window_size
    metrics['hidden_size'] = arch_params['hidden_size']
    metrics['num_layers'] = arch_params['num_layers']

    return metrics, model


def run_dataset_experiments(dataset_name, logger):
    """Executa todos os experimentos para um dataset"""
    
    dataset_config = get_dataset_config(dataset_name)
    logger.log_section(f"DATASET: {dataset_name}")
    
    # Carregar dados uma vez
    train_file_path = dataset_config['data_dir'] / dataset_config['train_file']
    if not train_file_path.exists():
        logger.log(f"ERRO: Arquivo não encontrado {train_file_path}")
        return None

    df_full = load_data(
        train_file_path,
        date_column=dataset_config['date_column'],
        target_column=dataset_config['target_column'],
        group_columns=dataset_config.get('group_columns')
    )

    all_metrics = []
    
    # Loop sobre tamanhos de janela
    for window_size in config.WINDOW_SIZES:
        logger.log_section(f"WINDOW SIZE: {window_size} dias")
        
        # Preparar dados para esta janela
        X_train, y_train, X_val, y_val, scaler = prepare_data(
            df_full,
            window_size=window_size,
            forecast_horizon=dataset_config['forecast_horizon'],
            validation_split=dataset_config['validation_split'],
            use_subset=dataset_config['use_subset'],
            subset_filter=dataset_config.get('subset_filter'),
            date_column=dataset_config['date_column'],
            target_column=dataset_config['target_column']
        )
        
        train_loader, val_loader = get_data_loaders(
            X_train, y_train, X_val, y_val,
            batch_size=dataset_config['batch_size']
        )
        
        # Loop sobre tipos de modelo
        for model_type in ['lstm', 'gru']:
            # Loop sobre arquiteturas
            for arch_name in config.ARCHITECTURES.keys():
                
                run_name = f"{model_type.upper()}_{arch_name}_W{window_size}_{dataset_name}"
                
                with mlflow.start_run(run_name=run_name):
                    # Log parâmetros no MLflow
                    mlflow.log_params({
                        "model_type": model_type.upper(),
                        "architecture": arch_name,
                        "window_size": window_size,
                        "dataset": dataset_name,
                        **config.ARCHITECTURES[arch_name]
                    })
                    
                    # Executar
                    metrics, model = run_single_experiment(
                        model_type, arch_name, window_size, 
                        train_loader, val_loader, dataset_config, logger
                    )
                    
                    # Log métricas no MLflow
                    mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
                    mlflow.pytorch.log_model(model, "model")
                    
                    all_metrics.append(metrics)
                    
    return all_metrics


def main():
    set_seed()
    
    # Configurar MLflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    
    available_datasets = list_available_datasets()
    all_dataset_metrics = []
    
    for dataset_name in available_datasets:
        logger = ExperimentLogger(config.RESULTS_DIR, dataset_name)
        
        try:
            metrics_list = run_dataset_experiments(dataset_name, logger)
            if metrics_list:
                all_dataset_metrics.extend(metrics_list)
        except Exception as e:
            logger.log(f"Erro fatal: {str(e)}")
            import traceback
            logger.log(traceback.format_exc())
        finally:
            logger.save()
            
    # Salvar resultados consolidados
    if all_dataset_metrics:
        df_results = pd.DataFrame(all_dataset_metrics)
        results_path = config.RESULTS_DIR / "comprehensive_comparison.csv"
        df_results.to_csv(results_path, index=False)
        print(f"\nResultados salvos em: {results_path}")
        
        # Gerar relatório resumo em TXT
        summary_path = config.RESULTS_DIR / "summary_report.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE COMPARAÇÃO EXAUSTIVA: LSTM vs GRU\n")
            f.write("="*50 + "\n\n")
            
            # Melhores por métrica
            f.write("TOP 5 MODELOS POR RMSE:\n")
            f.write(df_results.sort_values('RMSE').head(5)[['model_type', 'architecture', 'window_size', 'RMSE', 'MAE']].to_string() + "\n\n")
            
            # Comparação média por tipo de modelo
            f.write("MÉDIAS POR TIPO DE MODELO:\n")
            f.write(df_results.groupby('model_type')[['RMSE', 'MAE', 'training_time']].mean().to_string() + "\n\n")
            
            # Comparação por arquitetura
            f.write("MÉDIAS POR ARQUITETURA E MODELO:\n")
            f.write(df_results.groupby(['architecture', 'model_type'])[['RMSE', 'MAE', 'training_time']].mean().to_string() + "\n\n")

            # Comparação por janela
            f.write("MÉDIAS POR JANELA (WINDOW SIZE):\n")
            f.write(df_results.groupby('window_size')[['RMSE', 'MAE']].mean().to_string() + "\n\n")
            
        print(f"Relatório resumo salvo em: {summary_path}")
        
        # Mostrar melhores modelos no console
        print("\nTOP 5 MODELOS (por RMSE):")
        print(df_results.sort_values('RMSE').head(5)[['model_type', 'architecture', 'window_size', 'RMSE', 'MAE']])


if __name__ == "__main__":
    main()
