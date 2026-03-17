"""
Orquestrador de experimentos para comparação LSTM vs GRU
"""
import torch
import mlflow
import mlflow.pytorch
import pandas as pd
from src import config
from src.data.dataset_configs import get_dataset_config
from src.utils.utils import get_device
from src.data.data_loader import load_data, prepare_data, get_data_loaders
from src.models.models import get_model
from src.training.train import train_model, predict
from src.evaluation.evaluate import evaluate_model


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


def run_dataset_experiments(dataset_name, model_types, logger):
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
        group_columns=dataset_config.get('group_columns'),
        date_format=dataset_config.get('date_format')
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
        for model_type in model_types:
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
                    
                    # Log métricas formatadas
                    logger.log_metrics(run_name, metrics)
                    
                    # Log métricas no MLflow
                    mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
                    mlflow.pytorch.log_model(model, "model")
                    
                    all_metrics.append(metrics)
                    
    return all_metrics


def run_best_comparison(dataset_name, logger):
    """Compara os melhores modelos LSTM e GRU baseados em resultados anteriores"""
    
    results_csv = config.RESULTS_DIR / "comprehensive_comparison.csv"
    if not results_csv.exists():
        logger.log(f"ERRO: {results_csv} não encontrado. Execute experimentos primeiro.")
        return

    df = pd.read_csv(results_csv)
    
    # Filtrar melhor LSTM
    lstms = df[df['model_type'] == 'LSTM'].sort_values('RMSE')
    if lstms.empty:
        logger.log("ERRO: Nenhuma LSTM encontrada nos resultados.")
        return
    best_lstm_row = lstms.iloc[0]
    
    # Filtrar melhor GRU
    grus = df[df['model_type'] == 'GRU'].sort_values('RMSE')
    if grus.empty:
        logger.log("ERRO: Nenhuma GRU encontrada nos resultados.")
        return
    best_gru_row = grus.iloc[0]
    
    dataset_config = get_dataset_config(dataset_name)
    logger.log_section("COMPARAÇÃO DOS MELHORES MODELOS")
    
    # Carregar dados
    train_file_path = dataset_config['data_dir'] / dataset_config['train_file']
    df_full = load_data(
        train_file_path, 
        dataset_config['date_column'], 
        dataset_config['target_column'],
        date_format=dataset_config.get('date_format')
    )
    
    results = []
    
    for row in [best_lstm_row, best_gru_row]:
        model_type = row['model_type'].lower()
        arch_name = row['architecture']
        window_size = int(row['window_size'])
        
        logger.log_section(f"MELHOR {model_type.upper()} | ARCH: {arch_name} | WINDOW: {window_size}")
        
        # Preparar dados
        X_train, y_train, X_val, y_val, scaler = prepare_data(
            df_full, window_size=window_size, 
            forecast_horizon=dataset_config['forecast_horizon'],
            subset_filter=dataset_config.get('subset_filter'),
            target_column=dataset_config['target_column']
        )
        
        train_loader, val_loader = get_data_loaders(X_train, y_train, X_val, y_val)
        
        arch_params = config.ARCHITECTURES[arch_name]
        model = get_model(
            model_type,
            hidden_size=arch_params['hidden_size'],
            num_layers=arch_params['num_layers'],
            dropout=arch_params['dropout'],
            forecast_horizon=dataset_config['forecast_horizon']
        )
        
        model_name = f"BEST_{model_type.upper()}_{arch_name}_W{window_size}"
        
        # Treinar
        model, history, training_time = train_model(
            model, train_loader, val_loader, model_name,
            epochs=dataset_config['epochs'],
            learning_rate=dataset_config['learning_rate']
        )
        
        # Avaliar
        device = get_device()
        predictions, targets = predict(model, val_loader, device)
        metrics = evaluate_model(model, predictions, targets, history, training_time, model_name)
        
        logger.log_metrics(model_name, metrics)
        results.append(metrics)
        
    return results
