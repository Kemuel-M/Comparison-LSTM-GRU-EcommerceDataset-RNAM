"""
Módulo Orquestrador de Experimentos.
Gerencia o fluxo de alto nível: carregamento de dados, loops de hiperparâmetros,
integração com MLflow e consolidação de métricas comparativas.
"""
import torch
import mlflow
import mlflow.pytorch
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from torch.utils.data import DataLoader
from src import config
from src.data.dataset_configs import get_dataset_config
from src.utils.utils import get_device, ExperimentLogger
from src.data.data_loader import load_data, prepare_data, get_data_loaders
from src.models.models import get_model
from src.training.train import train_model, predict
from src.evaluation.evaluate import evaluate_model


def run_single_experiment(
    model_type: str, 
    arch_name: str, 
    window_size: int, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    dataset_config: Dict[str, Any], 
    logger: ExperimentLogger,
    save_models: bool = False
) -> Tuple[Dict[str, Any], torch.nn.Module]:
    """Executa um ciclo completo (treino/val/avaliação) para uma única configuração.

    Args:
        model_type (str): Tipo de arquitetura ('lstm', 'gru').
        arch_name (str): Nome da arquitetura pré-definida em config (ex: 'medium').
        window_size (int): Tamanho da janela temporal de entrada.
        train_loader (DataLoader): Carregador de dados de treino.
        val_loader (DataLoader): Carregador de dados de validação.
        dataset_config (Dict[str, Any]): Dicionário com parâmetros do dataset.
        logger (ExperimentLogger): Utilitário de log customizado.
        save_models (bool): Se True, salva o modelo em disco após o treinamento.

    Returns:
        Tuple[Dict[str, Any], torch.nn.Module]: Par contendo o dicionário 
            de métricas calculadas e o objeto do modelo PyTorch.
    """
    arch_params = config.ARCHITECTURES[arch_name]
    model_full_name = f"{model_type.upper()}_{arch_name}_W{window_size}"
    
    logger.log(f"Iniciando: {model_full_name}")
    
    # Instanciar modelo via Factory
    model = get_model(
        model_type,
        hidden_size=arch_params['hidden_size'],
        num_layers=arch_params['num_layers'],
        dropout=arch_params['dropout'],
        forecast_horizon=dataset_config['forecast_horizon']
    )

    # Treinamento
    model, history, training_time = train_model(
        model, train_loader, val_loader, model_full_name,
        epochs=dataset_config['epochs'],
        learning_rate=dataset_config['learning_rate']
    )

    # Salvar modelo em disco se solicitado
    if save_models:
        model_path = config.MODELS_DIR / f"{model_full_name}.pt"
        torch.save(model.state_dict(), model_path)
        logger.log(f"Modelo salvo em: {model_path}")

    # Inferência para avaliação
    device = get_device()
    predictions, targets = predict(model, val_loader, device)

    # Cálculo de métricas e geração de gráficos
    metrics = evaluate_model(
        model, predictions, targets, history, training_time, model_full_name
    )
    
    # Enriquecer métricas com metadados do experimento
    metrics['model_type'] = model_type.upper()
    metrics['architecture'] = arch_name
    metrics['window_size'] = window_size
    metrics['hidden_size'] = arch_params['hidden_size']
    metrics['num_layers'] = arch_params['num_layers']

    return metrics, model


def run_dataset_experiments(
    dataset_name: str, 
    model_types: List[str], 
    logger: ExperimentLogger,
    save_models: bool = False
) -> Optional[List[Dict[str, Any]]]:
    """Executa a bateria exaustiva de experimentos para um determinado dataset.

    Percorre todas as combinações de tipos de modelo, arquiteturas e tamanhos 
    de janela, registrando tudo no MLflow e no logger local.

    Args:
        dataset_name (str): Identificador do dataset (ex: 'demand_forecasting').
        model_types (List[str]): Lista de modelos a testar (ex: ['lstm', 'gru']).
        logger (ExperimentLogger): Utilitário de log.
        save_models (bool): Se True, salva os modelos em disco.

    Returns:
        Optional[List[Dict[str, Any]]]: Lista consolidada de métricas de todos 
            os experimentos realizados, ou None em caso de erro crítico.
    """
    dataset_config = get_dataset_config(dataset_name)
    logger.log_section(f"DATASET: {dataset_name}")
    
    # Carregar dados brutos apenas uma vez por dataset para otimizar tempo
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
    
    # Loop externo: Janelas temporais (Impacto na preparação dos dados)
    for window_size in config.WINDOW_SIZES:
        logger.log_section(f"WINDOW SIZE: {window_size} dias")
        
        # Preparar dados para esta janela específica
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
        
        # Loop intermediário: LSTM vs GRU
        for model_type in model_types:
            # Loop interno: Hiperparâmetros de arquitetura (Small, Medium, etc)
            for arch_name in config.ARCHITECTURES.keys():
                
                run_name = f"{model_type.upper()}_{arch_name}_W{window_size}_{dataset_name}"
                
                # Gerenciamento de execução via MLflow
                with mlflow.start_run(run_name=run_name):
                    # Rastreamento de hiperparâmetros
                    mlflow.log_params({
                        "model_type": model_type.upper(),
                        "architecture": arch_name,
                        "window_size": window_size,
                        "dataset": dataset_name,
                        **config.ARCHITECTURES[arch_name]
                    })
                    
                    # Executar o experimento
                    metrics, model = run_single_experiment(
                        model_type, arch_name, window_size, 
                        train_loader, val_loader, dataset_config, logger,
                        save_models=save_models
                    )
                    
                    # Logs locais formatados
                    logger.log_metrics(run_name, metrics)
                    
                    # Sincronização de métricas e persistência do modelo no MLflow
                    numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, complex))}
                    mlflow.log_metrics(numeric_metrics)
                    mlflow.pytorch.log_model(model, "model")
                    
                    all_metrics.append(metrics)
                    
    return all_metrics


def run_best_comparison(
    dataset_name: str, 
    logger: ExperimentLogger,
    model_types: List[str] = ['lstm', 'gru'],
    save_models: bool = False
) -> Optional[List[Dict[str, Any]]]:
    """Identifica e re-executa os melhores modelos para uma comparação final direta.

    Lê os resultados salvos em CSV, seleciona o melhor de cada tipo solicitado
    baseado no RMSE e executa uma rodada final de validação.

    Args:
        dataset_name (str): Nome do dataset.
        logger (ExperimentLogger): Utilitário de log.
        model_types (List[str]): Lista de tipos ('lstm', 'gru') para comparar.
        save_models (bool): Se True, salva os modelos treinados em disco.

    Returns:
        Optional[List[Dict[str, Any]]]: Lista com métricas dos melhores modelos selecionados.
    """
    results_csv = config.RESULTS_DIR / "comprehensive_comparison.csv"
    if not results_csv.exists():
        logger.log(f"ERRO: {results_csv} não encontrado. Execute experimentos primeiro.")
        return None

    df = pd.read_csv(results_csv)
    best_rows = []
    
    for m_type in model_types:
        filtered = df[df['model_type'] == m_type.upper()].sort_values('RMSE')
        if not filtered.empty:
            best_rows.append(filtered.iloc[0])
        else:
            logger.log(f"AVISO: Nenhum modelo do tipo {m_type.upper()} encontrado para comparar.")
    
    if not best_rows:
        return None
    
    dataset_config = get_dataset_config(dataset_name)
    logger.log_section("COMPARAÇÃO FINAL: TOP PERFORMERS")
    
    # Recarregar dados para a rodada final
    train_file_path = dataset_config['data_dir'] / dataset_config['train_file']
    df_full = load_data(
        train_file_path, 
        dataset_config['date_column'], 
        dataset_config['target_column'],
        date_format=dataset_config.get('date_format')
    )
    
    results = []
    
    # Rodar os vencedores selecionados
    for row in best_rows:
        model_type = row['model_type'].lower()
        arch_name = row['architecture']
        window_size = int(row['window_size'])
        
        logger.log_section(f"MELHOR {model_type.upper()} | ARCH: {arch_name} | WINDOW: {window_size}")
        
        # Preparação de dados específica para o modelo
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
        
        # Treinamento e Avaliação final
        model, history, training_time = train_model(
            model, train_loader, val_loader, model_name,
            epochs=dataset_config['epochs'],
            learning_rate=dataset_config['learning_rate']
        )
        
        # Salvar modelo em disco se solicitado
        if save_models:
            model_path = config.MODELS_DIR / f"{model_name}.pt"
            torch.save(model.state_dict(), model_path)
            logger.log(f"Modelo salvo em: {model_path}")
        
        device = get_device()
        predictions, targets = predict(model, val_loader, device)
        metrics = evaluate_model(model, predictions, targets, history, training_time, model_name)
        
        logger.log_metrics(model_name, metrics)
        results.append(metrics)
        
    return results
