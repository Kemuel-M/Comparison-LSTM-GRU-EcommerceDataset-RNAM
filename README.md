# Comparative Analysis of RNN Architectures for E-commerce Demand Forecasting

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0+-0194E2.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Este repositório apresenta um estudo técnico aprofundado comparando arquiteturas de Redes Neurais Recorrentes (**LSTM** e **GRU**) aplicadas à previsão de demanda em larga escala para o setor de E-commerce. O projeto utiliza um framework modular e extensível para benchmarking exaustivo, integrando pipelines de dados robustos e rastreamento de experimentos com **MLflow**.

---

## 📋 Resumo Executivo

A previsão precisa de demanda é um pilar crítico para a otimização de estoque e logística no E-commerce. Este projeto investiga o desempenho de diferentes variações de RNNs, avaliando não apenas a precisão (RMSE/MAE), mas também a eficiência computacional e a capacidade de captura de sazonalidade através de múltiplas janelas temporais (*look-back windows*).

**Principais conclusões:**
*   **GRU superou LSTM** consistentemente em termos de erro médio (RMSE) e tempo de convergência.
*   Modelos com **janelas temporais mais longas (180-365 dias)** demonstraram maior resiliência em padrões sazonais complexos.
*   Arquiteturas **Extra Large (4 camadas)** apresentaram os melhores resultados, indicando que a profundidade beneficia a extração de features em séries temporais de alta volatilidade.

---

## 🚀 Destaques Técnicos

*   **Arquiteturas Flexíveis**: Implementações customizadas de LSTM, GRU e suas variantes bidirecionais (Bi-LSTM, Bi-GRU) em PyTorch.
*   **Cabeçote MLP**: Integração de camadas densas (Multi-Layer Perceptron) pós-recorrência para aumentar o poder de representação não-linear.
*   **Benchmarking Automatizado**: Script de orquestração que executa dezenas de combinações (Modelo x Arquitetura x Janela) de forma autônoma.
*   **Métricas Profissionais**: Avaliação baseada em RMSE, MAE e tempo de treinamento, com log detalhado de cada run.
*   **MLOps Ready**: Integração total com MLflow para versionamento de modelos, parâmetros e métricas.

---

## 🛠️ Arquitetura do Sistema

O projeto foi construído seguindo princípios de **Clean Code** e **Modularidade**, facilitando a manutenção e escala:

```text
├── src/
│   ├── data/           # Pipeline de ETL e Data Loading (PyTorch Datasets/DataLoader)
│   ├── models/         # Definições de classes das Redes Neurais
│   ├── training/       # Lógica de treino, Early Stopping e otimização
│   ├── evaluation/     # Métricas de performance e gerador de gráficos
│   ├── experiments/    # Orquestrador de benchmarking exaustivo
│   └── utils/          # Helpers e configurações globais
├── reports/            # Relatórios consolidados e visualizações de resultados
└── tests/              # Testes unitários para garantia de integridade
```

---

## 📊 Resultados e Insights

Os experimentos foram realizados utilizando o dataset de *Demand Forecasting* (Kaggle), comparando 5 variações de profundidade (Small a Huge) e 4 tamanhos de janela (30 a 365 dias).

### Performance Geral (Média)

| Modelo | RMSE Médio | MAE Médio | Tempo de Treino (s) |
| :--- | :---: | :---: | :---: |
| **GRU** | **0.7593** | **0.6135** | **14.80** |
| LSTM | 0.8108 | 0.6479 | 20.37 |

### Top 3 Modelos (Líderes de Precisão)

1.  **GRU (Extra Large)** - Janela: 180 dias | **RMSE: 0.7342**
2.  **LSTM (Extra Large)** - Janela: 180 dias | **RMSE: 0.7345**
3.  **GRU (Extra Large)** - Janela: 90 dias | **RMSE: 0.7357**

### Análise de Janela Temporal
Modelos treinados com janelas de **90 e 180 dias** atingiram o equilíbrio ideal entre custo computacional e precisão, capturando tendências trimestrais e semestrais fundamentais para o comportamento de vendas.

---

## ⚙️ Configuração do Ambiente

1.  **Clonar o repositório**:
    ```bash
    git clone https://github.com/seu-usuario/Comparison-LSTM-GRU-EcommerceDataset-RNAM.git
    cd Comparison-LSTM-GRU-EcommerceDataset-RNAM
    ```

2.  **Instalar dependências**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Executar o Benchmarking**:
    ```bash
    python main.py
    ```

4.  **Visualizar no MLflow**:
    ```bash
    mlflow ui --backend-store-uri ./mlruns
    ```

---

## 🧪 Metodologia Científica

*   **Data Split**: 80% treino / 20% validação (mantendo a ordem temporal).
*   **Otimização**: Adam Optimizer com Learning Rate Schedulers.
*   **Regularização**: Dropout dinâmico por arquitetura e Early Stopping para prevenir overfitting.
*   **Hardware**: Suporte total a aceleração por GPU (CUDA).

---

## 👨‍💻 Autor

**Kemuel Marvila** - [LinkedIn](https://www.linkedin.com/in/kemuel-marvila-1a147b288/) - [GitHub](https://github.com/Kemuel-M)
