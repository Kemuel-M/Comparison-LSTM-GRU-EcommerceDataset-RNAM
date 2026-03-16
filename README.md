# Previsão de Séries Temporais: LSTM vs GRU

Sistema modular avançado para comparação de modelos LSTM e GRU em previsão de demanda, com análise de múltiplas arquiteturas e janelas temporais.

## 🚀 Novidades da Versão 3.0

- **Janelas Estendidas**: Suporte a janelas de 30, 90, 180 (6 meses) e 365 (1 ano) dias.
- **Arquiteturas Variadas**: Comparação automática entre modelos 'small', 'medium' e 'large'.
- **Poder de Representação**: Cabeçalho MLP (Multi-Layer Perceptron) integrado aos modelos para melhor captura de padrões complexos.
- **Comparação Automática**: Loop exaustivo que testa todas as combinações de (Janela x Modelo x Arquitetura).
- **Consolidação de Resultados**: Geração automática de `comprehensive_comparison.csv` com ranking dos melhores modelos.

## 📁 Estrutura do Projeto

```
projeto_series_temporais/
├── src/                    # Código fonte
│   ├── main.py            # Script principal
│   ├── config.py          # Configurações de Janelas e Arquiteturas
│   ├── dataset_configs.py # Configurações por dataset
│   ├── models.py          # Modelos LSTM e GRU com MLP Head
│   ├── train.py           # Treinamento com Early Stopping
│   ├── evaluate.py        # Avaliação e Geração de Gráficos
│   ├── data_loader.py     # Preparação de Sequências
│   └── utils.py           # Utilitários e Métricas
```

## ⚙️ Configurações Experimentais (v3.0)

### Janelas Temporais (`WINDOW_SIZES`)
- **1 Mês**: 30 dias
- **3 Meses**: 90 dias
- **6 Meses**: 180 dias
- **1 Ano**: 365 dias (Essencial para padrões anuais)

### Arquiteturas (`ARCHITECTURES`)
- **Small**: Hidden 64, 1 Layer, Dropout 0.1
- **Medium**: Hidden 128, 2 Layers, Dropout 0.2
- **Large**: Hidden 256, 3 Layers, Dropout 0.3

## 📊 Como Visualizar os Resultados

### Tabela Completa
Os resultados de todos os experimentos são salvos em:
`results/comprehensive_comparison.csv`

### Ranking de Modelos
Ao final da execução, o script exibe o **TOP 5** modelos baseados no RMSE.

### MLflow UI
Para uma comparação visual e detalhada de todos os runs:
```bash
mlflow ui --backend-store-uri ./mlruns
```

## 🎓 Para o Relatório Científico

permite uma análise estatística muito mais rica, possibilitando responder:
1. Qual o impacto do tamanho da janela na precisão? (Janelas longas de 1 ano capturam melhor a sazonalidade?)
2. Qual arquitetura oferece o melhor trade-off entre precisão e tempo de treino?
3. GRU continua superando LSTM em redes mais profundas (Large)?

## 🔬 Tecnologias

- Python 3.x
- PyTorch (CUDA enabled)
- MLflow
- Pandas, NumPy, Scikit-learn
- Matplotlib

---


