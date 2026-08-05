# pos-tech-fiap-ia-dev-fase-1

Tech Challenge de Machine Learning — Classificação de Riscos à Saúde da Mulher (Síndrome dos Ovários Policísticos).

## Estrutura do repositório

```text
.
├── base_dados/                # Dados brutos
├── base_dados_tratada/        # Dados limpos gerados pelo ETL
├── reports/                   # Métricas e relatórios gerados automaticamente
├── src/
│   ├── etl/
│   │   ├── etapa1_diagnostico.py   # Verificações de qualidade de dados
│   │   └── etapa2_tratamento.py    # Limpeza, conversão e imputação
│   ├── modelagem/
│   │   ├── __init__.py
│   │   └── treinamento.py          # Split, treino e avaliação de modelos
│   ├── main.py                     # Executa ETL (ETAPA 1 + ETAPA 2)
│   └── run_ml.py                   # Executa ETAPA 3 (treinamento/aval.)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Configuração do ambiente

Instale as bibliotecas listadas em `requirements.txt`:

```bash
pip install -r requirements.txt
```

Dependências principais:

- `pandas` / `numpy`: manipulação de dados
- `scikit-learn`: modelos de classificação e métricas
- `matplotlib` / `seaborn`: visualização
- `jupyterlab`: notebooks exploratórios
- `openpyxl`: leitura de arquivos `.xlsx`

## Execução passo a passo

### 1. Diagnóstico e tratamento (ETAPAS 1 e 2)

```bash
python -m src.main
```

Gera os arquivos limpos em `base_dados_tratada/`.

### 2. Treinamento e avaliação (ETAPA 3)

```bash
python -m src.run_ml
```

O script:

1. Carrega `base_dados_tratada/PCOS_data_without_infertility.xlsx`.
2. Separa 80% dos dados para treino e 20% para teste (`train_test_split` com `stratify`).
3. Treina cinco classificadores:
   - Regressão Logística
   - Árvore de Decisão
   - KNN (k=5)
   - Random Forest
   - SVM (RBF)
4. Avalia cada um com acurácia, precisão, recall, F1, AUC-ROC e matriz de confusão.
5. Salva os resultados em `reports/metricas_modelos.txt` e `.json`.

## Resultados esperados

Exemplo de saída em `reports/metricas_modelos.txt`:

```text
Modelo: Regressão Logística
  Acurácia : 0.8624
  Precisão : 0.7561
  Recall   : 0.8611
  F1-Score : 0.8052
  AUC-ROC  : 0.9288
  Matriz de confusão: [[63, 10], [5, 31]]
```

## Variável alvo

- `PCOS (Y/N)`: `1` indica presença de SOP, `0` indica ausência.
