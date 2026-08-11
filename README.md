# pos-tech-fiap-ia-dev-fase-1

Tech Challenge de Machine Learning — Classificação de Riscos à Saúde da Mulher (Síndrome dos Ovários Policísticos).

## Estrutura do repositório

```text
.
├── base_dados/                # Dados brutos (.xlsx e .csv)
├── base_dados_tratada/        # Dados limpos gerados pelo notebook
├── reports/                   # Métricas geradas automaticamente (não versionado)
├── notebooks/
│   └── pipeline_pcos.ipynb    # ETL completo + EDA + treinamento e avaliação
├── src/
│   ├── modelagem/
│   │   ├── __init__.py
│   │   └── treinamento.py     # Script de treinamento e avaliação
│   ├── main.py                # Mantido para compatibilidade; não executável sem src/etl
│   └── run_ml.py              # Ponto de entrada do treinamento via terminal
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

### Opção A — via notebook (recomendada)

O notebook `notebooks/pipeline_pcos.ipynb` contém o pipeline completo:

1. **ETAPA 1 — Diagnóstico:** 13 verificações de qualidade (somente leitura).
2. **ETAPA 1.B — Consolidação:** junção dos arquivos pela chave `Sl. No`.
3. **ETAPA 1.C — EDA:** exploração e análise dos dados.
4. **ETAPA 2 — Tratamento:** 14 correções e exportação para `base_dados_tratada/PCOS_unificado.csv`.
5. **ETAPA 3 — Modelagem:** split 80/20, treino e avaliação de cinco classificadores.

Abra com:

```bash
python -m jupyter lab notebooks/pipeline_pcos.ipynb
```

Execute as células em ordem.

### Opção B — via script

Se a base tratada já foi gerada (`base_dados_tratada/PCOS_unificado.csv`), rode diretamente:

```bash
python -m src.run_ml
```

Ou:

```bash
python src/run_ml.py
```

O script `src/modelagem/treinamento.py` carrega o CSV unificado, separa 80% treino / 20% teste (`stratify=y`), treina e avalia:

- Regressão Logística
- Árvore de Decisão
- KNN (k=5)
- Random Forest
- SVM (RBF)

As métricas são salvas em `reports/metricas_modelos.txt` e `.json`.

## Resultados esperados

Exemplo de saída (`reports/metricas_modelos.txt`):

```text
Modelo: Regressão Logística
  Acurácia : 0.8532
  Precisão : 0.7381
  Recall   : 0.8611
  F1-Score : 0.7949
  AUC-ROC  : 0.9277
  Matriz de confusão: [[62, 11], [5, 31]]
```

## Variável alvo

- `PCOS (Y/N)`: `1` indica presença de SOP, `0` indica ausência.
