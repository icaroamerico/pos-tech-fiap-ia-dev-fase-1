# pos-tech-fiap-ia-dev-fase-1

Pipeline de diagnóstico e tratamento do dataset PCOS (síndrome dos ovários policísticos).

## Estrutura

```
base_dados/                       arquivos de entrada (.xlsx e .csv)
base_dados_tratada/               saída: PCOS_unificado.csv (541 x 49)
notebooks/pipeline_pcos.ipynb     o pipeline completo
requirements.txt
```

## Como rodar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/pipeline_pcos.ipynb
```

## O que o notebook faz

| Etapa | Conteúdo | Altera os dados? |
|---|---|---|
| **1 — Diagnóstico** | 13 verificações de qualidade + exploração (EDA) | não |
| **1.B — Consolidação** | junção dos dois arquivos pela chave `Sl. No` | sim |
| **2 — Tratamento** | 14 correções na ordem correta + exportação | sim |

Não inclui divisão treino/teste, balanceamento, normalização nem treino de modelos:
isso é etapa de modelagem e pertence a um notebook seguinte.
