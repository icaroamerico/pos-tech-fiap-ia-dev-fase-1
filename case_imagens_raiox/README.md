# Case — classificação de raio-X de tórax (NORMAL × PNEUMONIA)

Dois ataques ao mesmo problema, para efeito de comparação:

| Notebook | Modelo | Atributos |
|---|---|---|
| `aprendizadado-maquina-imagem.ipynb` | MLP (`scikit-learn`) | vetor artesanal (mediana + Sobel) → PCA |
| `cnn-raiox-torax.ipynb` | CNN (`PyTorch`) | pixels crus, a rede aprende os filtros |

Complementos:

- `validacao-por-paciente.ipynb` — experimento que descarta a hipótese de vazamento
  por radiografias repetidas do mesmo paciente.
- `REPASSE.md` — repasse detalhado do notebook da MLP: o que foi corrigido, os
  resultados e por que a validação interna não prevê o teste.

## Os dados não estão aqui

O dataset tem ~1,2 GB e fica de fora do repositório de propósito (`.gitignore`).

Origem: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), Kaggle.

Estrutura esperada:

```
<PASTA_DO_DATASET>/
├── train/  NORMAL 1349 | PNEUMONIA 3883   (5232 imagens)
└── test/   NORMAL  234 | PNEUMONIA  390   ( 624 imagens)
```

As 16 imagens do `val/` original foram movidas para dentro do `train/` — eram
poucas demais para escolher hiperparâmetro contra elas. O papel de validação é
feito por um recorte de 20% do próprio `train/`, sorteado dentro do notebook.

Os dois notebooks têm uma constante de caminho na primeira célula de código
(`PASTA_DADOS` / `DIRETORIO_RAIZ`). Aponte-a para onde o dataset estiver.

## Como rodar

```bash
source /Users/icaroamerico/Documents/venv_icaro/bin/activate
jupyter lab case_imagens_raiox/
```

**Selecione o kernel na mão: "Python (venv_icaro)".** Ambiente: Python 3.12,
numpy 2.5, opencv-python 4.13, scikit-learn 1.9, torch 2.12 (com MPS), matplotlib 3.11.
