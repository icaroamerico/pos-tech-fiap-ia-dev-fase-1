# Classificação de raio-X de tórax com MLP — repasse

Pasta: `/Users/icaroamerico/Documents/imagens_fase_1_pos_fiap`
Notebook: `aprendizadado-maquina-imagem.ipynb`
Última execução completa: 10/08/2026, ~1min06 de ponta a ponta.

---

## 1. O que tem nesta pasta

```
imagens_fase_1_pos_fiap/
├── aprendizadado-maquina-imagem.ipynb   ← notebook principal, executado
├── validacao-por-paciente.ipynb         ← experimento: por que validação ≠ teste
├── REPASSE.md                           ← este arquivo
├── train/  NORMAL 1349 | PNEUMONIA 3883   (5232 imagens)
└── test/   NORMAL  234 | PNEUMONIA  390   ( 624 imagens)
```

Origem: cópia de `~/Downloads/chest_xray` (o original continua lá, intocado).
Na cópia foram descartados o diretório `__MACOSX` e uma duplicata idêntica do
dataset que estava aninhada em `chest_xray/chest_xray` — juntos, 1,2 GB de nada.

Sobraram dois diretórios `val/` vazios; podem ser apagados no Finder.

## 2. Como o dataset foi separado

Mantive a divisão original `train`/`test` do dataset (é a divisão canônica do
Kaggle, o que deixa o resultado comparável ao de outras pessoas).

A pasta `val/` original tinha **8 + 8 imagens** — pequena demais para escolher
hiperparâmetro contra ela. Suas 16 imagens foram movidas para dentro de `train/`,
e o papel de validação passou a ser feito por um recorte de 20% do próprio
`train/`, sorteado dentro do notebook. Daí os três conjuntos:

| conjunto | imagens | de onde vem | para que serve |
|---|---|---|---|
| treino | 4186 | 80% de `train/` | ajustar os pesos da MLP |
| validação | 1046 | 20% de `train/` | conferir o modelo durante o desenvolvimento |
| teste | 624 | `test/` inteiro | estimativa final e imparcial |

## 3. O que mudou no notebook (e por quê)

O original treinava e avaliava usando **só** a pasta `train/`, e tinha alguns
problemas que teriam custado nota. Mudei o mínimo necessário para corrigir:

**a. Rótulos viraram o nome da classe, não o caminho.**
Antes: `alvos.append(cls)` guardava `/Users/.../train/NORMAL`. Ao acrescentar a
pasta de teste, os rótulos virariam `/Users/.../test/NORMAL` — string diferente.
O modelo seria treinado contra dois rótulos e comparado contra outros dois, dando
**100% de erro sem levantar exceção nenhuma**: pareceria modelo ruim, seria bug.
Agora o alvo é `'NORMAL'`/`'PNEUMONIA'`.

**b. A divisão treino/validação passou para antes da normalização.**
Antes, `media`, `desvio` e o `PCA.fit_transform` eram calculados sobre a matriz
inteira, e só depois vinha a separação. As amostras de "validação" ajudavam a
definir a normalização e as componentes principais — vazamento de dados, que
infla o resultado. Agora a divisão vem primeiro e o PCA é ajustado só no treino.

**c. `transform` em vez de `fit_transform` para validação e teste.**
A mesma média, o mesmo desvio e as mesmas componentes principais são reaplicados
nos outros conjuntos. Sem isso, teste e treino não viveriam no mesmo espaço de
atributos.

**d. Resolução caiu de 500×500 para 64×64.**
Este é o que impedia o notebook de rodar. Com 500×500 são 250.000 atributos por
imagem: na hora da normalização, `(treinamento - media)/desvio` promove tudo para
float64 e a matriz vira **~10 GB por cópia, ~21 GB no pico**. Esta máquina tem
8 GB — travava na normalização, antes mesmo do PCA. Com 64×64 são 4.096 atributos,
e o dataset inteiro roda com pico de menos de 1 GB.

**e. Filtro de extensão no carregamento.**
`os.listdir` sem filtro devolveria `.DS_Store`, e `cv2.imread` nele retorna `None`,
quebrando o pré-processamento. Agora só entram arquivos `.jpeg`.

**f. Proteção contra desvio zero.**
`desvio[desvio == 0] = 1`. Pixel constante (borda preta) daria divisão por zero →
NaN → PCA falha com erro difícil de ler. No dataset completo não ocorre, mas a
normalização agora é ajustada sobre um sorteio de 80%, onde é mais provável.

**g. Semente fixa** (`random_state=42`, `random.seed(42)`), para os números deste
arquivo se repetirem quando você reexecutar.

**h. Seção de teste no fim**, com um bloco markdown marcando o fim do treinamento,
mais erro de teste, linha de base e matriz de confusão.

O que **não** mudou: `gera_vetor` (mediana + Sobel + média ponderada) e os
hiperparâmetros da MLP (logística, 1 camada de 100 neurônios, SGD, lr 0.03,
batch 20, 500 épocas).

## 4. Resultados da última execução

```
PCA: 4.096 → 140 atributos (85% da variância)

Erro de validação (1046 imagens de train/):  3,06%
Erro de teste     ( 624 imagens de test/ ): 23,08%
Linha de base (chutar sempre PNEUMONIA):    37,50%
```

Matriz de confusão no teste (linha = real, coluna = previsto):

|             | prev. NORMAL | prev. PNEUMONIA |
|-------------|---:|---:|
| **NORMAL**    |  95 | 139 |
| **PNEUMONIA** |   5 | 385 |

| classe | precisão | recall | f1 |
|---|---:|---:|---:|
| NORMAL | 0,95 | 0,41 | 0,57 |
| PNEUMONIA | 0,73 | 0,99 | 0,84 |
| acurácia | | | 0,77 |

O treino não emitiu `ConvergenceWarning`, ou seja, o SGD convergiu antes de
estourar as 500 épocas — o teto de `max_iter` não é o gargalo aqui.

## 5. Como ler esses números

**O modelo funciona, mas bem menos do que a validação sugere.** 23,08% de erro
contra 37,50% da linha de base: ele aprendeu alguma coisa de verdade, não está
apenas chutando a classe majoritária. Só que o salto de 3% na validação para 23%
no teste é grande demais para ser ruído, e é o achado principal aqui.

**Ele erra de um jeito bem específico:** acerta 385 de 390 pneumonias (recall
0,99) e só 95 de 234 normais (recall 0,41). Ou seja, na dúvida ele responde
"pneumonia". Clinicamente é o erro menos perigoso — quase não deixa passar doente
—, mas manda 59% dos saudáveis para exame desnecessário.

**Duas causas prováveis para o abismo validação × teste**, e vale citar as duas
no relatório:

1. **Desbalanceamento no treino** — 74% das imagens de `train/` são PNEUMONIA, e
   a rede aprende esse viés a priori. O `test/` é menos desbalanceado (62,5%).
2. ~~**Validação otimista por radiografias repetidas do mesmo paciente.**~~
   **Hipótese testada e DESCARTADA** — ver `validacao-por-paciente.ipynb`.
   O dataset de fato tem repetição (5232 imagens para 2650 pacientes, 1250 deles
   com mais de uma radiografia, até 30 de um mesmo paciente), então a suspeita era
   razoável. Mas ao refazer a validação cruzada **agrupando por paciente**, de
   forma que ninguém apareça dos dois lados do sorteio, o erro **não subiu**:
   3,57% na divisão aleatória por imagem contra 3,00% na agrupada por paciente.
   Não há vazamento relevante.

   O que sobra é a causa (1) mais **mudança de distribuição entre as pastas**:
   dividir o `train/` de qualquer maneira dá ~3%, e o `test/` dá 22%. As duas
   pastas não são amostras da mesma população — o `test/` deste dataset foi
   curado à parte. Por isso o número que vale é o de teste.

## 6. Se for continuar

Em ordem de retorno pelo esforço:

- ~~Sortear a divisão por paciente.~~ Testado, **não ajuda** — o erro agrupado por
  paciente é o mesmo do aleatório (`validacao-por-paciente.ipynb`). A validação
  interna continua sem prever o teste, e nenhuma forma de dividir o `train/`
  conserta isso: o problema é a diferença entre as pastas, não o sorteio.
- Como a diferença é de distribuição, o que tende a ajudar é reduzir a
  sensibilidade a como a imagem foi adquirida: normalização por imagem
  (equalização de histograma / CLAHE) no lugar da padronização por pixel, e
  aumento de dados (espelhamento, pequenas rotações).
- **Meça sempre no `test/`.** Ganhar meio ponto naquele 3% de validação não diz
  nada sobre os 22% reais.
- Balancear as classes **sem trocar de classificador nem jogar dados fora**: o
  `MLPClassifier` não tem `class_weight`, mas o `fit` desta versão do scikit-learn
  (1.9) aceita `sample_weight`. Testado nesta máquina, funciona:
  ```python
  from sklearn.utils.class_weight import compute_sample_weight
  pesos = compute_sample_weight('balanced', alvos[indices_treino])
  mlp.fit(treinamento_pca[indices_treino], alvos[indices_treino], sample_weight=pesos)
  ```
- Ajustar o limiar de decisão com `mlp.predict_proba` em vez de `predict`. Não
  exige retreinar nada e é a forma direta de escolher o ponto de operação entre
  "não deixar passar pneumonia" e "não alarmar saudável".
- Subir a resolução para 96×96 ou 128×128 provavelmente ajuda mais que mexer na
  rede. Deve caber, mas **meça antes de confiar**: 128×128 dá 16.384 atributos,
  que são ~685 MB numa cópia — só que a normalização mantém duas matrizes e o PCA
  aloca a dele. No 64×64 o pico medido foi 4,5× o tamanho de uma cópia, o que aqui
  projeta ~3 GB numa máquina de 8 GB.
- Reportar recall de PNEUMONIA como métrica principal, não acurácia — é o que
  importa no problema e o que a acurácia esconde.

## 7. Como rodar

```bash
source /Users/icaroamerico/Documents/venv_icaro/bin/activate
jupyter lab /Users/icaroamerico/Documents/imagens_fase_1_pos_fiap/aprendizadado-maquina-imagem.ipynb
```

**Selecione o kernel na mão: "Python (venv_icaro)".** A seleção automática cai no
kernel errado — o `kernelspec` herdado do arquivo original aponta para o `.venv`
de outro projeto, e é daí que vinham os `ModuleNotFoundError` de `sklearn` e `cv2`
nas saídas antigas. No VS Code: Select Kernel → `venv_icaro/bin/python3`.

Ambiente: Python 3.12.13, numpy 2.5.0, opencv-python 4.13.0, scikit-learn 1.9.0,
matplotlib 3.11.1, jupyterlab 4.6.3.
