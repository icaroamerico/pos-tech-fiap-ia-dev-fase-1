"""Diagnóstico e tratamento automático dos datasets em ``base_dados/``.

Baseado nas regras descritas em ``.claude/prompt.md``:

- ETAPA 1 (diagnóstico): reproduz automaticamente todas as verificações de
  qualidade de dados, sempre executadas, independente do resultado.
- ETAPA 2 (tratamento): aplica as correções sobre os dados, com comentários
  explicando problema encontrado, estratégia utilizada e justificativa.

Os datasets tratados são salvos em ``base_dados_tratada/`` preservando os
nomes originais dos arquivos.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "base_dados"
OUTPUT_DIR = BASE_DIR / "base_dados_tratada"

# Coluna(s) que nunca podem ser alteradas por nenhuma verificação/tratamento
# genérico. Regra do dataset (prompt.md): "NÃO converter Blood Group para
# números. Os valores originais devem permanecer exatamente como estão."
COLUNAS_PROTEGIDAS = {"Blood Group"}

# Tokens de texto que representam "ausência de valor" mas que o pandas não
# converte automaticamente para NaN ao ler csv/xlsx (prompt.md item 2).
TOKENS_AUSENTES = {"", "-", "--", "na", "n/a", "null", "none", "nan"}


# ---------------------------------------------------------------------------
# Carregamento dos datasets
# ---------------------------------------------------------------------------

def carregar_datasets(pasta: Path = INPUT_DIR) -> dict[str, pd.DataFrame]:
    """Carrega todos os arquivos .csv/.xlsx de ``pasta``.

    Retorna um dicionário ``{nome_do_arquivo: DataFrame}`` para que cada
    arquivo seja diagnosticado/tratado de forma independente.
    """
    datasets: dict[str, pd.DataFrame] = {}
    for caminho in sorted(Path(pasta).glob("*")):
        sufixo = caminho.suffix.lower()
        try:
            if sufixo == ".csv":
                datasets[caminho.name] = pd.read_csv(caminho)
            elif sufixo in (".xlsx", ".xls"):
                datasets[caminho.name] = _carregar_aba_de_dados(caminho)
        except Exception as exc:  # arquivo corrompido, aba ausente, etc.
            print(f"[ERRO] Falha ao carregar '{caminho.name}': {exc}")
    return datasets


def _carregar_aba_de_dados(caminho: Path) -> pd.DataFrame:
    """Seleciona a aba de dados de um .xlsx, ignorando abas de instrução.

    Problema encontrado: o arquivo PCOS_data_without_infertility.xlsx possui
    uma aba "Instructions" (texto explicativo) antes da aba com os dados reais
    ("Full_new"). Um leitor genérico que sempre pega a primeira aba (índice 0)
    carregaria a aba de instruções por engano.
    Estratégia: ignorar qualquer aba cujo nome contenha "instru" (case
    insensitive) e usar a primeira aba restante como dado real.
    """
    planilha = pd.ExcelFile(caminho)
    abas_de_dados = [
        nome for nome in planilha.sheet_names if "instru" not in nome.lower()
    ]
    aba_escolhida = abas_de_dados[0] if abas_de_dados else planilha.sheet_names[0]
    return planilha.parse(aba_escolhida)


# ---------------------------------------------------------------------------
# ETAPA 1 — Diagnóstico
# Verificação 1: Estrutura
# ---------------------------------------------------------------------------

def verificar_estrutura(nome: str, df: pd.DataFrame) -> None:
    """Verifica quantidade de linhas/colunas, nomes, tipos e memória usada."""
    print(f"\n=== [1] Estrutura — {nome} ===")
    print(f"Quantidade de linhas: {df.shape[0]}")
    print(f"Quantidade de colunas: {df.shape[1]}")
    print(f"Nomes das colunas: {list(df.columns)}")
    print("Tipo de cada coluna:")
    print(df.dtypes)
    memoria_bytes = df.memory_usage(deep=True).sum()
    print(f"Memória utilizada: {memoria_bytes / 1024:.2f} KB")


# ---------------------------------------------------------------------------
# Verificação 2: Valores ausentes
# ---------------------------------------------------------------------------

def _mascara_ausentes_disfarcados(coluna: pd.Series) -> pd.Series:
    """Identifica valores textuais que representam ausência de dado.

    Cobre células vazias, espaços em branco e tokens como "-", "--", "NA",
    "N/A", "null", "None" (maiúsculas/minúsculas variadas), que o pandas não
    reconhece como NaN por padrão.
    """
    texto = coluna.astype(str).str.strip().str.lower()
    return texto.isin(TOKENS_AUSENTES)


def verificar_valores_ausentes(nome: str, df: pd.DataFrame) -> None:
    """Verifica NaN reais e valores ausentes disfarçados em texto.

    Mostra quantidade e percentual por coluna, conforme exigido no prompt.
    """
    print(f"\n=== [2] Valores ausentes — {nome} ===")
    total_linhas = len(df)
    nan_reais = df.isna().sum()

    disfarcados = pd.Series(0, index=df.columns, dtype=int)
    for coluna in df.select_dtypes(include="object").columns:
        disfarcados[coluna] = int(_mascara_ausentes_disfarcados(df[coluna]).sum())

    total_ausentes = nan_reais.add(disfarcados, fill_value=0)
    percentual = (total_ausentes / total_linhas * 100).round(2)

    resumo = pd.DataFrame(
        {
            "nan_reais": nan_reais,
            "ausentes_disfarcados_em_texto": disfarcados,
            "total_ausentes": total_ausentes,
            "percentual (%)": percentual,
        }
    )
    resumo = resumo[resumo["total_ausentes"] > 0]

    if resumo.empty:
        print("Nenhum valor ausente (real ou disfarçado) encontrado.")
    else:
        print(resumo)


# ---------------------------------------------------------------------------
# Verificação 3: Linhas duplicadas
# ---------------------------------------------------------------------------

# Nomes de coluna que costumam representar identificadores de registro,
# usados apenas para uma checagem extra de duplicidade "por ID" (prompt.md
# item 3: "duplicatas considerando possíveis IDs").
CANDIDATOS_ID = {"sl. no", "patient file no."}


def verificar_linhas_duplicadas(nome: str, df: pd.DataFrame) -> None:
    """Verifica duplicidade de linhas completas e por possíveis IDs."""
    print(f"\n=== [3] Linhas duplicadas — {nome} ===")
    duplicadas_completas = int(df.duplicated().sum())
    print(f"Linhas totalmente duplicadas: {duplicadas_completas}")

    colunas_id = [c for c in df.columns if c.strip().lower() in CANDIDATOS_ID]
    if colunas_id:
        for coluna in colunas_id:
            duplicadas_id = int(df[coluna].duplicated().sum())
            print(f"Duplicidade em possível ID '{coluna}': {duplicadas_id}")
    else:
        print("Nenhuma coluna de possível ID identificada para esta checagem.")


# ---------------------------------------------------------------------------
# Verificação 4: Colunas duplicadas
# ---------------------------------------------------------------------------

def verificar_colunas_duplicadas(nome: str, df: pd.DataFrame) -> None:
    """Detecta colunas com nomes repetidos ou conteúdo idêntico."""
    print(f"\n=== [4] Colunas duplicadas — {nome} ===")

    nomes_repetidos = df.columns[df.columns.duplicated()].tolist()
    print(f"Nomes de coluna repetidos: {nomes_repetidos or 'nenhum'}")

    colunas_conteudo_igual = []
    colunas = list(df.columns)
    for i, col_a in enumerate(colunas):
        for col_b in colunas[i + 1:]:
            if df[col_a].equals(df[col_b]):
                colunas_conteudo_igual.append((col_a, col_b))
    print(f"Colunas com conteúdo idêntico: {colunas_conteudo_igual or 'nenhuma'}")


# ---------------------------------------------------------------------------
# Verificação 5: Tipos incorretos
# ---------------------------------------------------------------------------

# Padrões simples de data usados apenas para detecção (não faz parsing real).
_PADRAO_DATA = r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$"


def verificar_tipos_incorretos(nome: str, df: pd.DataFrame) -> None:
    """Detecta números armazenados como texto, datas como texto e colunas
    com mistura de tipos (texto e número na mesma coluna).

    Achado real neste dataset: as colunas 'AMH(ng/mL)' e
    'II    beta-HCG(mIU/mL)' vêm com dtype ``object`` por causa de sujeira
    pontual (ex.: valor "a" e "1.99." em vez de número), quando deveriam ser
    inteiramente numéricas.
    """
    print(f"\n=== [5] Tipos incorretos — {nome} ===")
    encontrou_problema = False

    for coluna in df.select_dtypes(include="object").columns:
        if coluna in COLUNAS_PROTEGIDAS:
            continue  # Blood Group é texto por definição do dataset.

        serie_texto = df[coluna].dropna().astype(str).str.strip()
        if serie_texto.empty:
            continue

        convertido_numero = pd.to_numeric(serie_texto, errors="coerce")
        percentual_numerico = convertido_numero.notna().mean() * 100

        if 0 < percentual_numerico < 100:
            qtd_nao_numerico = int(convertido_numero.isna().sum())
            print(
                f"Coluna '{coluna}': número armazenado como texto "
                f"({percentual_numerico:.1f}% dos valores são numéricos, "
                f"{qtd_nao_numerico} valor(es) não numérico(s))."
            )
            encontrou_problema = True

        percentual_data = serie_texto.str.match(_PADRAO_DATA).mean() * 100
        if percentual_data > 50:
            print(f"Coluna '{coluna}': possível data armazenada como texto.")
            encontrou_problema = True

        tipos_distintos = serie_texto.map(type).nunique()
        if tipos_distintos > 1:
            print(f"Coluna '{coluna}': mistura de tipos de dado na mesma coluna.")
            encontrou_problema = True

    if not encontrou_problema:
        print("Nenhum problema de tipo incorreto encontrado.")


def _colunas_texto(df: pd.DataFrame) -> list:
    """Colunas de texto elegíveis para as checagens 6-9 (exclui protegidas)."""
    return [
        c
        for c in df.select_dtypes(include="object").columns
        if c not in COLUNAS_PROTEGIDAS
    ]


# ---------------------------------------------------------------------------
# Verificação 6: Valores inconsistentes (categorias)
# ---------------------------------------------------------------------------

def verificar_valores_inconsistentes(nome: str, df: pd.DataFrame) -> None:
    """Detecta categorias que representam o mesmo valor com grafias
    diferentes (ex.: 'Sim'/'SIM'/'Yes'/'S', 'Não'/'NO'/'No'/'N'), agrupando
    por forma normalizada (minúsculo e sem espaços nas bordas)."""
    print(f"\n=== [6] Valores inconsistentes (categorias) — {nome} ===")
    encontrou_problema = False

    for coluna in _colunas_texto(df):
        valores = df[coluna].dropna().astype(str)
        normalizados = valores.str.strip().str.lower()
        agrupado = pd.DataFrame({"original": valores, "normalizado": normalizados})
        for _, grupo in agrupado.groupby("normalizado"):
            variantes = grupo["original"].unique()
            if len(variantes) > 1:
                print(f"Coluna '{coluna}': variantes para o mesmo valor -> {list(variantes)}")
                encontrou_problema = True

    if not encontrou_problema:
        print("Nenhuma inconsistência de categoria encontrada.")


# ---------------------------------------------------------------------------
# Verificação 7: Espaços extras
# ---------------------------------------------------------------------------

def verificar_espacos_extras(nome: str, df: pd.DataFrame) -> None:
    """Detecta valores de texto com espaços extras no início/fim/meio."""
    print(f"\n=== [7] Espaços extras — {nome} ===")
    encontrou_problema = False

    for coluna in _colunas_texto(df):
        valores = df[coluna].dropna().astype(str)
        com_espaco_borda = (valores != valores.str.strip()).sum()
        com_espaco_duplo = valores.str.contains(r"  +", regex=True).sum()
        if com_espaco_borda or com_espaco_duplo:
            print(
                f"Coluna '{coluna}': {com_espaco_borda} valor(es) com espaço "
                f"nas bordas, {com_espaco_duplo} valor(es) com espaços duplos."
            )
            encontrou_problema = True

    if not encontrou_problema:
        print("Nenhum espaço extra encontrado.")


# ---------------------------------------------------------------------------
# Verificação 8: Capitalização
# ---------------------------------------------------------------------------

def verificar_capitalizacao(nome: str, df: pd.DataFrame) -> None:
    """Detecta o mesmo valor textual escrito com capitalização diferente
    (ex.: 'joao'/'JOAO'/'Joao'/'João')."""
    print(f"\n=== [8] Capitalização — {nome} ===")
    encontrou_problema = False

    for coluna in _colunas_texto(df):
        valores = df[coluna].dropna().astype(str).str.strip()
        agrupado = pd.DataFrame({"original": valores, "chave": valores.str.lower()})
        for _, grupo in agrupado.groupby("chave"):
            capitalizacoes = grupo["original"].unique()
            if len(capitalizacoes) > 1:
                print(f"Coluna '{coluna}': capitalizações distintas -> {list(capitalizacoes)}")
                encontrou_problema = True

    if not encontrou_problema:
        print("Nenhuma inconsistência de capitalização encontrada.")


# ---------------------------------------------------------------------------
# Verificação 9: Caracteres especiais / encoding
# ---------------------------------------------------------------------------

def verificar_caracteres_especiais(nome: str, df: pd.DataFrame) -> None:
    """Detecta caracteres não imprimíveis ou indícios de problema de
    encoding (ex.: sequências mojibake típicas de UTF-8 lido como Latin-1)."""
    print(f"\n=== [9] Caracteres especiais — {nome} ===")
    encontrou_problema = False
    padrao_mojibake = r"[ÃÂ][\x80-\xBF]|�"
    padrao_nao_imprimivel = r"[\x00-\x08\x0b\x0c\x0e-\x1f]"

    for coluna in _colunas_texto(df):
        valores = df[coluna].dropna().astype(str)
        qtd_mojibake = valores.str.contains(padrao_mojibake, regex=True).sum()
        qtd_nao_imprimivel = valores.str.contains(padrao_nao_imprimivel, regex=True).sum()
        if qtd_mojibake or qtd_nao_imprimivel:
            print(
                f"Coluna '{coluna}': {qtd_mojibake} valor(es) com possível "
                f"mojibake, {qtd_nao_imprimivel} valor(es) com caractere não "
                f"imprimível."
            )
            encontrou_problema = True

    if not encontrou_problema:
        print("Nenhum caractere especial/problema de encoding encontrado.")


def _colunas_numericas_para_analise(df: pd.DataFrame) -> list:
    """Colunas numéricas relevantes para outliers/distribuição/correlação.

    Exclui a coluna protegida (Blood Group é um código categórico, não uma
    medida contínua) e colunas de identificação (Sl. No / Patient File No.),
    que não fazem sentido em análises estatísticas desse tipo.
    """
    resultado = []
    for coluna in df.select_dtypes(include=[np.number]).columns:
        nome_normalizado = coluna.strip().lower()
        if coluna in COLUNAS_PROTEGIDAS:
            continue
        if nome_normalizado in CANDIDATOS_ID:
            continue
        resultado.append(coluna)
    return resultado


# ---------------------------------------------------------------------------
# Verificação 10: Outliers (IQR, Z-score, Boxplot)
# ---------------------------------------------------------------------------

def verificar_outliers(nome: str, df: pd.DataFrame) -> None:
    """Conta outliers por IQR e Z-score, e imprime o resumo de quartis que
    fundamenta um boxplot (Q1, mediana, Q3, bigodes), sem remover nada.

    Apenas relata a quantidade encontrada; a decisão de tratar (e como) fica
    para a ETAPA 2, com justificativa.
    """
    print(f"\n=== [10] Outliers — {nome} ===")
    colunas = _colunas_numericas_para_analise(df)
    linhas = []

    for coluna in colunas:
        serie = df[coluna].dropna()
        if serie.empty or serie.nunique() <= 1:
            continue

        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        iqr = q3 - q1
        limite_inferior, limite_superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers_iqr = int(((serie < limite_inferior) | (serie > limite_superior)).sum())

        desvio = serie.std(ddof=0)
        if desvio > 0:
            z_scores = (serie - serie.mean()) / desvio
            outliers_zscore = int((z_scores.abs() > 3).sum())
        else:
            outliers_zscore = 0

        linhas.append(
            {
                "coluna": coluna,
                "outliers_iqr": outliers_iqr,
                "outliers_zscore": outliers_zscore,
                "boxplot_q1": round(q1, 2),
                "boxplot_mediana": round(serie.median(), 2),
                "boxplot_q3": round(q3, 2),
                "boxplot_bigode_inf": round(limite_inferior, 2),
                "boxplot_bigode_sup": round(limite_superior, 2),
            }
        )

    resumo = pd.DataFrame(linhas)
    com_outlier = resumo[(resumo["outliers_iqr"] > 0) | (resumo["outliers_zscore"] > 0)]
    if com_outlier.empty:
        print("Nenhum outlier encontrado (IQR e Z-score) nas colunas numéricas analisadas.")
    else:
        print(com_outlier.to_string(index=False))


# ---------------------------------------------------------------------------
# Verificação 11: Distribuição das variáveis
# ---------------------------------------------------------------------------

def verificar_distribuicao(nome: str, df: pd.DataFrame) -> None:
    """Imprime a distribuição das variáveis: frequência para colunas de
    baixa cardinalidade (categóricas/binárias) e estatísticas de forma
    (assimetria) para colunas contínuas, como substituto textual simples de
    histograma."""
    print(f"\n=== [11] Distribuição das variáveis — {nome} ===")
    colunas = _colunas_numericas_para_analise(df)

    for coluna in colunas:
        serie = df[coluna].dropna()
        if serie.empty:
            continue
        if serie.nunique() <= 10:
            frequencias = serie.value_counts(normalize=True).sort_index() * 100
            print(f"Coluna '{coluna}' (frequência %):")
            print(frequencias.round(2))
        else:
            assimetria = serie.skew()
            print(f"Coluna '{coluna}': assimetria (skew) = {assimetria:.2f}")


# ---------------------------------------------------------------------------
# Verificação 12: Balanceamento da variável alvo
# ---------------------------------------------------------------------------

def verificar_balanceamento_target(nome: str, df: pd.DataFrame) -> None:
    """Verifica o balanceamento da variável alvo (ex.: 'PCOS (Y/N)'), quando
    existir no dataset."""
    print(f"\n=== [12] Balanceamento da variável alvo — {nome} ===")
    colunas_alvo = [c for c in df.columns if "pcos" in c.strip().lower()]

    if not colunas_alvo:
        print("Nenhuma variável alvo identificada neste arquivo.")
        return

    for coluna in colunas_alvo:
        contagem = df[coluna].value_counts()
        percentual = (df[coluna].value_counts(normalize=True) * 100).round(2)
        print(f"Coluna alvo '{coluna}':")
        print(pd.DataFrame({"contagem": contagem, "percentual (%)": percentual}))


# ---------------------------------------------------------------------------
# Verificação 13: Correlação
# ---------------------------------------------------------------------------

def verificar_correlacao(nome: str, df: pd.DataFrame, limite: float = 0.8) -> None:
    """Mostra a matriz de correlação das colunas numéricas relevantes e
    destaca os pares com correlação absoluta acima de ``limite``.

    Colunas como AMH/II beta-HCG (object por sujeira pontual) são
    convertidas para numérico apenas para esta análise (via
    ``pd.to_numeric(errors="coerce")``), sem alterar o DataFrame original.
    """
    print(f"\n=== [13] Correlação — {nome} ===")
    colunas = _colunas_numericas_para_analise(df)
    if len(colunas) < 2:
        print("Colunas numéricas insuficientes para calcular correlação.")
        return

    numerico = df[colunas].apply(pd.to_numeric, errors="coerce")
    matriz = numerico.corr()
    print(matriz.round(2))

    pares_altos = []
    for i, col_a in enumerate(matriz.columns):
        for col_b in matriz.columns[i + 1:]:
            valor = matriz.loc[col_a, col_b]
            if pd.notna(valor) and abs(valor) >= limite:
                pares_altos.append((col_a, col_b, round(valor, 2)))

    if pares_altos:
        print(f"Pares com correlação absoluta >= {limite}: {pares_altos}")
    else:
        print(f"Nenhum par de colunas com correlação absoluta >= {limite}.")


# ---------------------------------------------------------------------------
# Verificação 14: Variância
# ---------------------------------------------------------------------------

def verificar_variancia(nome: str, df: pd.DataFrame) -> None:
    """Detecta colunas constantes (variância zero) e colunas com baixíssima
    variabilidade (mais de 99% dos valores concentrados em uma só
    categoria)."""
    print(f"\n=== [14] Variância — {nome} ===")
    constantes = []
    baixa_variabilidade = []

    for coluna in df.columns:
        serie = df[coluna].dropna()
        if serie.empty:
            continue
        if serie.nunique() == 1:
            constantes.append(coluna)
            continue
        proporcao_moda = serie.value_counts(normalize=True).iloc[0]
        if proporcao_moda >= 0.99:
            baixa_variabilidade.append((coluna, round(proporcao_moda * 100, 2)))

    print(f"Colunas constantes: {constantes or 'nenhuma'}")
    print(f"Colunas com baixa variabilidade (>=99% concentrado): {baixa_variabilidade or 'nenhuma'}")


# ---------------------------------------------------------------------------
# Verificação 15: Colunas irrelevantes
# ---------------------------------------------------------------------------

def verificar_colunas_irrelevantes(nome: str, df: pd.DataFrame) -> None:
    """Identifica possíveis colunas irrelevantes: IDs/códigos, colunas
    totalmente vazias e colunas geradas automaticamente pelo pandas
    (``Unnamed: N``) sem conteúdo útil."""
    print(f"\n=== [15] Colunas irrelevantes — {nome} ===")

    colunas_id = [c for c in df.columns if c.strip().lower() in CANDIDATOS_ID]
    colunas_vazias = [c for c in df.columns if df[c].isna().all()]
    colunas_unnamed = [
        c
        for c in df.columns
        if c.strip().lower().startswith("unnamed") and df[c].notna().mean() < 0.1
    ]

    print(f"Colunas de identificação (ID): {colunas_id or 'nenhuma'}")
    print(f"Colunas totalmente vazias: {colunas_vazias or 'nenhuma'}")
    print(f"Colunas 'Unnamed' quase vazias (<10% preenchido): {colunas_unnamed or 'nenhuma'}")


if __name__ == "__main__":
    dados = carregar_datasets()
    print(f"Quantidade de arquivos encontrados em base_dados/: {len(dados)}")
    for nome_arquivo, dataframe in dados.items():
        verificar_estrutura(nome_arquivo, dataframe)
        verificar_valores_ausentes(nome_arquivo, dataframe)
        verificar_linhas_duplicadas(nome_arquivo, dataframe)
        verificar_colunas_duplicadas(nome_arquivo, dataframe)
        verificar_tipos_incorretos(nome_arquivo, dataframe)
        verificar_valores_inconsistentes(nome_arquivo, dataframe)
        verificar_espacos_extras(nome_arquivo, dataframe)
        verificar_capitalizacao(nome_arquivo, dataframe)
        verificar_caracteres_especiais(nome_arquivo, dataframe)
        verificar_outliers(nome_arquivo, dataframe)
        verificar_distribuicao(nome_arquivo, dataframe)
        verificar_balanceamento_target(nome_arquivo, dataframe)
        verificar_correlacao(nome_arquivo, dataframe)
        verificar_variancia(nome_arquivo, dataframe)
        verificar_colunas_irrelevantes(nome_arquivo, dataframe)
