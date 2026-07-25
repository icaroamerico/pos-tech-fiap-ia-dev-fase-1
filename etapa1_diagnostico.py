"""ETAPA 1 — Diagnóstico dos datasets em ``base_dados/``.

Baseado nas regras descritas em ``.claude/prompt.md``: reproduz
automaticamente todas as verificações de qualidade de dados, sempre
executadas, independente do resultado. Esta etapa é somente leitura: nenhuma
verificação altera os DataFrames. O tratamento (ETAPA 2, em
``etapa2_tratamento.py``) só começa depois que todo o diagnóstico termina.
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
# Verificação 4: Tipos incorretos
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
    print(f"\n=== [4] Tipos incorretos — {nome} ===")
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
    """Colunas de texto elegíveis para tratamento de categorias/espaços
    (exclui protegidas)."""
    return [
        c
        for c in df.select_dtypes(include="object").columns
        if c not in COLUNAS_PROTEGIDAS
    ]


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
# Verificação 5: Outliers (IQR, Z-score, Boxplot)
# ---------------------------------------------------------------------------

def verificar_outliers(nome: str, df: pd.DataFrame) -> None:
    """Conta outliers por IQR e Z-score, e imprime o resumo de quartis que
    fundamenta um boxplot (Q1, mediana, Q3, bigodes), sem remover nada.

    Apenas relata a quantidade encontrada; a decisão de tratar (e como) fica
    para a ETAPA 2, com justificativa.
    """
    print(f"\n=== [5] Outliers — {nome} ===")
    colunas = _colunas_numericas_para_analise(df)
    linhas = []

    for coluna in colunas:
        serie = df[coluna].dropna()
        if serie.empty or serie.nunique() <= 1:
            continue

        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            # IQR degenerado (Q1 == Q3): comum em colunas binárias/discretas
            # muito concentradas (ex.: 'Reg.Exercise(Y/N)', 'No. of
            # aborptions'). O método IQR não se aplica bem aqui — a "minoria"
            # não é outlier estatístico, é a classe menos frequente de uma
            # variável discreta. Reportamos 0 outliers por IQR e confiamos
            # no Z-score, mais adequado neste caso.
            limite_inferior, limite_superior = serie.min(), serie.max()
            outliers_iqr = 0
        else:
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
# Verificação 6: Distribuição das variáveis
# ---------------------------------------------------------------------------

def verificar_distribuicao(nome: str, df: pd.DataFrame) -> None:
    """Imprime a distribuição das variáveis: frequência para colunas de
    baixa cardinalidade (categóricas/binárias) e estatísticas de forma
    (assimetria) para colunas contínuas, como substituto textual simples de
    histograma."""
    print(f"\n=== [6] Distribuição das variáveis — {nome} ===")
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
# Verificação 7: Balanceamento da variável alvo
# ---------------------------------------------------------------------------

def verificar_balanceamento_target(nome: str, df: pd.DataFrame) -> None:
    """Verifica o balanceamento da variável alvo (ex.: 'PCOS (Y/N)'), quando
    existir no dataset."""
    print(f"\n=== [7] Balanceamento da variável alvo — {nome} ===")
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
# Verificação 8: Correlação
# ---------------------------------------------------------------------------

def verificar_correlacao(nome: str, df: pd.DataFrame, limite: float = 0.8) -> None:
    """Mostra a matriz de correlação das colunas numéricas relevantes e
    destaca os pares com correlação absoluta acima de ``limite``.

    Colunas como AMH/II beta-HCG (object por sujeira pontual) são
    convertidas para numérico apenas para esta análise (via
    ``pd.to_numeric(errors="coerce")``), sem alterar o DataFrame original.
    """
    print(f"\n=== [8] Correlação — {nome} ===")
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
# Verificação 9: Variância
# ---------------------------------------------------------------------------

def verificar_variancia(nome: str, df: pd.DataFrame) -> None:
    """Detecta colunas constantes (variância zero) e colunas com baixíssima
    variabilidade (mais de 99% dos valores concentrados em uma só
    categoria)."""
    print(f"\n=== [9] Variância — {nome} ===")
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
    print(
        f"Colunas com baixa variabilidade (>=99% concentrado): "
        f"{baixa_variabilidade or 'nenhuma'}"
    )


# ---------------------------------------------------------------------------
# Verificação 10: Colunas irrelevantes
# ---------------------------------------------------------------------------

def verificar_colunas_irrelevantes(nome: str, df: pd.DataFrame) -> None:
    """Identifica possíveis colunas irrelevantes: IDs/códigos, colunas
    totalmente vazias e colunas geradas automaticamente pelo pandas
    (``Unnamed: N``) sem conteúdo útil."""
    print(f"\n=== [10] Colunas irrelevantes — {nome} ===")

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


# ---------------------------------------------------------------------------
# Verificação 11: Regras específicas do dataset
# ---------------------------------------------------------------------------

_PADRAO_PRESSAO_COMBINADA = r"^\d{2,3}\s*/\s*\d{2,3}$"


def verificar_regras_especificas(nome: str, df: pd.DataFrame) -> None:
    """Verifica as regras específicas do dataset descritas na aba
    'Instructions' do PCOS: unidades, Yes/No, Blood Group, Blood Pressure e
    Beta-HCG (Case I / Case II).

    Cada regra é condicional: só é aplicada de fato na ETAPA 2 se a coluna
    existir e ainda estiver no formato "bruto"; caso já esteja no formato
    esperado, é reportada como no-op.
    """
    print(f"\n=== [11] Regras específicas do dataset — {nome} ===")

    # --- Unidades (ex.: Feet -> cm) ---
    colunas_altura = [c for c in df.columns if "height" in c.strip().lower()]
    colunas_pes = [
        c for c in df.columns if "feet" in c.strip().lower() or "(ft)" in c.strip().lower()
    ]
    if colunas_pes:
        print(f"Conversão de unidade necessária (pés -> cm) em: {colunas_pes}")
    elif colunas_altura:
        print(f"Coluna de altura já em cm, sem conversão de unidade necessária: {colunas_altura}")
    else:
        print("Nenhuma coluna de altura/unidade encontrada.")

    # --- Yes/No -> 0/1 ---
    colunas_yn = [c for c in df.columns if "(y/n)" in c.strip().lower()]
    if not colunas_yn:
        print("Nenhuma coluna Yes/No (Y/N) encontrada.")
    for coluna in colunas_yn:
        if pd.api.types.is_numeric_dtype(df[coluna]):
            valores_fora_do_padrao = set(df[coluna].dropna().unique()) - {0, 1}
            if valores_fora_do_padrao:
                print(
                    f"Coluna '{coluna}': já é numérica, mas com valores "
                    f"fora de 0/1: {valores_fora_do_padrao}"
                )
            else:
                print(f"Coluna '{coluna}': já convertida para 0/1 (no-op).")
        else:
            print(f"Coluna '{coluna}': ainda textual, precisará ser convertida para 0/1.")

    # --- Blood Group (NÃO converter) ---
    if "Blood Group" in df.columns:
        print("Coluna 'Blood Group' presente — protegida, não será convertida (regra do dataset).")
    else:
        print("Coluna 'Blood Group' não encontrada neste arquivo.")

    # --- Blood Pressure (separar sistólica/diastólica) ---
    colunas_bp_separadas = [
        c for c in df.columns
        if "systolic" in c.strip().lower() or "diastolic" in c.strip().lower()
    ]
    colunas_bp_combinadas = []
    for coluna in df.select_dtypes(include="object").columns:
        valores = df[coluna].dropna().astype(str).str.strip()
        if not valores.empty and valores.str.match(_PADRAO_PRESSAO_COMBINADA).mean() > 0.5:
            colunas_bp_combinadas.append(coluna)
    if colunas_bp_combinadas:
        print(
            f"Blood Pressure combinada (ex.: '120/80') precisa ser "
            f"separada em: {colunas_bp_combinadas}"
        )
    elif colunas_bp_separadas:
        print(f"Blood Pressure já separada em Sistólica/Diastólica (no-op): {colunas_bp_separadas}")
    else:
        print("Nenhuma coluna de Blood Pressure encontrada.")

    # --- Beta-HCG (Case I / Case II) ---
    colunas_beta_hcg = [c for c in df.columns if "beta-hcg" in c.strip().lower().replace(" ", "")]
    if colunas_beta_hcg:
        print(f"Colunas Beta-HCG identificadas (Case I / Case II): {colunas_beta_hcg}")
    else:
        print("Nenhuma coluna Beta-HCG encontrada.")


# ---------------------------------------------------------------------------
# Verificação 12: Valores impossíveis
# ---------------------------------------------------------------------------

_PALAVRAS_CHAVE_NAO_NEGATIVAS = (
    "age", "idade", "height", "altura", "weight", "peso", "pressure", "bp ",
)


def verificar_valores_impossiveis(nome: str, df: pd.DataFrame) -> None:
    """Detecta valores impossíveis: idade/altura/peso/pressão negativos e
    datas futuras."""
    print(f"\n=== [12] Valores impossíveis — {nome} ===")
    encontrou_problema = False

    for coluna in df.select_dtypes(include=[np.number]).columns:
        nome_normalizado = coluna.strip().lower()
        if not any(palavra in nome_normalizado for palavra in _PALAVRAS_CHAVE_NAO_NEGATIVAS):
            continue
        qtd_negativos = int((df[coluna].dropna() < 0).sum())
        if qtd_negativos:
            print(f"Coluna '{coluna}': {qtd_negativos} valor(es) negativo(s) (impossível).")
            encontrou_problema = True

    hoje = pd.Timestamp.now().normalize()
    for coluna in df.select_dtypes(include="datetime").columns:
        qtd_futuras = int((df[coluna] > hoje).sum())
        if qtd_futuras:
            print(f"Coluna '{coluna}': {qtd_futuras} data(s) no futuro (impossível).")
            encontrou_problema = True

    if not encontrou_problema:
        print("Nenhum valor impossível encontrado.")


# ---------------------------------------------------------------------------
# Runner do diagnóstico completo
# ---------------------------------------------------------------------------

def executar_diagnostico(nome: str, df: pd.DataFrame) -> None:
    """Executa as 12 verificações mantidas sobre ``df``, sempre, na ordem do
    prompt, independente de haver ou não problema em cada uma.

    Esta função é somente leitura: nenhuma verificação altera o DataFrame.
    O tratamento (ETAPA 2) só começa depois que todo o diagnóstico termina.
    """
    print(f"\n{'#' * 70}\nDIAGNÓSTICO — {nome}\n{'#' * 70}")
    verificar_estrutura(nome, df)
    verificar_valores_ausentes(nome, df)
    verificar_linhas_duplicadas(nome, df)
    verificar_tipos_incorretos(nome, df)
    verificar_outliers(nome, df)
    verificar_distribuicao(nome, df)
    verificar_balanceamento_target(nome, df)
    verificar_correlacao(nome, df)
    verificar_variancia(nome, df)
    verificar_colunas_irrelevantes(nome, df)
    verificar_regras_especificas(nome, df)
    verificar_valores_impossiveis(nome, df)
