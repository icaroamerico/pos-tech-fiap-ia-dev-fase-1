"""Diagnóstico e tratamento automático dos datasets em ``base_dados/``.

Baseado nas regras descritas em ``.claude/prompt.md``:

- ETAPA 1 (diagnóstico): reproduz automaticamente todas as verificações de
  qualidade de dados, sempre executadas, independente do resultado.
- ETAPA 2 (tratamento): aplica as correções sobre os dados, com comentários
  explicando problema encontrado, estratégia utilizada e justificativa.

Os datasets tratados são salvos em ``base_dados_tratada/`` preservando os
nomes originais dos arquivos.
"""

import re
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
    """Colunas de texto elegíveis para as checagens 6-9 (exclui protegidas)."""
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
# ETAPA 1 — Runner do diagnóstico completo
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


# =============================================================================
# ETAPA 2 — TRATAMENTO
#
# Cada função abaixo resolve exatamente um problema identificado na ETAPA 1.
# O comentário de cada função documenta: problema encontrado, estratégia
# utilizada e justificativa. Nenhuma função toca em COLUNAS_PROTEGIDAS
# (Blood Group), conforme a regra do dataset.
# =============================================================================

def _resumo_vazio() -> dict:
    """Cria o dicionário de contadores usado no resumo final da ETAPA 2."""
    return {
        "linhas_removidas": 0,
        "duplicados_removidos": 0,
        "valores_ausentes_preenchidos": 0,
        "outliers_tratados": 0,
        "datas_convertidas": 0,
        "blood_pressure_separada": [],
        "yes_no_convertidos": [],
        "unidades_convertidas": [],
        "colunas_removidas": [],
    }


def tratar_linhas_duplicadas(df: pd.DataFrame, resumo: dict) -> pd.DataFrame:
    """Problema: linhas totalmente duplicadas inflam artificialmente a
    importância de certos registros para o modelo.
    Estratégia: remover duplicatas exatas, mantendo a primeira ocorrência.
    Justificativa: registros idênticos em todas as colunas não agregam
    informação nova e podem enviesar o treinamento.
    """
    linhas_antes = len(df)
    df = df.drop_duplicates(keep="first")
    removidas = linhas_antes - len(df)
    resumo["duplicados_removidos"] += removidas
    resumo["linhas_removidas"] += removidas
    return df


def tratar_tipos(df: pd.DataFrame, resumo: dict) -> pd.DataFrame:
    """Problema: colunas majoritariamente numéricas ficam com dtype
    ``object`` por causa de sujeira pontual (ex.: 'a' em AMH(ng/mL) e
    '1.99.' em II beta-HCG(mIU/mL)).
    Estratégia: colunas de texto (não protegidas) em que pelo menos 90% dos
    valores não nulos são convertíveis para número são coagidas com
    ``pd.to_numeric(errors="coerce")``; o valor problemático vira NaN e é
    resolvido depois, na etapa de valores ausentes.
    Justificativa: forçar o tipo correto é necessário para que essas colunas
    possam ser usadas em cálculos estatísticos e em modelos de ML.
    """
    for coluna in df.select_dtypes(include="object").columns:
        if coluna in COLUNAS_PROTEGIDAS:
            continue
        serie_texto = df[coluna].dropna().astype(str).str.strip()
        if serie_texto.empty:
            continue
        convertido = pd.to_numeric(serie_texto, errors="coerce")
        if convertido.notna().mean() >= 0.9:
            df[coluna] = pd.to_numeric(df[coluna].astype(str).str.strip(), errors="coerce")
    return df


def tratar_datas(df: pd.DataFrame, resumo: dict) -> pd.DataFrame:
    """Problema: colunas de data podem estar armazenadas como texto em
    formatos variados.
    Estratégia: colunas de texto cujo padrão bate com ``_PADRAO_DATA`` são
    convertidas com ``pd.to_datetime(errors="coerce")``.
    Justificativa: padronizar datas em ``datetime64`` evita erros de
    ordenação/comparação e permite extrair features temporais.
    Neste dataset (PCOS) não há colunas de data, então esta função tende a
    ser um no-op — mas continua sendo executada, pois o requisito é que o
    tratamento funcione para qualquer arquivo .csv/.xlsx.
    """
    for coluna in df.select_dtypes(include="object").columns:
        if coluna in COLUNAS_PROTEGIDAS:
            continue
        serie_texto = df[coluna].dropna().astype(str).str.strip()
        if serie_texto.empty:
            continue
        if serie_texto.str.match(_PADRAO_DATA).mean() > 0.5:
            convertidas_antes = df[coluna].notna().sum()
            df[coluna] = pd.to_datetime(df[coluna], errors="coerce")
            resumo["datas_convertidas"] += int(convertidas_antes)
    return df


def tratar_espacos(df: pd.DataFrame, resumo: dict) -> pd.DataFrame:
    """Problema real encontrado: nomes de coluna com espaços nas bordas ou
    duplicados (ex.: ' Age (yrs)', 'Height(Cm) ', '  I   beta-HCG(mIU/mL)').
    Estratégia: remover espaços nas bordas dos nomes de coluna e colapsar
    espaços internos duplicados para um único espaço; nos valores de texto
    (colunas não protegidas), aplicar a mesma limpeza.
    Justificativa: nomes/valores com espaços extras quebram comparações
    exatas (ex.: merges, filtros) e criam falsas categorias distintas.
    """
    df = df.rename(columns=lambda c: re.sub(r"\s+", " ", c).strip())

    for coluna in df.select_dtypes(include="object").columns:
        if coluna in COLUNAS_PROTEGIDAS:
            continue
        df[coluna] = df[coluna].apply(
            lambda v: re.sub(r"\s+", " ", v).strip() if isinstance(v, str) else v
        )
    return df


def padronizar_categorias(df: pd.DataFrame, resumo: dict) -> pd.DataFrame:
    """Problema: a mesma categoria pode aparecer com grafias diferentes
    (ex.: 'Sim'/'SIM'/'Yes'; 'joao'/'JOAO'/'João').
    Estratégia: para colunas de texto não protegidas, agrupar valores pela
    forma normalizada (minúscula, sem espaços) e substituir todas as
    variantes pela forma mais frequente do grupo.
    Justificativa: reduzir cardinalidade artificial evita que um modelo de
    ML trate a mesma categoria como classes diferentes.
    """
    for coluna in _colunas_texto(df):
        serie = df[coluna]
        nao_nulos = serie.dropna().astype(str)
        if nao_nulos.empty:
            continue

        normalizado = nao_nulos.str.strip().str.lower()
        mapa_canonico = (
            pd.DataFrame({"valor": nao_nulos, "chave": normalizado})
            .groupby("chave")["valor"]
            .agg(lambda valores: valores.value_counts().idxmax())
        )

        df[coluna] = serie.apply(
            lambda v: mapa_canonico.get(str(v).strip().lower(), v)
            if isinstance(v, str)
            else v
        )
    return df


def converter_yes_no(df: pd.DataFrame, resumo: dict) -> pd.DataFrame:
    """Problema: colunas de pergunta Sim/Não podem vir como texto em
    variações (Yes/No, Y/N, Sim/Não, S/N, 1/0).
    Estratégia (condicional): para toda coluna cujo nome contenha "(Y/N)",
    se já for numérica com valores em {0, 1} não faz nada (no-op); se for
    texto, mapeia variações afirmativas para 1 e negativas para 0.
    Justificativa: modelos de ML exigem entrada numérica; manter a
    codificação binária padronizada (1 = Yes/Sim, 0 = No/Não) simplifica o
    pré-processamento.
    """
    mapa_sim = {"yes", "y", "sim", "s", "true", "1"}
    mapa_nao = {"no", "n", "não", "nao", "false", "0"}

    for coluna in df.columns:
        if "(y/n)" not in coluna.strip().lower():
            continue
        if pd.api.types.is_numeric_dtype(df[coluna]):
            continue  # já convertida (no-op), conforme achado da ETAPA 1.

        def _mapear(valor):
            if pd.isna(valor):
                return valor
            chave = str(valor).strip().lower()
            if chave in mapa_sim:
                return 1
            if chave in mapa_nao:
                return 0
            return valor  # valor não reconhecido: mantido para análise manual.

        df[coluna] = df[coluna].apply(_mapear)
        resumo["yes_no_convertidos"].append(coluna)
    return df


def separar_blood_pressure(df: pd.DataFrame, resumo: dict) -> pd.DataFrame:
    """Problema: a pressão arterial pode vir em uma única coluna no formato
    "120/80" (sistólica/diastólica juntas).
    Estratégia (condicional): se existir coluna nesse formato, separá-la em
    duas colunas numéricas (Systolic/Diastolic) e remover a coluna
    combinada; se a separação já existir (como neste dataset, com
    'BP _Systolic (mmHg)' / 'BP _Diastolic (mmHg)'), não faz nada (no-op).
    Justificativa: manter sistólica e diastólica juntas em uma string
    impede seu uso direto como variáveis numéricas em um modelo.
    """
    for coluna in df.select_dtypes(include="object").columns:
        if coluna in COLUNAS_PROTEGIDAS:
            continue
        valores = df[coluna].dropna().astype(str).str.strip()
        if valores.empty or valores.str.match(_PADRAO_PRESSAO_COMBINADA).mean() <= 0.5:
            continue

        partes = df[coluna].astype(str).str.strip().str.split("/", expand=True)
        df[f"{coluna}_Systolic"] = pd.to_numeric(partes[0], errors="coerce")
        df[f"{coluna}_Diastolic"] = pd.to_numeric(partes[1], errors="coerce")
        df = df.drop(columns=[coluna])
        resumo["blood_pressure_separada"].append(coluna)
    return df


def converter_unidades(df: pd.DataFrame, resumo: dict) -> pd.DataFrame:
    """Problema: a altura pode estar em pés (feet) em vez de centímetros,
    conforme instrução do dataset ("converter unidades quando necessário").
    Estratégia (condicional): se existir coluna de altura em pés, converter
    para centímetros (1 pé = 30.48 cm); se a coluna já estiver em cm (como
    'Height(Cm)' neste dataset), não faz nada (no-op).
    Justificativa: manter uma unidade única e consistente evita distorções
    de escala em modelos sensíveis a magnitude (ex.: KNN, regressão linear).
    """
    colunas_pes = [
        c for c in df.columns if "feet" in c.strip().lower() or "(ft)" in c.strip().lower()
    ]
    for coluna in colunas_pes:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce") * 30.48
        resumo["unidades_convertidas"].append(coluna)
    return df


def tratar_valores_ausentes(df: pd.DataFrame, resumo: dict) -> pd.DataFrame:
    """Problema: valores ausentes (reais ou disfarçados em texto) impedem o
    uso direto da coluna por muitos algoritmos de ML.
    Estratégia: primeiro, normalizar tokens textuais de ausência ("-", "--",
    "NA", "N/A", "null", etc.) para NaN de verdade; depois, imputar colunas
    numéricas com a mediana (robusta a outliers) e colunas de texto com a
    moda (valor mais frequente). A coluna protegida (Blood Group) é
    imputada com a moda também, mas sem qualquer recodificação de valores.
    Justificativa: mediana/moda são estratégias simples e não distorcem a
    distribuição tanto quanto a média em colunas com outliers.
    """
    for coluna in df.select_dtypes(include="object").columns:
        mascara = _mascara_ausentes_disfarcados(df[coluna].astype(str))
        df.loc[df[coluna].notna() & mascara, coluna] = pd.NA

    for coluna in df.columns:
        qtd_ausente_antes = int(df[coluna].isna().sum())
        if qtd_ausente_antes == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[coluna]):
            valor_imputado = df[coluna].median()
        else:
            moda = df[coluna].mode(dropna=True)
            valor_imputado = moda.iloc[0] if not moda.empty else None

        if valor_imputado is not None:
            df[coluna] = df[coluna].fillna(valor_imputado)
            resumo["valores_ausentes_preenchidos"] += qtd_ausente_antes
    return df


def tratar_outliers(df: pd.DataFrame, resumo: dict) -> pd.DataFrame:
    """Problema: outliers extremos em colunas numéricas (ex.: beta-HCG,
    AMH) distorcem médias, desvios-padrão e o ajuste de modelos sensíveis a
    escala.
    Estratégia: capping (winsorização) pelos limites de IQR (Q1 - 1.5*IQR,
    Q3 + 1.5*IQR) nas colunas numéricas de análise (exclui protegidas e
    IDs). Os valores fora do limite são "grudados" no limite mais próximo.
    Justificativa: capping preserva todas as linhas (diferente de remover),
    reduzindo a influência de extremos sem descartar informação; a remoção
    automática sem justificativa individual foi explicitamente evitada,
    conforme o prompt.
    """
    for coluna in _colunas_numericas_para_analise(df):
        serie = df[coluna].dropna()
        if serie.empty or serie.nunique() <= 1:
            continue

        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            # Mesmo caso degenerado tratado em verificar_outliers: IQR não
            # se aplica a colunas binárias/discretas muito concentradas.
            # Fazer o capping aqui colapsaria a coluna inteira para um único
            # valor (ex.: 'Reg.Exercise(Y/N)' perderia a classe minoritária).
            continue
        limite_inferior, limite_superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        fora_do_limite = (
            (df[coluna] < limite_inferior) | (df[coluna] > limite_superior)
        ) & df[coluna].notna()
        qtd_tratada = int(fora_do_limite.sum())
        if qtd_tratada:
            df[coluna] = df[coluna].clip(lower=limite_inferior, upper=limite_superior)
            resumo["outliers_tratados"] += qtd_tratada
    return df


def remover_colunas_constantes(df: pd.DataFrame, resumo: dict) -> pd.DataFrame:
    """Problema: colunas constantes (um único valor em toda a coluna) não
    carregam nenhuma informação discriminativa para o modelo.
    Estratégia: remover colunas com exatamente 1 valor distinto entre os
    valores não nulos.
    Justificativa: colunas constantes só aumentam a dimensionalidade sem
    contribuir para a predição.
    """
    colunas_constantes = [
        c for c in df.columns if df[c].dropna().nunique() == 1 and c not in COLUNAS_PROTEGIDAS
    ]
    if colunas_constantes:
        df = df.drop(columns=colunas_constantes)
        resumo["colunas_removidas"].extend(colunas_constantes)
    return df


def remover_colunas_irrelevantes(df: pd.DataFrame, resumo: dict) -> pd.DataFrame:
    """Problema real encontrado: a coluna 'Unnamed: 44' do xlsx está vazia
    em 539 das 541 linhas — resíduo de formatação da planilha original, sem
    valor analítico.
    Estratégia: remover colunas totalmente vazias e colunas "Unnamed: N"
    com menos de 10% de preenchimento. Colunas de identificação (Sl. No,
    Patient File No.) são mantidas no arquivo de saída — mesmo que não
    sejam usadas como feature — para preservar a rastreabilidade dos
    registros.
    Justificativa: colunas sem conteúdo real não podem ser usadas por
    nenhum modelo; remover é seguro e não descarta informação.
    """
    colunas_vazias = [c for c in df.columns if df[c].isna().all()]
    colunas_unnamed_quase_vazias = [
        c
        for c in df.columns
        if c.strip().lower().startswith("unnamed") and df[c].notna().mean() < 0.1
    ]
    colunas_para_remover = sorted(set(colunas_vazias) | set(colunas_unnamed_quase_vazias))
    colunas_para_remover = [c for c in colunas_para_remover if c not in COLUNAS_PROTEGIDAS]

    if colunas_para_remover:
        df = df.drop(columns=colunas_para_remover)
        resumo["colunas_removidas"].extend(colunas_para_remover)
    return df


def executar_tratamento(nome: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica a ETAPA 2 completa, na ordem correta, e retorna o DataFrame
    tratado junto com o resumo de contadores.

    Ordem: duplicidade -> tipos -> datas -> espaços -> colunas irrelevantes
    -> colunas constantes -> categorias -> Yes/No -> Blood Pressure ->
    unidades -> valores ausentes -> outliers. Esta ordem garante que, por
    exemplo, colunas quase vazias sejam removidas antes de serem imputadas
    (senão a imputação as faria parecer preenchidas), e que os tipos já
    estejam corrigidos antes de imputar ausentes/calcular outliers.
    """
    resumo = _resumo_vazio()
    df = tratar_linhas_duplicadas(df, resumo)
    df = tratar_tipos(df, resumo)
    df = tratar_datas(df, resumo)
    df = tratar_espacos(df, resumo)
    # Remover colunas irrelevantes/constantes ANTES de imputar valores
    # ausentes: uma coluna quase vazia (ex.: 'Unnamed: 44', com só 2/541
    # valores preenchidos) precisa ser identificada como "quase vazia" e
    # descartada antes que a imputação a preencha por completo e a faça
    # parecer uma coluna válida.
    df = remover_colunas_irrelevantes(df, resumo)
    df = remover_colunas_constantes(df, resumo)
    df = padronizar_categorias(df, resumo)
    df = converter_yes_no(df, resumo)
    df = separar_blood_pressure(df, resumo)
    df = converter_unidades(df, resumo)
    df = tratar_valores_ausentes(df, resumo)
    df = tratar_outliers(df, resumo)
    return df, resumo


def imprimir_resumo(nome: str, resumo: dict) -> None:
    """Imprime o resumo final do tratamento no formato pedido no prompt."""
    print(f"\nResumo — {nome}")
    print(f"Linhas removidas: {resumo['linhas_removidas']}")
    print(f"Duplicados removidos: {resumo['duplicados_removidos']}")
    print(f"Valores ausentes preenchidos: {resumo['valores_ausentes_preenchidos']}")
    print(f"Outliers tratados: {resumo['outliers_tratados']}")
    print(f"Datas convertidas: {resumo['datas_convertidas']}")
    bp_separada = resumo["blood_pressure_separada"] or "não aplicável (já separada ou inexistente)"
    yn_convertidos = resumo["yes_no_convertidos"] or "não aplicável (já estavam em 0/1)"
    unidades = resumo["unidades_convertidas"] or "não aplicável (já na unidade esperada)"
    print(f"Blood Pressure separada: {bp_separada}")
    print(f"Yes/No convertidos: {yn_convertidos}")
    print(f"Unidades convertidas: {unidades}")
    colunas_removidas = resumo["colunas_removidas"] or "nenhuma"
    print(f"Colunas removidas (constantes/irrelevantes): {colunas_removidas}")


def salvar_dataset(nome: str, df: pd.DataFrame, pasta_saida: Path = OUTPUT_DIR) -> None:
    """Salva ``df`` em ``pasta_saida``, preservando o nome original do
    arquivo (csv salvo como csv, xlsx salvo como xlsx)."""
    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho_saida = pasta_saida / nome
    sufixo = caminho_saida.suffix.lower()

    if sufixo == ".csv":
        df.to_csv(caminho_saida, index=False)
    elif sufixo in (".xlsx", ".xls"):
        df.to_excel(caminho_saida, index=False)
    else:
        raise ValueError(f"Extensão não suportada para salvar: {sufixo}")

    print(f"Dataset tratado salvo em: {caminho_saida}")


def main() -> None:
    """Ponto de entrada: carrega os dados, roda a ETAPA 1 (diagnóstico) por
    completo e só então roda a ETAPA 2 (tratamento) e a exportação.

    Cada arquivo é isolado em seu próprio try/except: um problema em um
    dataset (ex.: dado corrompido pela sujeira do arquivo) não deve
    interromper o processamento dos demais arquivos.
    """
    dados = carregar_datasets()
    print(f"Quantidade de arquivos encontrados em base_dados/: {len(dados)}")

    # ETAPA 1 — diagnóstico completo de todos os arquivos, sempre executado
    # e independente da ETAPA 2 (nenhum dado é alterado aqui).
    for nome_arquivo, dataframe in dados.items():
        try:
            executar_diagnostico(nome_arquivo, dataframe)
        except Exception as exc:
            print(f"[ERRO] Falha no diagnóstico de '{nome_arquivo}': {exc}")

    # ETAPA 2 — tratamento e exportação, um arquivo por vez.
    for nome_arquivo, dataframe in dados.items():
        print(f"\n{'#' * 70}\nTRATAMENTO — {nome_arquivo}\n{'#' * 70}")
        try:
            dataframe_tratado, resumo = executar_tratamento(nome_arquivo, dataframe.copy())
            imprimir_resumo(nome_arquivo, resumo)
            salvar_dataset(nome_arquivo, dataframe_tratado)
        except Exception as exc:
            print(f"[ERRO] Falha no tratamento/exportação de '{nome_arquivo}': {exc}")


if __name__ == "__main__":
    main()
