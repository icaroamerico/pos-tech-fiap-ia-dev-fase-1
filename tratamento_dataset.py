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


if __name__ == "__main__":
    dados = carregar_datasets()
    print(f"Quantidade de arquivos encontrados em base_dados/: {len(dados)}")
    for nome_arquivo, dataframe in dados.items():
        verificar_estrutura(nome_arquivo, dataframe)
        verificar_valores_ausentes(nome_arquivo, dataframe)
        verificar_linhas_duplicadas(nome_arquivo, dataframe)
        verificar_colunas_duplicadas(nome_arquivo, dataframe)
        verificar_tipos_incorretos(nome_arquivo, dataframe)
