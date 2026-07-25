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


if __name__ == "__main__":
    dados = carregar_datasets()
    print(f"Quantidade de arquivos encontrados em base_dados/: {len(dados)}")
    for nome_arquivo, dataframe in dados.items():
        verificar_estrutura(nome_arquivo, dataframe)
