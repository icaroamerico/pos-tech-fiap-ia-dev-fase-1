"""Ponto de entrada do pipeline: chama a ETAPA 1 (diagnóstico) e, na
sequência, a ETAPA 2 (tratamento) para cada dataset em ``base_dados/``.

Execução: ``python3 -m src.main`` (ou ``python3 src/main.py``) a partir da
raiz do projeto.
"""

import sys
from pathlib import Path

# Garante que a raiz do projeto esteja em sys.path, para que ``from
# src.etl...`` funcione tanto rodando este arquivo diretamente
# (``python3 src/main.py``) quanto como módulo (``python3 -m src.main``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.etl.etapa1_diagnostico import carregar_datasets, executar_diagnostico  # noqa: E402
from src.etl.etapa2_tratamento import (  # noqa: E402
    executar_tratamento,
    imprimir_resumo,
    salvar_dataset,
)


def main() -> None:
    """Carrega os dados, roda a ETAPA 1 (diagnóstico) por completo para
    todos os arquivos e só então roda a ETAPA 2 (tratamento) e a exportação.

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