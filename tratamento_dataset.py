"""Ponto de entrada do pipeline, mantido pelo nome pedido em prompt.md.

A lógica foi dividida em ``etapa1_diagnostico.py`` (ETAPA 1) e
``etapa2_tratamento.py`` (ETAPA 2, que orquestra as duas etapas via
``main()``).
"""

from etapa2_tratamento import main

if __name__ == "__main__":
    main()
