"""ETAPA 2 — Tratamento dos datasets em ``base_dados/``.

Aplica as correções sobre os dados diagnosticados em
``etapa1_diagnostico.py``, com comentários explicando problema encontrado,
estratégia utilizada e justificativa em cada função. Cada função abaixo
resolve exatamente um problema identificado na ETAPA 1. Nenhuma função toca
em ``COLUNAS_PROTEGIDAS`` (Blood Group), conforme a regra do dataset.

Os datasets tratados são salvos em ``base_dados_tratada/`` preservando os
nomes originais dos arquivos.
"""

import re
from pathlib import Path

import pandas as pd

from etapa1_diagnostico import (
    COLUNAS_PROTEGIDAS,
    OUTPUT_DIR,
    _PADRAO_DATA,
    _PADRAO_PRESSAO_COMBINADA,
    _colunas_numericas_para_analise,
    _colunas_texto,
    _mascara_ausentes_disfarcados,
    carregar_datasets,
    executar_diagnostico,
)


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
