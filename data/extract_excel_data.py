import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extrair_dados_excel(caminho_arquivo, arquivo_saida):
    logger.info("Starting extraction from: %s", caminho_arquivo)

    try:
        excel = pd.ExcelFile(caminho_arquivo)
        abas = excel.sheet_names
        logger.info("Sheets found: %s", abas)
    except Exception as e:
        logger.error("Error opening Excel file: %s", e)
        return

    lista_dfs = []

    for aba in abas:
        logger.info("Reading sheet: %s", aba)
        df_temp = pd.read_excel(
            caminho_arquivo,
            sheet_name=aba,
            na_values=["#N/A", "#DIV/0!", ""],
        )
        df_temp["ANO_ORIGEM"] = aba
        lista_dfs.append(df_temp)

    df_unificado = pd.concat(lista_dfs, axis=0, ignore_index=True, sort=False)

    def limpar_valor(series):
        if series.dtype == "object":
            series_num = pd.to_numeric(
                series.astype(str).str.replace(",", "."), errors="coerce"
            )
        else:
            series_num = pd.to_numeric(series, errors="coerce")

        if not series_num.dropna().empty and not any(
            isinstance(x, str) and not x.replace(".", "").isdigit()
            for x in series.dropna()[:10]
        ):
            return series_num.fillna(series_num.median())
        else:
            return series.fillna("")

    logger.info("Applying data cleaning (median for numeric, empty string for text)")
    for col in df_unificado.columns:
        df_unificado[col] = limpar_valor(df_unificado[col])

    df_unificado.to_csv(arquivo_saida, index=False, sep=";", encoding="utf-8-sig")
    logger.info(
        "Unified CSV generated: %s (%d rows, %d columns)",
        arquivo_saida,
        len(df_unificado),
        len(df_unificado.columns),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    extrair_dados_excel(
        "BASE DE DADOS PEDE 2024 - DATATHON.xlsx", "dataset/PEDE_PASSOS_DATASET_FIAP.csv"
    )
