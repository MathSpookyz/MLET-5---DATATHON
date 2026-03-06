import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def extrair_dados_excel(caminho_arquivo):
    logger.info("Starting extraction from: %s", caminho_arquivo)

    try:
        excel = pd.ExcelFile(caminho_arquivo)
        abas = excel.sheet_names
        logger.info("Sheets found: %s", abas)
    except Exception as e:
        logger.error("Error opening Excel file: %s", e)
        return

    dfs = {}

    for aba in abas:
        logger.info("Extracting sheet: %s", aba)
        df_temporario = pd.read_excel(
            caminho_arquivo,
            sheet_name=aba,
            na_values=["#N/A", "#DIV/0!", ""],
        )

        nome_csv = f"extraido_{aba}.csv"
        os.makedirs("extraidos", exist_ok=True)
        df_temporario.to_csv(
            os.path.join("extraidos", nome_csv), index=False, encoding="utf-8-sig"
        )

        dfs[aba] = df_temporario
        logger.info("Sheet %s extracted with %d rows", aba, len(df_temporario))

    logger.info("Extraction complete")
    return dfs


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    extrair_dados_excel("BASE DE DADOS PEDE 2024 - DATATHON.xlsx")
