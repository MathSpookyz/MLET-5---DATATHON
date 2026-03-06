import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def obter_dados_e_preprocessar(caminho_csv, scaler):
    logger.info("Loading CSV from %s", caminho_csv)
    df = pd.read_csv(caminho_csv, sep=";", encoding="utf-8-sig")
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    metricas = ["INDE", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN"]

    for metrica in metricas:
        coluna_metrica = [c for c in df.columns if metrica in c]

        df[metrica + "_FINAL"] = df[coluna_metrica].apply(
            lambda x: x.dropna().iloc[-1] if x.dropna().any() else np.nan,
            axis=1,
        )

        df[metrica + "_FINAL"] = pd.to_numeric(
            df[metrica + "_FINAL"], errors="coerce"
        )

        df[metrica + "_FINAL"] = df[metrica + "_FINAL"].fillna(
            df[metrica + "_FINAL"].median()
        )

    logger.info("Consolidated %d metrics into *_FINAL columns", len(metricas))

    cluster_metricas = [m + "_FINAL" for m in metricas]

    metricas_normalizadas = scaler.fit_transform(df[cluster_metricas])
    logger.info("Normalization complete, shape=%s", metricas_normalizadas.shape)

    return df, cluster_metricas, metricas_normalizadas
