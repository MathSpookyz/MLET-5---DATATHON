import logging
import pandas as pd

logger = logging.getLogger(__name__)


def rankear_alunos_individual(df, cluster_metricas):
    logger.info("Generating individual rankings")

    df_ranking = df.copy()

    df_ranking["Score_Individual"] = df_ranking[cluster_metricas].mean(axis=1)

    df_ranking["Ranking_Grupo"] = (
        df_ranking.groupby("Grupo_ID")["Score_Individual"]
        .rank(ascending=False)
        .astype(int)
    )

    df_ranking["Ranking_Geral"] = (
        df_ranking["Score_Individual"].rank(ascending=False).astype(int)
    )

    logger.info("Rankings generated for %d students", len(df_ranking))
    return df_ranking

def ultimos_alunos(df: pd.DataFrame, top_n: int = 10):

    logger.info("Generating worst students ranking", extra={"top_n": top_n})

    if "Probabilidade_PV" not in df.columns:
        raise ValueError("Column 'Probabilidade_PV' not found in DataFrame")

    ranking = (
        df.sort_values(by="Probabilidade_PV", ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )

    logger.info("Worst ranking generated successfully")

    return ranking


def primeiros_alunos(df: pd.DataFrame, top_n: int = 10):

    logger.info("Generating best students ranking", extra={"top_n": top_n})

    if "Probabilidade_PV" not in df.columns:
        raise ValueError("Column 'Probabilidade_PV' not found in DataFrame")

    ranking = (
        df.sort_values(by="Probabilidade_PV", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    logger.info("Best ranking generated successfully")

    return ranking