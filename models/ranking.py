import logging

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
