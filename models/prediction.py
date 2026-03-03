import logging

import pandas as pd

logger = logging.getLogger(__name__)


def prever_grupo_aluno(novo_aluno_dados, scaler, kmeans, modelo_rf, cluster_metricas):
    logger.info("Predicting for new student data")

    df_aluno = pd.DataFrame([novo_aluno_dados])

    aluno_normalizado = scaler.transform(df_aluno[cluster_metricas])

    grupo_previsto = int(kmeans.predict(aluno_normalizado)[0])

    prob_pv = float(modelo_rf.predict_proba(df_aluno[cluster_metricas])[0, 1])

    resultado = {
        "Grupo": grupo_previsto,
        "Nivel": f"Nível {grupo_previsto + 1}",
        "Probabilidade_PV": round(prob_pv, 4),
    }

    logger.info("Prediction result: %s", resultado)
    return resultado
