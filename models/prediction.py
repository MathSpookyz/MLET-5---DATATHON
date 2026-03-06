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

def prever_todos_alunos(df, scaler, kmeans, modelo_rf, cluster_metricas):

    logger.info("Predicting for all students")

    df_result = df.copy()

    aluno_normalizado = scaler.transform(df_result[cluster_metricas])

    grupos = kmeans.predict(aluno_normalizado)

    prob_pv = modelo_rf.predict_proba(df_result[cluster_metricas])[:, 1]

    df_result["Grupo"] = grupos
    df_result["Nivel"] = ["Nível " + str(g + 1) for g in grupos]
    df_result["Probabilidade_PV"] = prob_pv

    logger.info("Prediction for all students completed")

    return df_result