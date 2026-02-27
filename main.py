from fastapi import FastAPI
from ranking_predicao import (
    obter_dados_e_preprocessar,
    treinar_e_agrupar_alunos,
    prever_grupo_aluno
)
from sklearn.preprocessing import StandardScaler
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

app = FastAPI()

scaler = StandardScaler()

df_final, cluster_metricas, metricas_normalizadas = \
    obter_dados_e_preprocessar(
        "PEDE_PASSOS_DATASET_FIAP.csv",
        scaler
    )

df_final, modelo_rf, kmeans = \
    treinar_e_agrupar_alunos(
        df_final,
        cluster_metricas,
        metricas_normalizadas
    )


@app.post("/prever")
def prever(dados: dict):
    return prever_grupo_aluno(
        dados,
        scaler,
        kmeans,
        modelo_rf,
        cluster_metricas
    )