import logging
import pandas as pd
from fastapi import APIRouter, Request

from api.schemas import PredictionRequest, PredictionResponse
from models.prediction import prever_grupo_aluno
from models.ranking import ultimos_alunos
from models.prediction import prever_todos_alunos


logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/prever", response_model=PredictionResponse)
def prever(dados: PredictionRequest, request: Request):
    state = request.app.state
    resultado = prever_grupo_aluno(
        dados.model_dump(),
        state.scaler,
        state.kmeans,
        state.modelo_rf,
        state.cluster_metricas,
    )
    return resultado

@router.get("/ranking/ultimos")
def ultimos_alunos_endpoint(request: Request, top: int = 10):

    logger.info("ultimos estudantes endpoint chamado", extra={"top": top})

    scaler = request.app.state.scaler
    kmeans = request.app.state.kmeans
    modelo_rf = request.app.state.modelo_rf
    cluster_metricas = request.app.state.cluster_metricas

    df = pd.read_csv("dataset/PEDE_PASSOS_DATASET_FIAP.csv")

    df_pred = prever_todos_alunos(
        df,
        scaler,
        kmeans,
        modelo_rf,
        cluster_metricas
    )

    ranking = ultimos_alunos(df_pred, top)

    logger.info("Últimos alunos retornados com sucesso", extra={"num_alunos": len(ranking)})

    return ranking.to_dict(orient="records")