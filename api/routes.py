import logging

from fastapi import APIRouter, Request

from api.schemas import PredictionRequest, PredictionResponse
from models.prediction import prever_grupo_aluno

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
