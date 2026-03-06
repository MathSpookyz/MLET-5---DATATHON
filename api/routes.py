import logging

from fastapi import APIRouter, Request

from api.schemas import PredictionRequest, PredictionResponse
from models.prediction import prever_grupo_aluno

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


# ✅ NOVO ENDPOINT DE OBSERVABILIDADE
@router.get("/metrics")
def metrics(request: Request):
    """
    Retorna metadados do modelo para monitoramento.
    """
    state = request.app.state

    return {
        "model": type(state.modelo_rf).__name__,
        "n_clusters": state.kmeans.n_clusters,
        "features": state.cluster_metricas,
        "n_estimators": state.modelo_rf.n_estimators,
    }


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