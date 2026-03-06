import json
import logging
from fastapi import APIRouter, Request

from api.schemas import PredictionRequest, PredictionResponse
from models.prediction import prever_grupo_aluno
from models.ranking import ultimos_alunos, primeiros_alunos


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

    logger.info("Ultimos estudantes endpoint chamado", extra={"top": top})

    df_pred = request.app.state.df_preprocessed
    ranking = ultimos_alunos(df_pred, top)

    logger.info("Ultimos alunos retornados com sucesso", extra={"num_alunos": len(ranking)})

    return json.loads(ranking.to_json(orient="records"))


@router.get("/ranking/primeiros")
def primeiros_alunos_endpoint(request: Request, top: int = 10):

    logger.info("Primeiros estudantes endpoint chamado", extra={"top": top})

    df_pred = request.app.state.df_preprocessed
    ranking = primeiros_alunos(df_pred, top)

    logger.info("Primeiros alunos retornados com sucesso", extra={"num_alunos": len(ranking)})

    return json.loads(ranking.to_json(orient="records"))