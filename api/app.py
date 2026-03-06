import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sklearn.preprocessing import StandardScaler

from api.routes import router
from models.model_io import load_model, save_model
from models.preprocessing import obter_dados_e_preprocessar
from models.training import treinar_e_agrupar_alunos

logger = logging.getLogger(__name__)

CSV_PATH = "dataset/PEDE_PASSOS_DATASET_FIAP.csv"
ARTIFACTS_PATH = "artifacts"


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        try:
            scaler, kmeans, modelo_rf, cluster_metricas = load_model(ARTIFACTS_PATH)
            logger.info("Loaded pre-trained model artifacts")
        except FileNotFoundError:
            logger.info("No saved artifacts found, training from scratch")
            scaler = StandardScaler()
            df, cluster_metricas, metricas_normalizadas = obter_dados_e_preprocessar(
                CSV_PATH, scaler
            )
            df, modelo_rf, kmeans = treinar_e_agrupar_alunos(
                df, cluster_metricas, metricas_normalizadas
            )
            save_model(scaler, kmeans, modelo_rf, cluster_metricas, ARTIFACTS_PATH)

        app.state.scaler = scaler
        app.state.kmeans = kmeans
        app.state.modelo_rf = modelo_rf
        app.state.cluster_metricas = cluster_metricas
        yield

    app = FastAPI(title="Pede Passos - Prediction API", lifespan=lifespan)
    app.include_router(router)
    return app

app = create_app()