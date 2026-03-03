import json
import logging
import os

import joblib

logger = logging.getLogger(__name__)

SCALER_FILE = "scaler.joblib"
KMEANS_FILE = "kmeans.joblib"
RF_FILE = "modelo_rf.joblib"
METRICAS_FILE = "cluster_metricas.json"


def save_model(scaler, kmeans, modelo_rf, cluster_metricas, path="artifacts"):
    os.makedirs(path, exist_ok=True)

    joblib.dump(scaler, os.path.join(path, SCALER_FILE))
    joblib.dump(kmeans, os.path.join(path, KMEANS_FILE))
    joblib.dump(modelo_rf, os.path.join(path, RF_FILE))

    with open(os.path.join(path, METRICAS_FILE), "w") as f:
        json.dump(cluster_metricas, f)

    logger.info("Model artifacts saved to %s", path)


def load_model(path="artifacts"):
    files = [SCALER_FILE, KMEANS_FILE, RF_FILE, METRICAS_FILE]
    for fname in files:
        full = os.path.join(path, fname)
        if not os.path.exists(full):
            raise FileNotFoundError(f"Missing model artifact: {full}")

    scaler = joblib.load(os.path.join(path, SCALER_FILE))
    kmeans = joblib.load(os.path.join(path, KMEANS_FILE))
    modelo_rf = joblib.load(os.path.join(path, RF_FILE))

    with open(os.path.join(path, METRICAS_FILE), "r") as f:
        cluster_metricas = json.load(f)

    logger.info("Model artifacts loaded from %s", path)
    return scaler, kmeans, modelo_rf, cluster_metricas
