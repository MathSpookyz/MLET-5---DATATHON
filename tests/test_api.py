import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from fastapi.testclient import TestClient

from api.app import create_app


@pytest.fixture
def mock_artifacts():
    rng = np.random.RandomState(42)
    scaler = StandardScaler()
    scaler.fit(rng.uniform(0, 10, (20, 8)))

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(rng.uniform(0, 10, (20, 8)))

    modelo_rf = RandomForestClassifier(n_estimators=10, random_state=42)
    X = rng.uniform(0, 10, (20, 8))
    y = rng.randint(0, 2, 20)
    modelo_rf.fit(X, y)

    cluster_metricas = [
        "INDE_FINAL", "IAA_FINAL", "IEG_FINAL", "IPS_FINAL",
        "IDA_FINAL", "IPP_FINAL", "IPV_FINAL", "IAN_FINAL",
    ]
    return scaler, kmeans, modelo_rf, cluster_metricas


@pytest.fixture
def client(mock_artifacts):
    scaler, kmeans, modelo_rf, cluster_metricas = mock_artifacts
    with patch("api.app.load_model", return_value=mock_artifacts):
        app = create_app()
        with TestClient(app) as c:
            yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prever_valid(client):
    payload = {
        "INDE_FINAL": 5.0,
        "IAA_FINAL": 5.0,
        "IEG_FINAL": 5.0,
        "IPS_FINAL": 5.0,
        "IDA_FINAL": 5.0,
        "IPP_FINAL": 5.0,
        "IPV_FINAL": 5.0,
        "IAN_FINAL": 5.0,
    }
    response = client.post("/prever", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "Grupo" in body
    assert "Nivel" in body
    assert "Probabilidade_PV" in body
    assert 0.0 <= body["Probabilidade_PV"] <= 1.0


def test_prever_missing_field(client):
    payload = {"INDE_FINAL": 5.0}
    response = client.post("/prever", json=payload)
    assert response.status_code == 422


def test_prever_invalid_type(client):
    payload = {
        "INDE_FINAL": "not_a_number",
        "IAA_FINAL": 5.0,
        "IEG_FINAL": 5.0,
        "IPS_FINAL": 5.0,
        "IDA_FINAL": 5.0,
        "IPP_FINAL": 5.0,
        "IPV_FINAL": 5.0,
        "IAN_FINAL": 5.0,
    }
    response = client.post("/prever", json=payload)
    assert response.status_code == 422
