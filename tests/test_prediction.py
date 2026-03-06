import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from models.prediction import prever_grupo_aluno


@pytest.fixture
def fitted_objects():
    rng = np.random.RandomState(42)
    n = 50
    cluster_metricas = [
        "INDE_FINAL", "IAA_FINAL", "IEG_FINAL", "IPS_FINAL",
        "IDA_FINAL", "IPP_FINAL", "IPV_FINAL", "IAN_FINAL",
    ]
    X = rng.uniform(1, 10, (n, 8))
    y = rng.randint(0, 2, n)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    df_train = pd.DataFrame(X, columns=cluster_metricas)
    modelo_rf = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo_rf.fit(df_train, y)

    return scaler, kmeans, modelo_rf, cluster_metricas


def test_prediction_keys(fitted_objects):
    scaler, kmeans, modelo_rf, cluster_metricas = fitted_objects
    dados = {m: 5.0 for m in cluster_metricas}
    result = prever_grupo_aluno(dados, scaler, kmeans, modelo_rf, cluster_metricas)
    assert "Grupo" in result
    assert "Nivel" in result
    assert "Probabilidade_PV" in result


def test_probability_between_0_and_1(fitted_objects):
    scaler, kmeans, modelo_rf, cluster_metricas = fitted_objects
    dados = {m: 5.0 for m in cluster_metricas}
    result = prever_grupo_aluno(dados, scaler, kmeans, modelo_rf, cluster_metricas)
    assert 0.0 <= result["Probabilidade_PV"] <= 1.0


def test_grupo_is_int(fitted_objects):
    scaler, kmeans, modelo_rf, cluster_metricas = fitted_objects
    dados = {m: 5.0 for m in cluster_metricas}
    result = prever_grupo_aluno(dados, scaler, kmeans, modelo_rf, cluster_metricas)
    assert isinstance(result["Grupo"], int)


def test_nivel_format(fitted_objects):
    scaler, kmeans, modelo_rf, cluster_metricas = fitted_objects
    dados = {m: 5.0 for m in cluster_metricas}
    result = prever_grupo_aluno(dados, scaler, kmeans, modelo_rf, cluster_metricas)
    assert result["Nivel"].startswith("Nível ")


def test_missing_field_raises(fitted_objects):
    scaler, kmeans, modelo_rf, cluster_metricas = fitted_objects
    dados = {"INDE_FINAL": 5.0}
    with pytest.raises(KeyError):
        prever_grupo_aluno(dados, scaler, kmeans, modelo_rf, cluster_metricas)
