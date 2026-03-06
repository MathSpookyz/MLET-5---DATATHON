import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from models.preprocessing import obter_dados_e_preprocessar
from models.training import treinar_e_agrupar_alunos


@pytest.fixture
def training_data(tmp_path):
    rng = np.random.RandomState(42)
    n = 60
    data = {
        "INDE": rng.uniform(2, 10, n),
        "IAA": rng.uniform(1, 10, n),
        "IEG": rng.uniform(1, 10, n),
        "IPS": rng.uniform(1, 10, n),
        "IDA": rng.uniform(1, 10, n),
        "IPP": rng.uniform(1, 10, n),
        "IPV": rng.uniform(1, 10, n),
        "IAN": rng.uniform(1, 10, n),
        "Atingiu PV": rng.choice(["Sim", "Não"], n),
    }
    df = pd.DataFrame(data)
    path = tmp_path / "train_data.csv"
    df.to_csv(path, sep=";", index=False, encoding="utf-8-sig")

    scaler = StandardScaler()
    df_proc, cluster_metricas, metricas_norm = obter_dados_e_preprocessar(
        str(path), scaler
    )
    return df_proc, cluster_metricas, metricas_norm


def test_columns_added(training_data):
    df, cluster_metricas, metricas_norm = training_data
    df_result, _, _ = treinar_e_agrupar_alunos(df, cluster_metricas, metricas_norm)
    for col in ["Grupo_ID", "Perfil_Nivel", "Alvo_PV", "Probabilidade_PV"]:
        assert col in df_result.columns


def test_kmeans_has_4_clusters(training_data):
    df, cluster_metricas, metricas_norm = training_data
    _, _, kmeans = treinar_e_agrupar_alunos(df, cluster_metricas, metricas_norm)
    assert kmeans.n_clusters == 4


def test_model_is_fitted(training_data):
    df, cluster_metricas, metricas_norm = training_data
    _, modelo_rf, _ = treinar_e_agrupar_alunos(df, cluster_metricas, metricas_norm)
    assert hasattr(modelo_rf, "classes_")


def test_probability_range(training_data):
    df, cluster_metricas, metricas_norm = training_data
    df_result, _, _ = treinar_e_agrupar_alunos(df, cluster_metricas, metricas_norm)
    assert df_result["Probabilidade_PV"].between(0, 1).all()


def test_perfil_nivel_values(training_data):
    df, cluster_metricas, metricas_norm = training_data
    df_result, _, _ = treinar_e_agrupar_alunos(df, cluster_metricas, metricas_norm)
    allowed = {"Nível 1", "Nível 2", "Nível 3", "Nível 4"}
    assert set(df_result["Perfil_Nivel"].unique()).issubset(allowed)
