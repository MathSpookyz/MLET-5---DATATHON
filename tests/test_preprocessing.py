import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from models.preprocessing import obter_dados_e_preprocessar


@pytest.fixture
def sample_csv(tmp_path):
    data = {
        "INDE 22": [5.0, np.nan, 7.0, 6.0, 8.0],
        "INDE 2023": [6.0, 7.0, np.nan, 5.5, 9.0],
        "IAA": [3.0, 4.0, 5.0, 3.5, 4.5],
        "IEG": [2.0, 3.0, 4.0, 2.5, 3.5],
        "IPS": [1.0, 2.0, 3.0, 1.5, 2.5],
        "IDA": [4.0, 5.0, 6.0, 4.5, 5.5],
        "IPP": [7.0, 8.0, 9.0, 7.5, 8.5],
        "IPV": [2.0, 3.0, 4.0, 2.5, 3.5],
        "IAN": [1.0, 1.5, 2.0, 1.2, 1.8],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "test_data.csv"
    df.to_csv(path, sep=";", index=False, encoding="utf-8-sig")
    return str(path)


def test_output_has_final_columns(sample_csv):
    scaler = StandardScaler()
    df, cluster_metricas, _ = obter_dados_e_preprocessar(sample_csv, scaler)
    expected = [
        "INDE_FINAL", "IAA_FINAL", "IEG_FINAL", "IPS_FINAL",
        "IDA_FINAL", "IPP_FINAL", "IPV_FINAL", "IAN_FINAL",
    ]
    for col in expected:
        assert col in df.columns


def test_no_nans_in_final_columns(sample_csv):
    scaler = StandardScaler()
    df, cluster_metricas, _ = obter_dados_e_preprocessar(sample_csv, scaler)
    for col in cluster_metricas:
        assert df[col].isna().sum() == 0


def test_scaler_is_fitted(sample_csv):
    scaler = StandardScaler()
    obter_dados_e_preprocessar(sample_csv, scaler)
    assert hasattr(scaler, "mean_")
    assert len(scaler.mean_) == 8


def test_normalized_shape(sample_csv):
    scaler = StandardScaler()
    df, cluster_metricas, metricas_normalizadas = obter_dados_e_preprocessar(
        sample_csv, scaler
    )
    assert metricas_normalizadas.shape == (len(df), 8)


def test_cluster_metricas_list(sample_csv):
    scaler = StandardScaler()
    _, cluster_metricas, _ = obter_dados_e_preprocessar(sample_csv, scaler)
    assert isinstance(cluster_metricas, list)
    assert len(cluster_metricas) == 8
    assert all(m.endswith("_FINAL") for m in cluster_metricas)
