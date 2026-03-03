import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from models.model_io import load_model, save_model


@pytest.fixture
def model_artifacts():
    rng = np.random.RandomState(42)
    scaler = StandardScaler()
    scaler.fit(rng.uniform(0, 10, (20, 8)))

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(rng.uniform(0, 10, (20, 8)))

    modelo_rf = RandomForestClassifier(n_estimators=10, random_state=42)
    modelo_rf.fit(rng.uniform(0, 10, (20, 8)), rng.randint(0, 2, 20))

    cluster_metricas = [
        "INDE_FINAL", "IAA_FINAL", "IEG_FINAL", "IPS_FINAL",
        "IDA_FINAL", "IPP_FINAL", "IPV_FINAL", "IAN_FINAL",
    ]
    return scaler, kmeans, modelo_rf, cluster_metricas


def test_save_and_load_roundtrip(tmp_path, model_artifacts):
    scaler, kmeans, modelo_rf, cluster_metricas = model_artifacts
    path = str(tmp_path / "artifacts")

    save_model(scaler, kmeans, modelo_rf, cluster_metricas, path)

    loaded_scaler, loaded_kmeans, loaded_rf, loaded_metricas = load_model(path)

    np.testing.assert_array_equal(loaded_scaler.mean_, scaler.mean_)
    assert loaded_kmeans.n_clusters == kmeans.n_clusters
    assert loaded_metricas == cluster_metricas
    assert hasattr(loaded_rf, "classes_")


def test_load_missing_raises(tmp_path):
    path = str(tmp_path / "nonexistent")
    with pytest.raises(FileNotFoundError):
        load_model(path)


def test_save_creates_directory(tmp_path, model_artifacts):
    scaler, kmeans, modelo_rf, cluster_metricas = model_artifacts
    path = str(tmp_path / "new_dir" / "artifacts")
    save_model(scaler, kmeans, modelo_rf, cluster_metricas, path)
    loaded = load_model(path)
    assert loaded is not None


def test_saved_files_exist(tmp_path, model_artifacts):
    scaler, kmeans, modelo_rf, cluster_metricas = model_artifacts
    path = tmp_path / "artifacts"
    save_model(scaler, kmeans, modelo_rf, cluster_metricas, str(path))
    assert (path / "scaler.joblib").exists()
    assert (path / "kmeans.joblib").exists()
    assert (path / "modelo_rf.joblib").exists()
    assert (path / "cluster_metricas.json").exists()
