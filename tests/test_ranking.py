import numpy as np
import pandas as pd
import pytest

from models.ranking import rankear_alunos_individual


@pytest.fixture
def sample_df():
    rng = np.random.RandomState(42)
    n = 30
    cluster_metricas = [
        "INDE_FINAL", "IAA_FINAL", "IEG_FINAL", "IPS_FINAL",
        "IDA_FINAL", "IPP_FINAL", "IPV_FINAL", "IAN_FINAL",
    ]
    data = {m: rng.uniform(1, 10, n) for m in cluster_metricas}
    data["Grupo_ID"] = rng.choice([0, 1, 2, 3], n)
    return pd.DataFrame(data), cluster_metricas


def test_ranking_columns_added(sample_df):
    df, cluster_metricas = sample_df
    result = rankear_alunos_individual(df, cluster_metricas)
    assert "Score_Individual" in result.columns
    assert "Ranking_Grupo" in result.columns
    assert "Ranking_Geral" in result.columns


def test_ranking_geral_contiguous(sample_df):
    df, cluster_metricas = sample_df
    result = rankear_alunos_individual(df, cluster_metricas)
    rankings = sorted(result["Ranking_Geral"].tolist())
    assert rankings == list(range(1, len(df) + 1))


def test_rankings_are_ints(sample_df):
    df, cluster_metricas = sample_df
    result = rankear_alunos_individual(df, cluster_metricas)
    assert result["Ranking_Grupo"].dtype in [np.int32, np.int64]
    assert result["Ranking_Geral"].dtype in [np.int32, np.int64]


def test_original_df_unchanged(sample_df):
    df, cluster_metricas = sample_df
    original_cols = set(df.columns)
    rankear_alunos_individual(df, cluster_metricas)
    assert set(df.columns) == original_cols


def test_score_is_mean_of_metrics(sample_df):
    df, cluster_metricas = sample_df
    result = rankear_alunos_individual(df, cluster_metricas)
    expected = df[cluster_metricas].mean(axis=1)
    pd.testing.assert_series_equal(
        result["Score_Individual"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )
