import json
import numpy as np
import pandas as pd
import pytest

from models.ranking import rankear_alunos_individual, ultimos_alunos, primeiros_alunos


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


# --- Fixtures for ultimos/primeiros ---

@pytest.fixture
def ranking_df():
    """DataFrame with Probabilidade_PV, including NaN values."""
    return pd.DataFrame({
        "NOME": [f"Aluno_{i}" for i in range(15)],
        "Probabilidade_PV": [0.1, 0.5, np.nan, 0.9, 0.3, 0.2, 0.8, 0.7, 0.4, 0.6, 0.05, 0.95, np.nan, 0.15, 0.85],
    })


# --- Tests for ultimos_alunos ---

def test_ultimos_returns_correct_count(ranking_df):
    result = ultimos_alunos(ranking_df, top_n=5)
    assert len(result) == 5


def test_ultimos_sorted_ascending(ranking_df):
    result = ultimos_alunos(ranking_df, top_n=5)
    probs = result["Probabilidade_PV"].dropna().tolist()
    assert probs == sorted(probs)


def test_ultimos_raises_without_column():
    df = pd.DataFrame({"NOME": ["A", "B"]})
    with pytest.raises(ValueError, match="Probabilidade_PV"):
        ultimos_alunos(df)


def test_ultimos_default_top(ranking_df):
    result = ultimos_alunos(ranking_df)
    assert len(result) == 10


def test_ultimos_json_serializable(ranking_df):
    """Ensure result with NaN can be serialized to JSON (the original bug)."""
    result = ultimos_alunos(ranking_df, top_n=15)
    records = json.loads(result.to_json(orient="records"))
    assert isinstance(records, list)
    # NaN should become None/null, not the string "NaN"
    for rec in records:
        for v in rec.values():
            assert v != float("inf") and v != float("-inf")


# --- Tests for primeiros_alunos ---

def test_primeiros_returns_correct_count(ranking_df):
    result = primeiros_alunos(ranking_df, top_n=5)
    assert len(result) == 5


def test_primeiros_sorted_descending(ranking_df):
    result = primeiros_alunos(ranking_df, top_n=5)
    probs = result["Probabilidade_PV"].dropna().tolist()
    assert probs == sorted(probs, reverse=True)


def test_primeiros_raises_without_column():
    df = pd.DataFrame({"NOME": ["A", "B"]})
    with pytest.raises(ValueError, match="Probabilidade_PV"):
        primeiros_alunos(df)


def test_primeiros_default_top(ranking_df):
    result = primeiros_alunos(ranking_df)
    assert len(result) == 10


def test_primeiros_json_serializable(ranking_df):
    """Ensure result with NaN can be serialized to JSON (the original bug)."""
    result = primeiros_alunos(ranking_df, top_n=15)
    records = json.loads(result.to_json(orient="records"))
    assert isinstance(records, list)
    for rec in records:
        for v in rec.values():
            assert v != float("inf") and v != float("-inf")
