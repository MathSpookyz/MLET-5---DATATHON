import logging

from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def treinar_e_agrupar_alunos(df, cluster_metricas, metricas_normalizadas):
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20, max_iter=500, tol=1e-4)
    df["Grupo_ID"] = kmeans.fit_predict(metricas_normalizadas)
    logger.info("KMeans fitted with 4 clusters")

    ranking = (
        df.groupby("Grupo_ID")["INDE_FINAL"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )

    mapa_grupos = {idx: f"Nível {i+1}" for i, idx in enumerate(ranking)}
    df["Perfil_Nivel"] = df["Grupo_ID"].map(mapa_grupos)
    logger.info("Cluster-to-level mapping: %s", mapa_grupos)

    df["Alvo_PV"] = df["Atingiu PV"].apply(
        lambda x: 1 if str(x).strip().upper() == "SIM"
        else (0 if str(x).strip().upper() == "NÃO" or str(x).strip().upper() == "NAO"
              else None)
    )

    df_rf = df.dropna(subset=["Alvo_PV"]).copy()
    df_rf["Alvo_PV"] = df_rf["Alvo_PV"].astype(int)
    logger.info(
        "RF training set: %d rows (Sim=%d, Não=%d)",
        len(df_rf),
        (df_rf["Alvo_PV"] == 1).sum(),
        (df_rf["Alvo_PV"] == 0).sum(),
    )

    metricas = df_rf[cluster_metricas]
    alvo = df_rf["Alvo_PV"]

    X_train, X_test, y_train, y_test = train_test_split(
        metricas, alvo, test_size=0.2, stratify=alvo, random_state=42
    )

    logger.info("Distribuição antes do SMOTE: %s", y_train.value_counts().to_dict())

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    logger.info("Distribuição depois do SMOTE: %s", y_train_res.value_counts().to_dict())

    modelo_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )
    modelo_rf.fit(X_train_res, y_train_res)

    accuracy = modelo_rf.score(X_test, y_test)
    logger.info("RandomForest trained, test accuracy=%.4f", accuracy)

    df["Probabilidade_PV"] = modelo_rf.predict_proba(df[cluster_metricas])[:, 1]

    return df, modelo_rf, kmeans
