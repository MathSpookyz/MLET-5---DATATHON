import logging

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
        lambda x: 1 if str(x).strip().upper() == "SIM" else 0
    )

    metricas = df[cluster_metricas]
    alvo = df["Alvo_PV"]

    X_train, X_test, y_train, y_test = train_test_split(
        metricas, alvo, test_size=0.2, random_state=42
    )

    modelo_rf = RandomForestClassifier(
        n_estimators=150, min_samples_leaf=4, max_features="sqrt", random_state=42
    )
    modelo_rf.fit(X_train, y_train)

    accuracy = modelo_rf.score(X_test, y_test)
    logger.info("RandomForest trained, test accuracy=%.4f", accuracy)

    df["Probabilidade_PV"] = modelo_rf.predict_proba(metricas)[:, 1]

    return df, modelo_rf, kmeans
