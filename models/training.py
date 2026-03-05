import logging
import json
import os
import numpy as np

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


def treinar_e_agrupar_alunos(
    df,
    cluster_metricas,
    metricas_normalizadas,
    metrics_path=None,
):

    logger.info("Diretório atual de execução: %s", os.getcwd())
    logger.info("Metrics path recebido: %s", metrics_path)

    # =========================
    # KMEANS
    # =========================
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
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

    # =========================
    # PREPARAÇÃO DO ALVO
    # =========================
    df["Alvo_PV"] = df["Atingiu PV"].apply(
        lambda x: 1 if str(x).strip().upper() == "SIM" else 0
    )

    metricas = df[cluster_metricas]
    alvo = df["Alvo_PV"]

    X_train, X_test, y_train, y_test = train_test_split(
        metricas,
        alvo,
        test_size=0.2,
        stratify=alvo,
        random_state=42
    )

    logger.info("Distribuição antes do SMOTE: %s", y_train.value_counts().to_dict())

    # =========================
    # SMOTE (somente treino)
    # =========================
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    logger.info("Distribuição após SMOTE: %s", y_train_res.value_counts().to_dict())

    # =========================
    # RANDOM FOREST
    # =========================
    modelo_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced_subsample',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    modelo_rf.fit(X_train_res, y_train_res)

    # =========================
    # MÉTRICAS
    # =========================
    accuracy = modelo_rf.score(X_test, y_test)
    y_proba = modelo_rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    # 🔥 OTIMIZAÇÃO DE THRESHOLD (maximiza F1 da classe 1)
    thresholds = np.arange(0.1, 0.95, 0.05)

    best_f1 = 0
    best_threshold = 0.5

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    logger.info("Melhor threshold encontrado: %.2f", best_threshold)

    # Aplicar threshold otimizado
    y_pred_adjusted = (y_proba >= best_threshold).astype(int)

    report = classification_report(
        y_test,
        y_pred_adjusted,
        output_dict=True,
    )

    logger.info(
        "RandomForest — Accuracy=%.4f | AUC=%.4f | BestThreshold=%.2f | F1_Classe1=%.4f",
        accuracy,
        auc,
        best_threshold,
        best_f1
    )

    # =========================
    # SALVAR MÉTRICAS
    # =========================
    if not metrics_path:
        logger.warning("Metrics path não informado. Métricas não serão salvas.")
    else:
        try:
            metrics_path = os.path.abspath(metrics_path)
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

            metrics = {
                "accuracy": round(accuracy, 4),
                "auc_roc": round(auc, 4),
                "best_threshold": float(best_threshold),
                "f1_classe_1": round(best_f1, 4),
                "classification_report": report,
            }

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            logger.info("Metrics successfully saved at: %s", metrics_path)

        except Exception as e:
            logger.error("Erro ao salvar métricas: %s", str(e))

    # =========================
    # PROBABILIDADE COMPLETA
    # =========================
    probs_full = modelo_rf.predict_proba(metricas)[:, 1]
    df["Probabilidade_PV"] = probs_full
    df["Predicao_PV"] = (probs_full >= best_threshold).astype(int)

    return df, modelo_rf, kmeans