import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ===============================
# PRÉ-PROCESSAMENTO
# ===============================

def obter_dados_e_preprocessar(caminho_csv, scaler):
    df = pd.read_csv(caminho_csv, sep=';', encoding='utf-8-sig')

    metricas = ['INDE', 'IAA', 'IEG', 'IPS', 'IDA', 'IPP', 'IPV', 'IAN']

    for metrica in metricas:
        coluna_metrica = [c for c in df.columns if metrica in c]

        df[metrica + '_FINAL'] = df[coluna_metrica].apply(
            lambda x: x.dropna().iloc[-1] if x.dropna().any() else np.nan,
            axis=1
        )

        df[metrica + '_FINAL'] = pd.to_numeric(
            df[metrica + '_FINAL'], errors='coerce'
        )

        df[metrica + '_FINAL'] = df[metrica + '_FINAL'].fillna(
            df[metrica + '_FINAL'].median()
        )

    cluster_metricas = [m + '_FINAL' for m in metricas]

    metricas_normalizadas = scaler.fit_transform(df[cluster_metricas])

    return df, cluster_metricas, metricas_normalizadas


# ===============================
# TREINAMENTO E CLUSTERIZAÇÃO
# ===============================

def treinar_e_agrupar_alunos(df, cluster_metricas, metricas_normalizadas):

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    df['Grupo_ID'] = kmeans.fit_predict(metricas_normalizadas)

    ranking = (
        df.groupby('Grupo_ID')['INDE_FINAL']
        .mean()
        .sort_values()
        .index
        .tolist()
    )

    mapa_grupos = {idx: f"Nível {i+1}" for i, idx in enumerate(ranking)}
    df['Perfil_Nivel'] = df['Grupo_ID'].map(mapa_grupos)

    df['Alvo_PV'] = df['Atingiu PV'].apply(
        lambda x: 1 if str(x).strip().upper() == 'SIM' else 0
    )

    metricas = df[cluster_metricas]
    alvo = df['Alvo_PV']

    X_train, X_test, y_train, y_test = train_test_split(
        metricas, alvo, test_size=0.2, random_state=42
    )

    modelo_rf = RandomForestClassifier(
        n_estimators=150, random_state=42
    )

    modelo_rf.fit(X_train, y_train)

    df['Probabilidade_PV'] = modelo_rf.predict_proba(metricas)[:, 1]

    return df, modelo_rf, kmeans


# ===============================
# PREVISÃO NOVO ALUNO
# ===============================

def prever_grupo_aluno(
    novo_aluno_dados,
    scaler,
    kmeans,
    modelo_rf,
    cluster_metricas
):
    df_aluno = pd.DataFrame([novo_aluno_dados])

    aluno_normalizado = scaler.transform(df_aluno[cluster_metricas])

    grupo_previsto = int(kmeans.predict(aluno_normalizado)[0])

    prob_pv = float(
        modelo_rf.predict_proba(df_aluno[cluster_metricas])[0, 1]
    )

    resultado = {
        "Grupo": grupo_previsto,
        "Nivel": f"Nível {grupo_previsto + 1}",
        "Probabilidade_PV": round(prob_pv, 4)
    }

    return resultado


# ===============================
# RANKING INDIVIDUAL
# ===============================

def rankear_alunos_individual(df, cluster_metricas):

    df_ranking = df.copy()

    df_ranking['Score_Individual'] = df_ranking[
        cluster_metricas
    ].mean(axis=1)

    df_ranking['Ranking_Grupo'] = (
        df_ranking.groupby('Grupo_ID')['Score_Individual']
        .rank(ascending=False)
        .astype(int)
    )

    df_ranking['Ranking_Geral'] = (
        df_ranking['Score_Individual']
        .rank(ascending=False)
        .astype(int)
    )

    return df_ranking


# ===============================
# EXECUÇÃO SOMENTE SE RODAR DIRETO
# ===============================

if __name__ == "__main__":

    print("Executando modelo localmente...")

    scaler = StandardScaler()

    df_final, cluster_metricas, metricas_normalizadas = \
        obter_dados_e_preprocessar(
            "PEDE_PASSOS_DATASET_FIAP.csv",
            scaler
        )

    df_final, modelo, kmeans = \
        treinar_e_agrupar_alunos(
            df_final,
            cluster_metricas,
            metricas_normalizadas
        )

    print("Modelo treinado com sucesso!")