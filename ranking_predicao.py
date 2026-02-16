import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def obter_dados_e_preprocessar(caminho_csv, scaler):
    df = pd.read_csv(caminho_csv, sep=';', encoding='utf-8-sig')

    metricas = ['INDE', 'IAA', 'IEG', 'IPS', 'IDA', 'IPP', 'IPV', 'IAN']
    
    for metrica in metricas:
        coluna_metrica = [c for c in df.columns if metrica in c]
        df[metrica + '_FINAL'] = df[coluna_metrica].apply(lambda x: x.dropna().iloc[-1] if x.dropna().any() else np.nan, axis=1) # Pegamos o último valor não nulo
        df[metrica + '_FINAL'] = pd.to_numeric(df[metrica + '_FINAL'], errors='coerce')    #extrai valores para numerico ou NaN
        df[metrica + '_FINAL'] = df[metrica + '_FINAL'].fillna(df[metrica + '_FINAL'].median()) # Preenche NaN com a mediana da coluna

    cluster_metricas = [m + '_FINAL' for m in metricas]
    
    metricas_normalizadas = scaler.fit_transform(df[cluster_metricas])
    
    return df, cluster_metricas, metricas_normalizadas

def treinar_e_agrupar_alunos(df, cluster_metricas, metricas_normalizadas):

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    df['Grupo_ID'] = kmeans.fit_predict(metricas_normalizadas)

    def obter_ranking_grupos(dataframe):
        # Calculamos a média do INDE por grupo para definir quem é o "melhor"
        ranking = dataframe.groupby('Grupo_ID')['INDE_FINAL'].mean().sort_values().index.tolist()
        
        mapa_grupos = {idx: f"Nível {i+1}" for i, idx in enumerate(ranking)}
        dataframe['Perfil_Nivel'] = dataframe['Grupo_ID'].map(mapa_grupos)
        
        print("\n--- Ranking de Grupos (Média INDE) ---")
        for i, idx in enumerate(ranking):
            print(f"Lugar {i+1} (Pior p/ Melhor): Grupo {idx} - Média INDE: {dataframe[dataframe['Grupo_ID']==idx]['INDE_FINAL'].mean():.2f}")
        
        return dataframe

    df = obter_ranking_grupos(df)

    coluna_alvo = 'Atingiu PV'
    df['Alvo_PV'] = df[coluna_alvo].apply(lambda x: 1 if str(x).strip().upper() == 'SIM' else 0)

    metricas = df[cluster_metricas]
    alvo = df['Alvo_PV']

    massa_treino, massa_teste, resposta_treino, resposta_teste = train_test_split(metricas, alvo, test_size=0.2, random_state=42)
    
    modelo_rf = RandomForestClassifier(n_estimators=150, random_state=42)
    modelo_rf.fit(massa_treino, resposta_treino)

    df['Probabilidade_PV'] = modelo_rf.predict_proba(metricas)[:, 1]

    alunos_baixo_rank = df[df['Alvo_PV'] == 0].sort_values(by=['Probabilidade_PV', 'INDE_FINAL']).head(10)

    print("\n--- Top 10 Alunos com Maior Risco / Menor Evolução ---")
    print(alunos_baixo_rank[['Nome', 'Perfil_Nivel', 'INDE_FINAL', 'Probabilidade_PV']])

    return df, modelo_rf, kmeans


# ==================================

def gerar_elbow_curve(metricas_normalizadas, k_max=10):
    """
    Gera a curva de cotovelo (Elbow Curve) para determinar o melhor número de clusters.
    """
    inercias = []
    silhuetas = []
    ks = range(2, k_max + 1)
    
    for k in ks:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
        kmeans_temp.fit(metricas_normalizadas)
        inercias.append(kmeans_temp.inertia_)
        
        from sklearn.metrics import silhouette_score
        score = silhouette_score(metricas_normalizadas, kmeans_temp.labels_)
        silhuetas.append(score)
        print(f"K={k}: Inércia={kmeans_temp.inertia_:.2f}, Silhueta={score:.4f}")
    
    # Plotar ambas as métricas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow Curve
    ax1.plot(ks, inercias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Número de Clusters (K)', fontsize=12)
    ax1.set_ylabel('Inércia', fontsize=12)
    ax1.set_title('Elbow Curve - KMeans', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Silhueta Score
    ax2.plot(ks, silhuetas, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Número de Clusters (K)', fontsize=12)
    ax2.set_ylabel('Silhueta Score', fontsize=12)
    ax2.set_title('Silhueta Score - KMeans', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return inercias, silhuetas

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualizar_distribuicao_grupos(df, metricas_normalizadas):

    pca = PCA(n_components=2)
    metricas_2d = pca.fit_transform(metricas_normalizadas)
    
    plt.figure(figsize=(12, 8))
    
    grupos_unicos = df['Grupo_ID'].unique()
    cores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for i, grupo in enumerate(sorted(grupos_unicos)):
        mascara = df['Grupo_ID'] == grupo
        plt.scatter(
            metricas_2d[mascara, 0],
            metricas_2d[mascara, 1],
            c=cores[i],
            label=f"{df[df['Grupo_ID']==grupo]['Perfil_Nivel'].iloc[0]} (Grupo {grupo})",
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variância)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variância)')
    plt.title('Distribuição de Alunos por Grupo (KMeans)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def prever_grupo_aluno(novo_aluno_dados, scaler, kmeans, modelo_rf, cluster_metricas):
    df_aluno = pd.DataFrame([novo_aluno_dados])
    
    aluno_normalizado = scaler.transform(df_aluno[cluster_metricas])
    
    grupo_previsto = kmeans.predict(aluno_normalizado)[0]
    
    mapa_niveis = {
        0: "Nível 1",
        1: "Nível 2",
        2: "Nível 3",
        3: "Nível 4"
    }
    
    prob_pv = modelo_rf.predict_proba(df_aluno[cluster_metricas])[0, 1]
    
    resultado = {
        'Grupo': grupo_previsto,
        'Nivel': mapa_niveis.get(grupo_previsto, f"Grupo {grupo_previsto}"),
        'Probabilidade_PV': f"{prob_pv:.2%}",
        'Metricas': novo_aluno_dados
    }
    
    return resultado


def rankear_alunos_individual(df, cluster_metricas):
    """
    Calcula um ranking individual para cada aluno dentro do seu grupo.
    """
    df_ranking = df.copy()
    
    # Calcula score individual baseado na média das métricas
    metricas_numericas = [col for col in cluster_metricas if col in df.columns]
    df_ranking['Score_Individual'] = df_ranking[metricas_numericas].mean(axis=1)
    
    # Rankeia dentro de cada grupo
    df_ranking['Ranking_Grupo'] = df_ranking.groupby('Grupo_ID')['Score_Individual'].rank(ascending=False).astype(int)
    
    # Ranking geral
    df_ranking['Ranking_Geral'] = df_ranking['Score_Individual'].rank(ascending=False).astype(int)
    
    return df_ranking[['Nome', 'Grupo_ID', 'Perfil_Nivel', 'Score_Individual', 'Ranking_Grupo', 'Ranking_Geral', 'Probabilidade_PV']]



scaler = StandardScaler()

df_final, cluster_metricas, metricas_normalizadas = obter_dados_e_preprocessar("PEDE_PASSOS_DATASET_FIAP.csv", scaler)

print("Gerando Elbow Curve...")
inercias, silhuetas = gerar_elbow_curve(metricas_normalizadas, k_max=10)


df_final, modelo, kmeans = treinar_e_agrupar_alunos(df_final, cluster_metricas, metricas_normalizadas)

# 1. Visualizar distribuição dos grupos
visualizar_distribuicao_grupos(df_final, metricas_normalizadas)

# 2. Rankear alunos individualmente
ranking = rankear_alunos_individual(df_final, cluster_metricas)
print(ranking.head(20))

# 3. Prever grupo de um novo aluno
novo_aluno = {
    'INDE_FINAL': 7.5,
    'IAA_FINAL': 8.0,
    'IEG_FINAL': 7.2,
    'IPS_FINAL': 6.8,
    'IDA_FINAL': 7.9,
    'IPP_FINAL': 8.1,
    'IPV_FINAL': 7.4,
    'IAN_FINAL': 6.5
}

# Use o kmeans retornado da função, NÃO crie um novo
resultado = prever_grupo_aluno(novo_aluno, scaler, kmeans, modelo, cluster_metricas)
print(resultado)