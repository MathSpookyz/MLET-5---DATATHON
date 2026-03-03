import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Importando nosso módulo local
from preprocessamento import criar_pipeline_pre_processamento

# --- CONFIGURAÇÕES ---
ARQUIVO_DADOS = 'BASE_DE_DADOS_PEDE_2024-DATATHON.xlsx' 
TARGET = 'INDE 22'  # <--- CORRIGIDO: O alvo é o INDE de 2022
NOME_MODELO = 'modelo_previsao_inde.pkl'

def carregar_dados(caminho):
    print(f"Carregando dados de: {caminho}")
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    if caminho.endswith('.xlsx'):
        df = pd.read_excel(caminho)
    else:
        df = pd.read_csv(caminho)
    return df

def main():
    # 1. Carga de Dados
    try:
        df = carregar_dados(ARQUIVO_DADOS)
        print(f"Dados carregados com sucesso! Dimensões: {df.shape}")
    except Exception as e:
        print(f"Erro ao carregar arquivo: {e}")
        return

    # 2. Seleção Inicial de Variáveis
    # Ajustamos os nomes baseados na lista que você me mandou
    colunas_para_remover = [
        'RA', 'Nome', 'Turma',  # Identificadores
        'Pedra 22',             # É a classificação final (resposta vazada)
        'Pedra 21', 'Pedra 20', # Histórico de pedras pode enviesar ou não estar disponível no futuro
        TARGET                  # O próprio target (INDE 22)
    ]
    
    # Filtrar colunas que realmente existem no DF
    drop_cols = [c for c in colunas_para_remover if c in df.columns]
    
    print(f"Removendo colunas: {drop_cols}")
    X = df.drop(columns=drop_cols)
    y = df[TARGET]

    # Tratamento para garantir que o Target não tenha Nulos
    if y.isnull().sum() > 0:
        print(f"Aviso: Removendo {y.isnull().sum()} linhas sem o valor do Target (INDE).")
        X = X[y.notnull()]
        y = y[y.notnull()]

    # 3. Divisão Treino e Teste
    print("Dividindo dados em Treino e Teste...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Construção da Pipeline
    print("Construindo Pipeline...")
    try:
        pipeline_processamento = criar_pipeline_pre_processamento(X_train)
    except Exception as e:
        print(f"Erro ao criar pipeline de pré-processamento: {e}")
        return

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    
    pipeline_final = Pipeline(steps=[
        ('preprocessor', pipeline_processamento),
        ('model', modelo)
    ])

    # 5. Treinamento
    print("Iniciando treinamento...")
    pipeline_final.fit(X_train, y_train)

    # 6. Validação Cruzada
    print("Executando Validação Cruzada (K-Fold)...")
    cv_scores = cross_val_score(pipeline_final, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    rmse_cv = -cv_scores.mean()
    print(f"RMSE Médio na Validação Cruzada: {rmse_cv:.4f}")

    # 7. Avaliação Final no Teste
    y_pred = pipeline_final.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_test = r2_score(y_test, y_pred)

    print("\n" + "="*30)
    print("RESULTADOS FINAIS DO MODELO")
    print("="*30)
    print(f"RMSE (Erro Médio) no Teste: {rmse_test:.4f}")
    print(f"R² (Explicação da Variância): {r2_test:.4f}")
    print("="*30)

    # Análise de Importância das Features
    try:
        # Tenta pegar nomes das features. Se usar OneHotEncoder, o nome muda.
        if hasattr(pipeline_final.named_steps['preprocessor'], 'get_feature_names_out'):
             feature_names = pipeline_final.named_steps['preprocessor'].get_feature_names_out()
        else:
             # Fallback genérico se não conseguir pegar os nomes exatos
             feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]

        importances = pipeline_final.named_steps['model'].feature_importances_
        
        # Só imprime se os tamanhos baterem
        if len(feature_names) == len(importances):
            feat_importances = pd.Series(importances, index=feature_names)
            print("\nTop 5 Variáveis mais importantes para prever o INDE:")
            print(feat_importances.nlargest(5))
    except Exception as e:
        print(f"Nota: Não foi possível detalhar a importância das features: {e}")

    # 8. Salvando o Modelo
    print(f"\nSalvando modelo em {NOME_MODELO}...")
    joblib.dump(pipeline_final, NOME_MODELO)
    print("Modelo salvo com sucesso!")

if __name__ == "__main__":
    main()