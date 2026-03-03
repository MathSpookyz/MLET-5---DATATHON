import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Classe responsável pela criação de novas variáveis (Feature Engineering).
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Tratamento de Strings e Nulos básicos
        # Remove espaços extras das colunas de texto
        cat_cols = X.select_dtypes(include=['object']).columns
        for col in cat_cols:
            X[col] = X[col].astype(str).str.strip()

        # 2. Criação de Features (Exemplos práticos para o Datathon)
        
        # Feature: Média dos indicadores Psicossociais (se existirem)
        # Agrupa a "saúde emocional" do aluno em uma métrica única
        psico_cols = ['IAA', 'IPS', 'IPP']
        valid_psico = [c for c in psico_cols if c in X.columns]
        if valid_psico:
            X['MEDIA_PSICO'] = X[valid_psico].mean(axis=1)
        
        # Feature: Defasagem Escolar (Transformar em booleano ou manter numérico)
        # Se 'Defasagem' < 0, o aluno está atrasado
        if 'Defasagem' in X.columns:
            X['EM_DEFASAGEM'] = np.where(X['Defasagem'] < 0, 1, 0)
            
        return X

def criar_pipeline_pre_processamento(X_train):
    """
    Cria o pipeline completo de pré-processamento (Numérico + Categórico).
    """
    
    # Separa colunas numéricas e categóricas automaticamente
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    # Pipeline Numérico: Preenche nulos com a mediana e padroniza a escala
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline Categórico: Preenche nulos com 'Desconhecido' e transforma em números (OneHot)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Desconhecido')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Junta tudo
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Adiciona a etapa de Feature Engineering antes do tratamento padrão
    pipeline_completo = Pipeline(steps=[
        ('feature_engineering', FeatureEngineering()),
        ('preprocessor', preprocessor)
    ])

    return pipeline_completo