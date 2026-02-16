import pandas as pd
import numpy as np

def extrair_dados_excel(caminho_arquivo, arquivo_saida):
    print(f"Iniciando extração do arquivo: {caminho_arquivo}")
    
    try:
        excel = pd.ExcelFile(caminho_arquivo)
        abas = excel.sheet_names
        print(f"Abas encontradas: {abas}")
    except Exception as e:
        print(f"Erro ao abrir o arquivo Excel: {e}")
        return
    
    lista_dfs = []
    
    for aba in abas:
        print(f"Lendo aba: {aba}...")
        df_temp = pd.read_excel(caminho_arquivo, sheet_name=aba, na_values=['#N/A', '#DIV/0!', ''])
        
        # Adicionamos uma coluna para identificar a origem do dado (Ano/Aba)
        df_temp['ANO_ORIGEM'] = aba
        
        lista_dfs.append(df_temp)

    df_unificado = pd.concat(lista_dfs, axis=0, ignore_index=True, sort=False)

    def limpar_valor(series):
        if series.dtype == 'object':
            series_num = pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')
        else:
            series_num = pd.to_numeric(series, errors='coerce')
        
        if not series_num.dropna().empty and not any(isinstance(x, str) and not x.replace('.','').isdigit() for x in series.dropna()[:10]):
            return series_num.fillna(series_num.median())
        else:
            return series.fillna("")

    print("Aplicando limpeza e preenchimento (Mediana para números, '' para textos)...")
    for col in df_unificado.columns:
        df_unificado[col] = limpar_valor(df_unificado[col])

    df_unificado.to_csv(arquivo_saida, index=False, sep=';', encoding='utf-8-sig')
    print(f"\nSucesso! Arquivo único gerado: {arquivo_saida}")
    print(f"Total de registros: {len(df_unificado)}")
    print(f"Total de colunas: {len(df_unificado.columns)}")

# Execução do processo
extrair_dados_excel("BASE DE DADOS PEDE 2024 - DATATHON.xlsx", "PEDE_PASSOS_DATASET_FIAP.csv")