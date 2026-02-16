import pandas as pd
import os

def extrair_dados_excel(caminho_arquivo):
    print(f"Iniciando extração do arquivo: {caminho_arquivo}")
    
    try:
        excel = pd.ExcelFile(caminho_arquivo)
        abas = excel.sheet_names
        print(f"Abas encontradas: {abas}")
    except Exception as e:
        print(f"Erro ao abrir o arquivo Excel: {e}")
        return

    dfs = {}

    for aba in abas:
        print(f"Extraindo dados da aba: {aba}...")
        
        df_temporario = pd.read_excel(caminho_arquivo, sheet_name=aba, na_values=['#N/A', '#DIV/0!', ''])
        
        nome_csv = f"extraido_{aba}.csv"

        os.makedirs('extraidos', exist_ok=True)

        df_temporario.to_csv(os.path.join('extraidos', nome_csv), index=False, encoding='utf-8-sig')
        
        dfs[aba] = df_temporario
        print(f"Aba {aba} extraída com {len(df_temporario)} registros.")

    print("\nProcesso de extração concluído com sucesso!")
    return dfs

arquivo_origem = "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"

dados_extraidos = extrair_dados_excel(arquivo_origem)