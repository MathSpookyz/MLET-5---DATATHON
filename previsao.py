import pandas as pd
import joblib
import os

# --- CONFIGURAÇÕES ---
NOME_MODELO = 'modelo_previsao_inde.pkl'

# Aqui você coloca a planilha dos alunos "sem nota" que precisa prever.
# Por enquanto, vou deixar a mesma base só para você ver funcionando e validar o resultado.
ARQUIVO_NOVOS_DADOS = 'BASE_DE_DADOS_PEDE_2024-DATATHON.xlsx' 
ARQUIVO_SAIDA = 'previsoes_finais.xlsx'

def prever_novos_alunos():
    print(f"1. Carregando o modelo treinado ({NOME_MODELO})...")
    if not os.path.exists(NOME_MODELO):
        print("Erro: Arquivo do modelo não encontrado. Rode o treinamento primeiro.")
        return
        
    modelo_carregado = joblib.load(NOME_MODELO)
    
    print(f"2. Carregando os novos dados ({ARQUIVO_NOVOS_DADOS})...")
    if not os.path.exists(ARQUIVO_NOVOS_DADOS):
        print("Erro: Arquivo de dados não encontrado.")
        return
        
    if ARQUIVO_NOVOS_DADOS.endswith('.xlsx'):
        df_novos = pd.read_excel(ARQUIVO_NOVOS_DADOS)
    else:
        df_novos = pd.read_csv(ARQUIVO_NOVOS_DADOS)
        
    print(f"Dados carregados! Dimensões: {df_novos.shape}")
    
    # 3. O TRUQUE DO DATATHON: Separar a identificação!
    # Guardamos a identificação para saber de quem é a nota depois.
    if 'RA' in df_novos.columns and 'Nome' in df_novos.columns:
        identificacao = df_novos[['RA', 'Nome']].copy()
    else:
        print("Aviso: Colunas 'RA' ou 'Nome' não encontradas. Usando índice da linha.")
        identificacao = pd.DataFrame(index=df_novos.index)
        
    # 4. Preparando os dados (Removendo o que o modelo não deve ver, igual no treino)
    colunas_para_remover = [
        'RA', 'Nome', 'Turma', 
        'Pedra 22', 'Pedra 21', 'Pedra 20', 
        'INDE 22' # Removemos o INDE caso ele venha na planilha nova (para não trapacear)
    ]
    
    drop_cols = [c for c in colunas_para_remover if c in df_novos.columns]
    X_novos = df_novos.drop(columns=drop_cols)
    
    # 5. Fazendo a mágica acontecer
    print("3. Alimentando o modelo com os dados e gerando previsões...")
    # Como salvamos a Pipeline inteira, ela faz o pré-processamento (imputação, escala) 
    # automaticamente antes de fazer a previsão!
    previsoes = modelo_carregado.predict(X_novos)
    
    # 6. Juntando os nomes com as previsões
    resultados = identificacao
    resultados['Previsao_INDE'] = previsoes
    
    # Arredondando para 3 casas decimais (padrão comum em notas)
    resultados['Previsao_INDE'] = resultados['Previsao_INDE'].round(3)
    
    # 7. Salvando o resultado final
    print(f"4. Salvando os resultados na planilha '{ARQUIVO_SAIDA}'...")
    resultados.to_excel(ARQUIVO_SAIDA, index=False)
    
    print("\n" + "="*45)
    print("PREVISÃO CONCLUÍDA COM SUCESSO!")
    print("="*45)
    print("Aqui estão os 5 primeiros alunos e suas notas previstas:")
    print(resultados.head())

if __name__ == "__main__":
    prever_novos_alunos()