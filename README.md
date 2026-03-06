# Pede Passos - Predicao de Risco de Alunos

Sistema de machine learning desenvolvido para o Datathon da FIAP que prediz niveis de desempenho de alunos e o risco de nao atingirem o Ponto de Virada (PV), utilizando modelos de clusterizacao e classificacao treinados com o dataset do Pede Passos.

## Arquitetura

```
models/          Logica de ML (preprocessamento, treino, predicao, ranking, I/O de modelo)
api/             Aplicacao FastAPI (factory, rotas, schemas Pydantic)
data/            Scripts de extracao de dados (Excel para CSV)
dataset/         Arquivos CSV do dataset
tests/           Testes unitarios (pytest)
artifacts/       Artefatos do modelo salvo (gerados pelo train.py)
train.py         CLI para treinar e exportar artefatos do modelo
main.py          Ponto de entrada da aplicacao (uvicorn)
```

### Modelos

- **KMeans** (k=4) agrupa alunos em niveis de desempenho com base em 8 metricas normalizadas (INDE, IAA, IEG, IPS, IDA, IPP, IPV, IAN).
- **RandomForestClassifier** (150 estimadores) prediz a probabilidade de um aluno atingir o Ponto de Virada.
- Os artefatos do modelo sao persistidos com `joblib` e carregados na inicializacao da API, evitando retreinamento a cada reinicio.

## Configuracao

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

## Treinamento do Modelo

```bash
python train.py
```

Este comando le o arquivo `dataset/PEDE_PASSOS_DATASET_FIAP.csv`, treina ambos os modelos e salva os artefatos em `artifacts/`.

Opcoes:

```
--csv <caminho>    Caminho para o CSV de entrada (padrao: dataset/PEDE_PASSOS_DATASET_FIAP.csv)
--output <dir>     Diretorio para salvar os artefatos (padrao: artifacts)
```

## Executando a API

```bash
python main.py
```

Ou diretamente com uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Se `artifacts/` contiver modelos pre-treinados, a API os carrega diretamente. Caso contrario, treina do zero na primeira inicializacao e salva os artefatos para execucoes seguintes.

## Endpoints da API

### GET /health

Retorna o status do servico.

```json
{"status": "ok"}
```

### POST /prever

Prediz o grupo de cluster e a probabilidade de PV para um aluno.

Corpo da requisicao:

```json
{
  "INDE_FINAL": 7.5,
  "IAA_FINAL": 8.0,
  "IEG_FINAL": 6.5,
  "IPS_FINAL": 7.0,
  "IDA_FINAL": 8.5,
  "IPP_FINAL": 7.0,
  "IPV_FINAL": 6.0,
  "IAN_FINAL": 5.5
}
```

Resposta:

```json
{
  "Grupo": 2,
  "Nivel": "Nivel 3",
  "Probabilidade_PV": 0.7823
}
```

## Executando Testes

```bash
pytest tests/ -v
```

Todos os testes utilizam dados sinteticos e nao dependem do CSV real.

## Docker

Build e execucao:

```bash
docker build -t pede-passos .
docker run -p 8000:8000 pede-passos
```

O build do Docker treina o modelo durante a construcao da imagem, de modo que o container inicia com os artefatos pre-treinados.

## Extracao de Dados

Para regenerar o CSV unificado a partir do arquivo Excel original:

```bash
python -m data.extract_excel_data
```

Para extrair abas individuais em CSVs separados na pasta `extraidos/`:

```bash
python -m data.extract_data
```
