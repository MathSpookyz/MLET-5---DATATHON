import argparse
import logging

from sklearn.preprocessing import StandardScaler

from models.model_io import save_model
from models.preprocessing import obter_dados_e_preprocessar
from models.training import treinar_e_agrupar_alunos


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Train and export model artifacts")
    parser.add_argument(
        "--csv",
        default="dataset/PEDE_PASSOS_DATASET_FIAP.csv",
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "--output",
        default="artifacts",
        help="Directory to save model artifacts",
    )
    args = parser.parse_args()

    logger.info("Starting training pipeline")

    scaler = StandardScaler()
    df, cluster_metricas, metricas_normalizadas = obter_dados_e_preprocessar(
        args.csv, scaler
    )

    df, modelo_rf, kmeans = treinar_e_agrupar_alunos(
        df, cluster_metricas, metricas_normalizadas
    )

    save_model(scaler, kmeans, modelo_rf, cluster_metricas, args.output)
    logger.info("Training pipeline complete")


if __name__ == "__main__":
    main()
