from pathlib import Path

from src.compute_vision_pipeline.pipeline import QualityInspectionPipeline, build_default_config


def main() -> None:
    """Ponto de entrada do projeto."""
    project_root: Path = Path(__file__).resolve().parent
    config = build_default_config(project_root)

    pipeline = QualityInspectionPipeline(config)

    # Sprint 2 e 3: análise exploratória com OpenCV.
    pipeline.run_exploratory_analysis(sample_count=3)

    # Sprint 4, 5 e 6: ingestão, treinamento e auditoria gráfica.
    pipeline.run_training()


if __name__ == "__main__":
    main()
