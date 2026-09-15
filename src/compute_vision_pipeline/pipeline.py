from pathlib import Path

from .config import OpenCVConfig, PathsConfig, ProjectConfig, TrainingConfig
from .exploratory import ExploratoryAnalyzer
from .training import CNNTrainer


class QualityInspectionPipeline:
    """Orquestra análise clássica e treinamento da CNN."""

    def __init__(self, config: ProjectConfig) -> None:
        self._config: ProjectConfig = config
        self._exploratory: ExploratoryAnalyzer = ExploratoryAnalyzer(config.opencv)
        self._trainer: CNNTrainer = CNNTrainer(config.paths, config.training)

    def run_exploratory_analysis(self, sample_count: int = 3) -> None:
        """Processa uma amostra para demonstrar destaque de defeitos."""
        image_paths: list[Path] = sorted(
            [*self._config.paths.train_dir.rglob("*.jpeg"), *self._config.paths.train_dir.rglob("*.jpg"), *self._config.paths.train_dir.rglob("*.png")]
        )

        selected_images: list[Path] = image_paths[:sample_count]
        if len(selected_images) == 0:
            raise FileNotFoundError(
                "Nenhuma imagem encontrada na pasta de treino. "
                f"Esperado em: {self._config.paths.train_dir} "
                "(extensões aceitas: .jpeg, .jpg, .png)."
            )

        for image_path in selected_images:
            outputs = self._exploratory.process_image(image_path)
            self._exploratory.save_results(
                outputs=outputs,
                output_dir=self._config.paths.exploratory_output_dir,
                image_stem=image_path.stem,
            )

    def run_training(self) -> None:
        """Executa ingestão, treinamento e geração de gráfico de auditoria."""
        datasets = self._trainer.load_datasets()
        model = self._trainer.build_model()
        history = self._trainer.train(model, datasets)
        self._trainer.plot_history(history)


def _choose_existing_path(candidates: list[Path]) -> Path | None:
    """Retorna o primeiro caminho existente dentre candidatos."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_dataset_root(project_root: Path) -> Path:
    """Resolve automaticamente a raiz do dataset em caminhos comuns."""
    candidates: list[Path] = [
        project_root / "data" / "casting_data",
        project_root / "data" / "casting_data" / "casting_data",
        project_root / "data" / "CastingProductImageData",
        project_root / "data" / "castings_data",
        project_root / "data" / "sample_images" / "casting_512x512",
    ]

    existing = _choose_existing_path(candidates)
    if existing is not None:
        return existing

    # Fallback padrão para manter comportamento previsível mesmo sem dataset ainda baixado.
    return project_root / "data" / "casting_data"


def build_default_config(project_root: Path) -> ProjectConfig:
    """Cria configuração padrão baseada na estrutura esperada do projeto."""
    dataset_root: Path = _resolve_dataset_root(project_root)

    train_dir: Path = dataset_root / "train"
    test_dir: Path = dataset_root / "test"
    if not train_dir.exists() and (dataset_root / "ok_front").exists() and (dataset_root / "def_front").exists():
        # Alguns recortes do dataset vêm sem subpastas train/test.
        train_dir = dataset_root
        test_dir = dataset_root

    paths = PathsConfig(
        dataset_root=dataset_root,
        train_dir=train_dir,
        test_dir=test_dir,
        output_root=project_root / "outputs",
        exploratory_output_dir=project_root / "outputs" / "exploratory",
        plots_output_dir=project_root / "outputs" / "plots",
    )

    return ProjectConfig(
        paths=paths,
        opencv=OpenCVConfig(),
        training=TrainingConfig(),
    )
