from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    """Centraliza os caminhos usados no projeto."""

    dataset_root: Path
    train_dir: Path
    test_dir: Path
    output_root: Path
    exploratory_output_dir: Path
    plots_output_dir: Path


@dataclass(frozen=True)
class OpenCVConfig:
    """Hiperparâmetros da análise exploratória clássica."""

    gaussian_kernel_size: tuple[int, int] = (5, 5)
    gaussian_sigma: float = 0.0
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    threshold_value: int = 127
    threshold_max_value: int = 255
    morphology_kernel_size: tuple[int, int] = (3, 3)
    morphology_iterations: int = 1


@dataclass(frozen=True)
class TrainingConfig:
    """Configurações do treinamento da CNN."""

    image_height: int = 300
    image_width: int = 300
    batch_size: int = 32
    validation_split: float = 0.2
    seed: int = 42
    epochs: int = 10
    learning_rate: float = 1e-3


@dataclass(frozen=True)
class ProjectConfig:
    """Configuração principal que agrega todo o pipeline."""

    paths: PathsConfig
    opencv: OpenCVConfig
    training: TrainingConfig
