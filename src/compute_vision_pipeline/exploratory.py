from pathlib import Path

import cv2
import numpy as np

from .config import OpenCVConfig


class ExploratoryAnalyzer:
    """Executa pipeline clássico de OpenCV para evidenciar defeitos."""

    def __init__(self, config: OpenCVConfig) -> None:
        self._config: OpenCVConfig = config

    def process_image(self, image_path: Path) -> dict[str, np.ndarray]:
        """Aplica todas as etapas clássicas em uma imagem."""
        image_bgr: np.ndarray | None = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Não foi possível ler a imagem: {image_path}")

        # Conversão para tons de cinza para reduzir dimensionalidade e focar em estrutura.
        gray: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Suavização gaussiana para reduzir ruído local da superfície metálica.
        blurred: np.ndarray = cv2.GaussianBlur(
            gray,
            self._config.gaussian_kernel_size,
            self._config.gaussian_sigma,
        )

        # Limiarização binária para separar regiões claras/escuras.
        _, thresholded = cv2.threshold(
            blurred,
            self._config.threshold_value,
            self._config.threshold_max_value,
            cv2.THRESH_BINARY,
        )

        # Canny destaca contornos abruptos, úteis para ranhuras e trincas.
        edges: np.ndarray = cv2.Canny(
            blurred,
            self._config.canny_threshold1,
            self._config.canny_threshold2,
        )

        # Morfologia ajuda a consolidar regiões de defeito detectadas nas bordas.
        kernel: np.ndarray = np.ones(self._config.morphology_kernel_size, np.uint8)
        dilated: np.ndarray = cv2.dilate(edges, kernel, iterations=self._config.morphology_iterations)
        eroded: np.ndarray = cv2.erode(dilated, kernel, iterations=self._config.morphology_iterations)

        return {
            "original": image_bgr,
            "gray": gray,
            "blurred": blurred,
            "thresholded": thresholded,
            "edges": edges,
            "dilated": dilated,
            "eroded": eroded,
        }

    def save_results(self, outputs: dict[str, np.ndarray], output_dir: Path, image_stem: str) -> None:
        """Salva os resultados intermediários da análise exploratória."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for step_name, image in outputs.items():
            target_path: Path = output_dir / f"{image_stem}_{step_name}.png"
            cv2.imwrite(str(target_path), image)
