from dataclasses import dataclass

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.layers import RandomBrightness, RandomFlip, RandomRotation, RandomZoom, Rescaling

from .config import PathsConfig, TrainingConfig


@dataclass(frozen=True)
class DatasetBundle:
    """Agrupa datasets de treino e validação."""

    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset


class CNNTrainer:
    """Responsável por ingestão, augmentation, treinamento e auditoria."""

    def __init__(self, paths: PathsConfig, config: TrainingConfig) -> None:
        self._paths: PathsConfig = paths
        self._config: TrainingConfig = config

    def load_datasets(self) -> DatasetBundle:
        """Carrega o dataset em lote com split treino/validação."""
        train_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
            self._paths.train_dir,
            validation_split=self._config.validation_split,
            subset="training",
            seed=self._config.seed,
            image_size=(self._config.image_height, self._config.image_width),
            batch_size=self._config.batch_size,
            label_mode="binary",
        )

        val_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
            self._paths.train_dir,
            validation_split=self._config.validation_split,
            subset="validation",
            seed=self._config.seed,
            image_size=(self._config.image_height, self._config.image_width),
            batch_size=self._config.batch_size,
            label_mode="binary",
        )

        autotune: int = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=autotune)
        val_ds = val_ds.cache().prefetch(buffer_size=autotune)

        return DatasetBundle(train_ds=train_ds, val_ds=val_ds)

    def build_model(self) -> tf.keras.Model:
        """Cria CNN sequencial com augmentation dinâmico."""
        augmentation: Sequential = Sequential(
            [
                RandomFlip("horizontal"),
                RandomRotation(0.1),
                RandomZoom(0.1),
                RandomBrightness(0.2),
            ],
            name="data_augmentation",
        )

        model: Sequential = Sequential(
            [
                tf.keras.Input(shape=(self._config.image_height, self._config.image_width, 3)),
                augmentation,
                Rescaling(1.0 / 255.0),
                Conv2D(32, (3, 3), activation="relu"),
                MaxPooling2D(),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPooling2D(),
                Conv2D(128, (3, 3), activation="relu"),
                MaxPooling2D(),
                Flatten(),
                Dense(128, activation="relu"),
                Dense(1, activation="sigmoid"),
            ],
            name="casting_quality_cnn",
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self._config.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(self, model: tf.keras.Model, datasets: DatasetBundle) -> tf.keras.callbacks.History:
        """Treina modelo de classificação binária."""
        history: tf.keras.callbacks.History = model.fit(
            datasets.train_ds,
            validation_data=datasets.val_ds,
            epochs=self._config.epochs,
            verbose=1,
        )
        return history

    def plot_history(self, history: tf.keras.callbacks.History) -> None:
        """Gera gráfico com curvas de loss e acurácia de treino/validação."""
        self._paths.plots_output_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.title("Loss por Época")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Val Accuracy")
        plt.title("Acurácia por Época")
        plt.xlabel("Época")
        plt.ylabel("Acurácia")
        plt.legend()

        output_file = self._paths.plots_output_dir / "training_curves.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
