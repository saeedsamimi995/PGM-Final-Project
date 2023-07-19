from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from .config import TrainingConfig
from .data import DatasetSplit, load_dataset
from .models import build_autoencoder, build_classifier

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class HistoryBundle:
    """Holds the Keras history objects for analysis and plotting."""

    autoencoder: Optional[tf.keras.callbacks.History]
    classifier: tf.keras.callbacks.History


@dataclass(slots=True)
class TrainingResult:
    """Artifacts produced by the training pipeline."""

    config: TrainingConfig
    histories: HistoryBundle
    models: "ModelBundle"
    class_names: tuple[str, ...]
    train_confusion: np.ndarray
    test_confusion: np.ndarray
    report: str
    train_uncertainty: np.ndarray
    test_uncertainty: np.ndarray


def _configure_environment(random_state: int) -> None:
    """Ensure reproducible behaviour and quieter TensorFlow logging."""
    tf.random.set_seed(random_state)
    np.random.seed(random_state)
    tf.get_logger().setLevel("ERROR")


def _monte_carlo_dropout_predictions(
    model: tf.keras.Model,
    features: np.ndarray,
    samples: int,
) -> np.ndarray:
    """Return class probability estimates using Monte Carlo dropout."""
    tensor_features = tf.convert_to_tensor(features, dtype=tf.float32)
    mc_predictions = []

    for _ in range(samples):
        logits = model(tensor_features, training=True)
        mc_predictions.append(tf.convert_to_tensor(logits))

    stacked = tf.stack(mc_predictions, axis=0)
    return tf.math.reduce_std(stacked, axis=0).numpy()


def _prediction_mean(
    model: tf.keras.Model, features: np.ndarray, batch_size: int
) -> np.ndarray:
    """Return the deterministic prediction mean."""
    return model.predict(
        features,
        batch_size=batch_size,
        verbose=0,
    )


def _train_autoencoder(
    dataset: DatasetSplit,
    config: TrainingConfig,
) -> tuple[tf.keras.Model, tf.keras.Model, Optional[tf.keras.callbacks.History]]:
    """Train the autoencoder if requested and return encoder representations."""
    autoencoder, encoder = build_autoencoder(
        input_dim=dataset.x_train.shape[1],
        latent_dim=config.latent_dim,
        learning_rate=config.learning_rate,
    )

    LOGGER.info("Training autoencoder for %s epochs", config.epochs_autoencoder)
    history = autoencoder.fit(
        dataset.x_train,
        dataset.x_train,
        epochs=config.epochs_autoencoder,
        batch_size=config.batch_size,
        validation_data=(dataset.x_test, dataset.x_test),
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.patience,
                restore_best_weights=True,
            )
        ],
    )

    return autoencoder, encoder, history


def _train_classifier(
    encoded_train: np.ndarray,
    encoded_test: np.ndarray,
    dataset: DatasetSplit,
    config: TrainingConfig,
    num_classes: int,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train the classifier on top of the latent space."""
    classifier = build_classifier(
        latent_dim=config.latent_dim,
        num_classes=num_classes,
        dropout_rate=config.dropout_rate,
        learning_rate=config.learning_rate,
    )

    LOGGER.info("Training classifier for %s epochs", config.epochs_classifier)
    history = classifier.fit(
        encoded_train,
        dataset.y_train,
        epochs=config.epochs_classifier,
        batch_size=config.batch_size,
        validation_data=(encoded_test, dataset.y_test),
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.patience,
                restore_best_weights=True,
            )
        ],
    )

    return classifier, history


@dataclass(slots=True)
class ModelBundle:
    """Convenience wrapper around trained models."""

    autoencoder: tf.keras.Model
    encoder: tf.keras.Model
    classifier: tf.keras.Model


def run_training(config: TrainingConfig) -> TrainingResult:
    """Execute the end-to-end training pipeline."""
    _configure_environment(config.random_state)
    dataset = load_dataset(config)

    autoencoder, encoder, auto_history = _train_autoencoder(dataset, config)

    encoded_train = encoder.predict(
        dataset.x_train, batch_size=config.batch_size, verbose=0
    )
    encoded_test = encoder.predict(
        dataset.x_test, batch_size=config.batch_size, verbose=0
    )

    num_classes = len(dataset.label_encoder.classes_)

    classifier, clf_history = _train_classifier(
        encoded_train,
        encoded_test,
        dataset,
        config,
        num_classes,
    )

    train_probs = _prediction_mean(classifier, encoded_train, config.batch_size)
    test_probs = _prediction_mean(classifier, encoded_test, config.batch_size)

    train_uncertainty = _monte_carlo_dropout_predictions(
        classifier,
        encoded_train,
        config.monte_carlo_samples,
    )
    test_uncertainty = _monte_carlo_dropout_predictions(
        classifier,
        encoded_test,
        config.monte_carlo_samples,
    )

    train_predicted = np.argmax(train_probs, axis=1)
    test_predicted = np.argmax(test_probs, axis=1)

    train_confusion = confusion_matrix(dataset.y_train, train_predicted)
    test_confusion = confusion_matrix(dataset.y_test, test_predicted)

    report = classification_report(
        dataset.y_test,
        test_predicted,
        target_names=dataset.label_encoder.classes_.astype(str),
    )

    histories = HistoryBundle(
        autoencoder=auto_history,
        classifier=clf_history,
    )
    models = ModelBundle(
        autoencoder=autoencoder,
        encoder=encoder,
        classifier=classifier,
    )

    return TrainingResult(
        config=config,
        histories=histories,
        models=models,
        class_names=tuple(dataset.label_encoder.classes_.astype(str)),
        train_confusion=train_confusion,
        test_confusion=test_confusion,
        report=report,
        train_uncertainty=train_uncertainty,
        test_uncertainty=test_uncertainty,
    )

