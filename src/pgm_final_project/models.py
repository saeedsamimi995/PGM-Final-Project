from __future__ import annotations

from typing import Tuple

import tensorflow as tf


def build_autoencoder(
    input_dim: int,
    latent_dim: int,
    learning_rate: float,
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Return a compiled autoencoder and its encoder sub-model."""
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = tf.keras.layers.Dense(256, activation="relu", name="enc_dense_1")(inputs)
    x = tf.keras.layers.Dense(128, activation="relu", name="enc_dense_2")(x)
    latent = tf.keras.layers.Dense(
        latent_dim, activation="relu", name="latent_space"
    )(x)

    x = tf.keras.layers.Dense(128, activation="relu", name="dec_dense_1")(latent)
    x = tf.keras.layers.Dense(256, activation="relu", name="dec_dense_2")(x)
    outputs = tf.keras.layers.Dense(
        input_dim, activation="linear", name="reconstruction"
    )(x)

    autoencoder = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="autoencoder",
    )
    encoder = tf.keras.Model(
        inputs=inputs,
        outputs=latent,
        name="encoder",
    )

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )

    return autoencoder, encoder


def build_classifier(
    latent_dim: int,
    num_classes: int,
    dropout_rate: float,
    learning_rate: float,
) -> tf.keras.Model:
    """Return a compiled classifier operating on latent representations."""
    inputs = tf.keras.Input(shape=(latent_dim,), name="encoded_features")
    x = tf.keras.layers.Dense(128, activation="relu", name="clf_dense_1")(inputs)
    x = tf.keras.layers.Dropout(dropout_rate, name="clf_dropout_1")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="clf_dense_2")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="clf_dropout_2")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="class_probabilities"
    )(x)

    classifier = tf.keras.Model(inputs=inputs, outputs=outputs, name="classifier")
    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )
    return classifier

