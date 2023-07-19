from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import TrainingConfig


@dataclass(slots=True)
class DatasetSplit:
    """Container for train/test splits."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder


def load_dataset(config: TrainingConfig) -> DatasetSplit:
    """Load the dataset, encode labels, and return train/test splits."""
    dataset_path = config.resolve_data_path()
    frame = pd.read_csv(dataset_path)

    target = (
        config.target_column
        if config.target_column is not None
        else frame.columns[-1]
    )

    if target not in frame.columns:
        raise ValueError(
            f"Target column `{target}` is not present in {dataset_path}"
        )

    features = frame.drop(columns=[target]).to_numpy(dtype=np.float32)
    labels_raw = frame[target].to_numpy()

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels_raw).astype(np.int32)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels,
    )

    return DatasetSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        label_encoder=label_encoder,
    )

