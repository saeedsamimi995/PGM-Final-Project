from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(slots=True)
class TrainingConfig:
    """Configuration values used across the training pipeline."""

    data_path: Path
    target_column: Optional[str] = None
    test_size: float = 0.3
    random_state: int = 42
    epochs_autoencoder: int = 75
    epochs_classifier: int = 100
    batch_size: int = 32
    latent_dim: int = 64
    dropout_rate: float = 0.4
    monte_carlo_samples: int = 100
    patience: int = 10
    learning_rate: float = 1e-3

    project_name: str = field(default="PGM Final Project", init=False)

    def resolve_data_path(self) -> Path:
        """Return the fully resolved dataset path, ensuring it exists."""
        resolved = self.data_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Cannot find dataset at {resolved}. "
                "Update the path via CLI option `--data-path` or config."
            )
        return resolved


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"


def load_config_from_file(path: Optional[Path] = None) -> TrainingConfig:
    """Load a :class:`TrainingConfig` from a JSON configuration file."""
    config_path = path or DEFAULT_CONFIG_PATH
    config_path = config_path.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data: Dict[str, Any] = json.load(file)

    if "data_path" not in data:
        raise KeyError("Configuration file must define `data_path`.")

    data["data_path"] = Path(str(data["data_path"]))
    field_names = {
        name for name, field in TrainingConfig.__dataclass_fields__.items() if field.init
    }
    return TrainingConfig(**{k: v for k, v in data.items() if k in field_names})

