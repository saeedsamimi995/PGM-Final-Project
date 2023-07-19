from __future__ import annotations

import argparse
import logging
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt

from .config import DEFAULT_CONFIG_PATH, TrainingConfig, load_config_from_file
from .pipeline import run_training
from .visualization import plot_confusion, plot_training_curves


def _positive_float(value: str) -> float:
    result = float(value)
    if not 0 < result < 1:
        raise argparse.ArgumentTypeError("Value must be between 0 and 1.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the PGM final project training pipeline.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON configuration file (defaults to repository config).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to the CSV file containing features and labels.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Name of the target column (defaults to last column).",
    )
    parser.add_argument(
        "--test-size",
        type=_positive_float,
        default=None,
        help="Proportion of the dataset used for the test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Random seed used by numpy, sklearn, and TensorFlow.",
    )
    parser.add_argument(
        "--epochs-autoencoder",
        type=int,
        default=None,
        help="Maximum number of epochs for the autoencoder.",
    )
    parser.add_argument(
        "--epochs-classifier",
        type=int,
        default=None,
        help="Maximum number of epochs for the classifier.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size used across training steps.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=None,
        help="Dimensionality of the autoencoder latent space.",
    )
    parser.add_argument(
        "--dropout-rate",
        type=_positive_float,
        default=None,
        help="Dropout rate applied within the classifier.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate used by Adam optimizers.",
    )
    parser.add_argument(
        "--monte-carlo-samples",
        type=int,
        default=None,
        help="Number of stochastic forward passes for uncertainty.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Patience for the early stopping callback.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory to store generated plots (if omitted, plots are not saved).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Verbosity level for logging output.",
    )
    return parser


def _save_plots(
    plots_dir: Path,
    class_names: Iterable[str],
    train_confusion: np.ndarray,
    test_confusion: np.ndarray,
    histories,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.close("all")
    training_fig = plot_training_curves(histories)
    training_fig.savefig(plots_dir / "training_curves.png", dpi=150)
    plt.close(training_fig)

    train_fig = plot_confusion(train_confusion, class_names, "Train Confusion Matrix")
    train_fig.savefig(plots_dir / "train_confusion.png", dpi=150)
    plt.close(train_fig)

    test_fig = plot_confusion(test_confusion, class_names, "Test Confusion Matrix")
    test_fig.savefig(plots_dir / "test_confusion.png", dpi=150)
    plt.close(test_fig)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    base_config = None
    config_path = args.config or DEFAULT_CONFIG_PATH
    if config_path:
        try:
            base_config = load_config_from_file(config_path)
        except FileNotFoundError:
            logging.warning("Configuration file not found at %s", config_path)
        except KeyError as error:
            logging.warning("Invalid configuration file (%s): %s", config_path, error)

    config_kwargs = {}
    base_values = (
        {
            field.name: getattr(base_config, field.name)
            for field in fields(TrainingConfig)
            if base_config is not None and field.init
        }
        if base_config
        else {}
    )

    for field in fields(TrainingConfig):
        if not field.init:
            continue
        arg_value = getattr(args, field.name, None)
        if arg_value is not None:
            config_kwargs[field.name] = arg_value
        elif field.name in base_values:
            config_kwargs[field.name] = base_values[field.name]
        elif field.default is not MISSING:
            config_kwargs[field.name] = field.default
        elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
            config_kwargs[field.name] = field.default_factory()  # type: ignore[attr-defined]

    if "data_path" not in config_kwargs or config_kwargs["data_path"] is None:
        raise ValueError(
            "Dataset path not provided. Use `--data-path` or specify `data_path` in config."
        )

    if not isinstance(config_kwargs["data_path"], Path):
        config_kwargs["data_path"] = Path(config_kwargs["data_path"])

    config = TrainingConfig(**config_kwargs)

    training_result = run_training(config)

    logging.info("Training complete.")
    logging.info("Classification report:%s%s", "\n", training_result.report)
    logging.info(
        "Mean predictive uncertainty (test): %.4f ± %.4f",
        training_result.test_uncertainty.mean(),
        training_result.test_uncertainty.std(),
    )

    if args.plots_dir:
        _save_plots(
            plots_dir=args.plots_dir,
            class_names=training_result.class_names,
            train_confusion=training_result.train_confusion,
            test_confusion=training_result.test_confusion,
            histories=training_result.histories,
        )


if __name__ == "__main__":
    main()

