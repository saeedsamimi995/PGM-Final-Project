## PGM Final Project

Reproducible implementation of a stacked autoencoder plus Monte Carlo dropout
classifier for classifying breast cancer sub-types. The project has been
refactored into a maintainable Python package with a single entry-point script,
configuration management, and optional plotting utilities.

### Key Features

- End-to-end training pipeline with clearly separated data loading, modelling,
  and evaluation logic.
- Autoencoder for dimensionality reduction and a dropout-enabled classifier for
  calibrated probability estimates.
- Monte Carlo dropout inference to quantify predictive uncertainty.
- CLI with configurable hyper-parameters and optional plot export.
- Central JSON config (`config.json`) for dataset path and default hyper-parameters.
- Modular codebase organised inside `src/pgm_final_project`.

### Project Structure

```
PGM-Final-Project/
├── main.py
├── config.json
├── src/
│   └── pgm_final_project/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── models.py
│       ├── pipeline.py
│       └── visualization.py
├── legacy_scripts/
│   ├── pgm_auto_encoder_mlp_both_bayesian.py
│   ├── pgm_bayesian_auto_encoder_mlp.py
│   ├── pgm_project_last_version.py
│   └── PGM_project_saeed_samimi_40108724_3layerae_3layermlp.py
└── README.md
```

> The original Colab notebooks have been preserved under `legacy_scripts/` for
> reference but are no longer required for running the project.

### Getting Started

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. **Configure the dataset path**

   Edit `config.json` and point `data_path` to the CSV you want to train on.
   The same config is consumed by both the CLI and the legacy scripts.

3. **Run the training pipeline**

   ```bash
   pgm-train \
     --config config.json \
     --plots-dir outputs/plots
   ```

   Adjust arguments as needed. Use `pgm-train --help` to see the available
   options (batch size, learning rate, dropout rate, etc.). CLI arguments take
   precedence over values supplied in `config.json`.

### Output

- Classification report (logged to the console).
- Confusion matrices for train/test splits.
- Monte Carlo dropout uncertainty estimates.
- Optional plots saved into `--plots-dir`.

### Requirements

- Python 3.10+
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- TensorFlow 2.12+

See `requirements.txt` for the canonical dependency list.
