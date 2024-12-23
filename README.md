# Shuffled Style Assembly Network (SSAN) for Face Anti-Spoofing

This repository contains the PyTorch implementation of the Shuffled Style Assembly Network (SSAN) for Face Anti-Spoofing, as described in the paper:

**Domain Generalization via Shuffled Style Assembly for Face Anti-Spoofing**
([Paper](https://doi.org/10.48550/arXiv.2203.05340))

## Overview 

This project provides a clean and well-structured implementation of the SSAN model from scratch, based on the original paper and the official implementation (included as a submodule). It includes data loading, model definition, training, and testing pipelines.

## Key Features

*   **Clean Implementation:** A from-scratch implementation of the SSAN model in PyTorch.
*   **Modular Design:** Code is organized into logical modules for data loading, model components, training, and evaluation.
*   **Reproducible Results:** Includes configuration files and scripts to reproduce the results.
*   **Hyperparameter Optimization:** Supports hyperparameter tuning using Optuna.
*   **Mixed Precision Training:** Utilizes mixed precision training for faster training on compatible GPUs.
*   **Comprehensive Metrics:** Tracks various metrics during training and testing, including accuracy, AUC, TPR@FPR, and HTER.
*   **Flexible Protocols:** Supports various training and testing protocols as described in the original paper.
*   **Large-Scale Benchmark:** Includes support for the large-scale benchmark proposed in the paper.
*   **Test Cases:** Includes test cases for core components.

## Directory Structure

```
├── data/                     # Contains datasets and protocols
│   ├── CATI_FAS_dataset/
│   ├── CelebA_Spoof_dataset/
│   ├── LCC_FASD_dataset/
│   ├── NUAAA_dataset/
│   ├── Zalo_AIC_dataset/
│   └── protocols/
├── docs/                     # Documentation files
│   ├── Domain Generalization via Shuffled Style Assembly for Face Anti-Spoofing.pdf
│   ├── enviroment.md
│   └── protocols.md
├── output/                   # Output directory for checkpoints, logs, and results
│   ├── hp_optimization/
│   ├── test_YYYYMMDD_HHMMSS/
│   └── train_YYYYMMDD_HHMMSS/
├── src/                      # Source code
│   ├── data/
│   ├── model/
│   ├── runner/
│   ├── tests/
│   ├── utils/
│   ├── config.py
│   └── main.py
├── SSAN/                    # Official implementation code (as a submodule)
│   ├── configs/
│   ├── datasets/
│   ├── images/
│   ├── Large_Scale_FAS_Benchmarks/
│   ├── loss/
│   ├── networks/
│   ├── optimizers/
│   ├── transformers/
│   └── utils/
├── .env
├── kaggle_run.ipynb
├── README.md
└── requirements.txt
```

## Setup

1.  **Clone the Repository:**

```bash
git clone https://github.com/n24q02m/SSAN.git
cd SSAN
```

2.  **Set up Environment:**

*   Follow the instructions in `docs/enviroment.md` to create a conda environment and install the required dependencies.
```bash
conda create -n ssan python=3.10
conda activate ssan
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
pip install -r requirements.txt
```

1.  **Prepare Datasets:**

*   Download the required datasets and place them in the `data` directory according to the structure described in `docs/protocols.md`.
    *   [CelebA-Spoof](https://www.kaggle.com/datasets/n24q02m/celeba-spoof-face-anti-spoofing-dataset) (private dataset)
    *   [NUAA](https://www.kaggle.com/datasets/n24q02m/nuaa-face-anti-spoofing-dataset) (private dataset)
    *   [Zalo AIC](https://www.kaggle.com/datasets/n24q02m/zalo-aic-face-anti-spoofing-dataset) (private dataset)
    *   [CATI-FAS](https://www.kaggle.com/datasets/n24q02m/cati-fas-face-anti-spoofing-dataset) (private dataset)
    *   [LCC-FASD](https://www.kaggle.com/datasets/n24q02m/lcc-fasd-face-anti-spoofing-dataset) (private dataset)
*   Run the preprocessing script to generate the necessary protocol files:
```bash
python -m src.data.datasets
```

## Usage

### Training

To train the SSAN model, use the following command:
```bash
python -m src.main --mode train --protocol < protocol_name > [optional arguments]
```

Replace `< protocol_name >` with one of the available protocols: `protocol_1`, `protocol_2`, `protocol_3`, `protocol_4`.

**Optional arguments:**

*   `--epochs`: Number of training epochs.
*   `--auto_hp`: Automatically optimize hyperparameters using Optuna.
*   `--hp_trials`: Number of hyperparameter optimization trials.
*   `--fraction`: Fraction of data to use (0.01-1.0).
*   `--lr`: Learning rate.
*   `--batch_size`: Batch size.
*   `--optimizer`: Optimizer type (`adam` or `sgd`).
*   `--scheduler`: Learning rate scheduler (`step` or `cosine`).
*   `--device`: Device to run on (`cuda` or `cpu`).
*   `--no_workers`: Disable data loading workers (useful for GPU).

Example:
```bash
python -m src.main --mode train --protocol protocol_1 --epochs 100 --auto_hp --hp_trials 5
```

### Testing

To test the trained model, use the following command:
```bash
python -m src.main --mode test --checkpoint < checkpoint_path > --protocol < protocol_name > [optional arguments]
```

Replace `< checkpoint_path >` with the path to the checkpoint file (`.pth`) or a folder containing checkpoints.

Example:
```bash
python -m src.main --mode test --checkpoint output/train_20241223_092631/checkpoints/best.pth --protocol protocol_1
```
