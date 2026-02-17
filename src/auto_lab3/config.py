from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path = Path("auto_lab3/OpenML/data")
    index_path: Path = Path("auto_lab3/OpenML/data.csv")
    out_dir: Path = Path("auto_lab3/outputs")

    seed: int = 42
    min_datasets: int = 12
    max_datasets: int = 16

    min_rows_raw: int = 120
    max_rows_raw: int = 4000
    max_raw_classes: int = 25

    target_num_classes: int = 6
    target_num_features: int = 6
    samples_per_class: int = 96
    min_samples_per_class: int = 48
    test_size: float = 0.3

    max_categories_per_feature: int = 15
    max_corr_features: int = 32

    f_hidden_dim_1: int = 96
    f_hidden_dim_2: int = 48
    f_train_epochs: int = 30
    f_learning_rate: float = 5e-3
    f_weight_decay: float = 1e-4

    g_hidden_dim: int = 64
    g_supervised_epochs: int = 500
    g_supervised_lr: float = 5e-3
    g_dynamic_steps: int = 1600
    g_dynamic_lr: float = 2e-3

    device: str = "auto"
