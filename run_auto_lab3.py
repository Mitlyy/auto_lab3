from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_lab3.config import ExperimentConfig
from auto_lab3.pipeline import run_experiment



def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Auto Lab 3: hypernetwork meta-learning for classification")
    parser.add_argument("--data-dir", type=str, default="auto_lab3/OpenML/data")
    parser.add_argument("--index-path", type=str, default="auto_lab3/OpenML/data.csv")
    parser.add_argument("--out-dir", type=str, default="auto_lab3/outputs")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-datasets", type=int, default=12)
    parser.add_argument("--max-datasets", type=int, default=16)

    parser.add_argument("--target-num-classes", type=int, default=6)
    parser.add_argument("--target-num-features", type=int, default=6)
    parser.add_argument("--samples-per-class", type=int, default=96)
    parser.add_argument("--test-size", type=float, default=0.3)

    parser.add_argument("--f-train-epochs", type=int, default=30)
    parser.add_argument("--f-learning-rate", type=float, default=5e-3)
    parser.add_argument("--f-hidden-dim-1", type=int, default=96)
    parser.add_argument("--f-hidden-dim-2", type=int, default=48)

    parser.add_argument("--g-supervised-epochs", type=int, default=500)
    parser.add_argument("--g-supervised-lr", type=float, default=5e-3)
    parser.add_argument("--g-dynamic-steps", type=int, default=1600)
    parser.add_argument("--g-dynamic-lr", type=float, default=2e-3)
    parser.add_argument("--g-hidden-dim", type=int, default=64)

    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")

    args = parser.parse_args()

    return ExperimentConfig(
        data_dir=Path(args.data_dir),
        index_path=Path(args.index_path),
        out_dir=Path(args.out_dir),
        seed=args.seed,
        min_datasets=args.min_datasets,
        max_datasets=args.max_datasets,
        target_num_classes=args.target_num_classes,
        target_num_features=args.target_num_features,
        samples_per_class=args.samples_per_class,
        test_size=args.test_size,
        f_train_epochs=args.f_train_epochs,
        f_learning_rate=args.f_learning_rate,
        f_hidden_dim_1=args.f_hidden_dim_1,
        f_hidden_dim_2=args.f_hidden_dim_2,
        g_supervised_epochs=args.g_supervised_epochs,
        g_supervised_lr=args.g_supervised_lr,
        g_dynamic_steps=args.g_dynamic_steps,
        g_dynamic_lr=args.g_dynamic_lr,
        g_hidden_dim=args.g_hidden_dim,
        device=args.device,
    )



def main() -> None:
    config = parse_args()
    artifacts = run_experiment(config)

    print(f"Artifacts dir: {artifacts['out_dir']}")
    print(f"Summary report: {artifacts['report_json']}")
    print(f"Per-dataset results: {artifacts['results_csv']}")
    print(f"Stage summary: {artifacts['summary_csv']}")
    print(f"Device: {artifacts['device']}")
    print(f"Datasets used: {artifacts['n_datasets']}")


if __name__ == "__main__":
    main()
