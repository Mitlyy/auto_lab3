from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .data_io import list_dataset_paths, load_openml_index, load_raw_dataset
from .models import FParamSpec
from .preprocessing import META_FEATURE_NAMES, ProcessedTask, build_task_from_raw
from .trainers import (
    TaskTensors,
    evaluate_hypernet_zero_shot,
    get_device,
    set_seed,
    train_classifier_on_task,
    train_hypernet_dynamic,
    train_hypernet_supervised,
)
from .visualization import (
    plot_average_learning_curves,
    plot_dynamic_curve,
    plot_final_comparison,
    plot_preprocessing_hist,
)



def _ensure_dirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "out": out_dir,
        "tables": out_dir / "tables",
        "curves": out_dir / "curves",
        "plots": out_dir / "plots",
        "models_root": out_dir / "models",
        "models_random": out_dir / "models" / "f_random",
        "models_g_sup": out_dir / "models" / "f_g_supervised_init",
        "models_g_dyn": out_dir / "models" / "f_g_dynamic_init",
        "models_hyper": out_dir / "models" / "hypernets",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths



def _average_curves(curves: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(curves, dtype=np.float32)
    return matrix.mean(axis=0)



def _save_curve(curve: np.ndarray, path: Path) -> None:
    frame = pd.DataFrame({"step": np.arange(len(curve)), "balanced_accuracy": curve})
    frame.to_csv(path, index=False)



def _results_to_frame(stage: str, results: list) -> pd.DataFrame:
    rows = []
    for item in results:
        rows.append(
            {
                "stage": stage,
                "dataset_id": item.dataset_id,
                "dataset_name": item.dataset_name,
                "initial_balanced_accuracy": float(item.curve[0]),
                "final_balanced_accuracy": float(item.final_balanced_accuracy),
            }
        )
    return pd.DataFrame(rows)



def _write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")



def _select_tasks(config: ExperimentConfig, index_map: dict[int, dict[str, str]]) -> tuple[list[ProcessedTask], pd.DataFrame]:
    tasks: list[ProcessedTask] = []
    skipped: list[dict[str, str | int]] = []

    for path in list_dataset_paths(config.data_dir):
        if len(tasks) >= config.max_datasets:
            break

        dataset_id = int(path.stem) if path.stem.isdigit() else -1
        try:
            raw = load_raw_dataset(path=path, index_map=index_map)
            task = build_task_from_raw(raw=raw, config=config)
            tasks.append(task)
        except Exception as exc:  # noqa: BLE001
            skipped.append(
                {
                    "dataset_id": dataset_id,
                    "file": path.name,
                    "reason": str(exc),
                }
            )

    if len(tasks) < config.min_datasets:
        raise RuntimeError(f"only {len(tasks)} datasets were prepared, expected at least {config.min_datasets}")

    skipped_frame = pd.DataFrame(skipped)
    return tasks, skipped_frame



def _tasks_to_tensors(tasks: list[ProcessedTask], device: torch.device) -> tuple[list[TaskTensors], np.ndarray, np.ndarray]:
    meta_matrix = np.stack([task.meta_features for task in tasks], axis=0)
    meta_mean = meta_matrix.mean(axis=0)
    meta_std = meta_matrix.std(axis=0) + 1e-6
    meta_norm = (meta_matrix - meta_mean) / meta_std

    tensors: list[TaskTensors] = []
    for idx, task in enumerate(tasks):
        tensors.append(
            TaskTensors(
                dataset_id=task.dataset_id,
                dataset_name=task.dataset_name,
                X_train=torch.tensor(task.X_train, dtype=torch.float32, device=device),
                y_train=torch.tensor(task.y_train, dtype=torch.long, device=device),
                X_test=torch.tensor(task.X_test, dtype=torch.float32, device=device),
                y_test=torch.tensor(task.y_test, dtype=torch.long, device=device),
                meta=torch.tensor(meta_norm[idx], dtype=torch.float32, device=device),
            )
        )

    return tensors, meta_mean, meta_std



def _stage_train_f(
    stage_name: str,
    tasks: list[TaskTensors],
    spec: FParamSpec,
    config: ExperimentConfig,
    output_dir: Path,
    init_thetas: list[torch.Tensor] | None = None,
) -> tuple[list, np.ndarray]:
    results = []
    curves = []

    for idx, task in enumerate(tasks):
        init_theta = None if init_thetas is None else init_thetas[idx]
        result = train_classifier_on_task(
            task=task,
            spec=spec,
            epochs=config.f_train_epochs,
            lr=config.f_learning_rate,
            weight_decay=config.f_weight_decay,
            seed=config.seed + task.dataset_id + idx,
            init_theta=init_theta,
        )
        torch.save(
            {
                "dataset_id": task.dataset_id,
                "dataset_name": task.dataset_name,
                "theta": result.theta,
                "stage": stage_name,
            },
            output_dir / f"{task.dataset_id}.pt",
        )
        results.append(result)
        curves.append(result.curve)

    avg_curve = _average_curves(curves)
    return results, avg_curve



def _write_readme(
    readme_path: Path,
    config: ExperimentConfig,
    report: dict,
    stage_summary: pd.DataFrame,
    device_name: str,
    artifacts_prefix: str,
) -> None:
    best_random = stage_summary.loc[stage_summary["stage"] == "step4_random", "mean_final_balanced_accuracy"].iloc[0]
    best_sup = stage_summary.loc[
        stage_summary["stage"] == "step6_g_supervised_init", "mean_final_balanced_accuracy"
    ].iloc[0]
    best_dyn = stage_summary.loc[
        stage_summary["stage"] == "step8_g_dynamic_init", "mean_final_balanced_accuracy"
    ].iloc[0]

    text = f"""# Auto Lab 3: Hypernetwork for Fast Classification Initialization

## Кратко
- Выполнены пункты 1-9 на `{report['n_datasets']}` датасетах OpenML.
- После предобработки каждый датасет имеет `{config.target_num_classes}` класса и `{config.target_num_features}` признака.
- Использована метрика: `balanced_accuracy`.
- Обучение запущено на устройстве: `{device_name}`.

## Запуск
```bash
source .venv/bin/activate
python auto_lab3/run_auto_lab3.py
```

## Структура
```text
auto_lab3/
  run_auto_lab3.py
  src/auto_lab3/
    config.py
    data_io.py
    preprocessing.py
    models.py
    trainers.py
    visualization.py
    pipeline.py
  outputs/
    tables/
    curves/
    models/
    plots/
    report.json
```

## По пунктам задания
| Пункт | Что сделано | Файлы |
|---|---|---|
| 1 | Отбор и предобработка OpenML датасетов: баланс классов, сэмплирование, проекция к `K=C` | `{artifacts_prefix}/tables/processed_datasets.csv`, `{artifacts_prefix}/plots/preprocessing_hist.png` |
| 2 | Стратифицированный train/test split и выбор `balanced_accuracy` | `src/auto_lab3/preprocessing.py` |
| 3 | Усложненная сеть `f`: MLP (`{config.target_num_features} -> {config.f_hidden_dim_1} -> {config.f_hidden_dim_2} -> {config.target_num_classes}`), loss `CrossEntropy`, optimizer `AdamW` | `src/auto_lab3/models.py` |
| 4 | Обучение `f` со случайной инициализацией на каждом датасете, средняя кривая, сохранение весов | `{artifacts_prefix}/models/f_random`, `{artifacts_prefix}/curves/step4_random_avg.csv` |
| 5 | Обучение гиперсети `g` (one-to-many) по мета-признакам -> параметры `f` | `{artifacts_prefix}/models/hypernets/g_supervised.pt`, `{artifacts_prefix}/tables/g_supervised_history.csv` |
| 6 | Повтор шага 4 с инициализацией `f` от `g` | `{artifacts_prefix}/models/f_g_supervised_init`, `{artifacts_prefix}/curves/step6_g_supervised_init_avg.csv` |
| 7 | Динамическое дообучение `g` через градиент риска через `f` и `g` | `{artifacts_prefix}/models/hypernets/g_dynamic.pt`, `{artifacts_prefix}/tables/g_dynamic_history.csv`, `{artifacts_prefix}/plots/dynamic_curve.png` |
| 8 | Повтор шага 6 с динамически обученной `g` | `{artifacts_prefix}/models/f_g_dynamic_init`, `{artifacts_prefix}/curves/step8_g_dynamic_init_avg.csv` |
| 9 | Сравнение средних кривых обучения | `{artifacts_prefix}/plots/learning_curves_compare.png`, `{artifacts_prefix}/tables/stage_summary.csv` |

## Основные результаты
| Этап | Mean final balanced_accuracy | Std |
|---|---:|---:|
| Step 4: random init | {best_random:.4f} | {stage_summary.loc[stage_summary['stage']=='step4_random', 'std_final_balanced_accuracy'].iloc[0]:.4f} |
| Step 6: init from supervised g | {best_sup:.4f} | {stage_summary.loc[stage_summary['stage']=='step6_g_supervised_init', 'std_final_balanced_accuracy'].iloc[0]:.4f} |
| Step 8: init from dynamic g | {best_dyn:.4f} | {stage_summary.loc[stage_summary['stage']=='step8_g_dynamic_init', 'std_final_balanced_accuracy'].iloc[0]:.4f} |

## Визуализации
### Шаг 1: датасеты до/после предобработки
![Preprocessing]({artifacts_prefix}/plots/preprocessing_hist.png)

### Шаг 7: динамическое обучение гиперсети
![Dynamic curve]({artifacts_prefix}/plots/dynamic_curve.png)

### Шаг 9: сравнение кривых обучения
![Curve compare]({artifacts_prefix}/plots/learning_curves_compare.png)

### Сравнение финального качества
![Final compare]({artifacts_prefix}/plots/final_comparison.png)

## Артефакты
- Сводка: `{artifacts_prefix}/report.json`
- Подробная таблица датасетов: `{artifacts_prefix}/tables/processed_datasets.csv`
- Метрики по датасетам: `{artifacts_prefix}/tables/all_stage_results.csv`
- Кривые: `{artifacts_prefix}/curves/*.csv`
- Веса моделей `f` и `g`: `{artifacts_prefix}/models/`
"""
    readme_path.write_text(text, encoding="utf-8")



def run_experiment(config: ExperimentConfig) -> dict[str, Path | str | int | float]:
    set_seed(config.seed)
    paths = _ensure_dirs(config.out_dir)

    device = get_device(config.device)
    device_name = "cuda" if device.type == "cuda" else "cpu"

    index_map = load_openml_index(config.index_path)
    tasks, skipped = _select_tasks(config=config, index_map=index_map)

    processed_df = pd.DataFrame([task.summary for task in tasks]).sort_values("dataset_id").reset_index(drop=True)
    processed_df.to_csv(paths["tables"] / "processed_datasets.csv", index=False)
    if not skipped.empty:
        skipped.to_csv(paths["tables"] / "skipped_datasets.csv", index=False)

    plot_preprocessing_hist(processed_df, paths["plots"] / "preprocessing_hist.png")

    tensor_tasks, meta_mean, meta_std = _tasks_to_tensors(tasks=tasks, device=device)
    spec = FParamSpec(
        input_dim=config.target_num_features,
        hidden_dim_1=config.f_hidden_dim_1,
        hidden_dim_2=config.f_hidden_dim_2,
        output_dim=config.target_num_classes,
    )

    step4_results, step4_curve = _stage_train_f(
        stage_name="step4_random",
        tasks=tensor_tasks,
        spec=spec,
        config=config,
        output_dir=paths["models_random"],
        init_thetas=None,
    )

    step4_frame = _results_to_frame("step4_random", step4_results)
    step4_frame.to_csv(paths["tables"] / "step4_random_results.csv", index=False)
    _save_curve(step4_curve, paths["curves"] / "step4_random_avg.csv")

    target_thetas = torch.stack([item.theta for item in step4_results], dim=0).to(device)
    metas = torch.stack([task.meta for task in tensor_tasks], dim=0)

    g_supervised, g_supervised_hist = train_hypernet_supervised(
        metas=metas,
        target_thetas=target_thetas,
        meta_dim=len(META_FEATURE_NAMES),
        param_dim=spec.param_dim,
        hidden_dim=config.g_hidden_dim,
        epochs=config.g_supervised_epochs,
        lr=config.g_supervised_lr,
        device=device,
    )

    torch.save(
        {
            "state_dict": g_supervised.state_dict(),
            "meta_feature_names": META_FEATURE_NAMES,
            "meta_mean": meta_mean,
            "meta_std": meta_std,
        },
        paths["models_hyper"] / "g_supervised.pt",
    )

    pd.DataFrame(g_supervised_hist).to_csv(paths["tables"] / "g_supervised_history.csv", index=False)
    zero_shot_sup = evaluate_hypernet_zero_shot(g_supervised, tensor_tasks, spec=spec)

    init_thetas_sup: list[torch.Tensor] = []
    g_supervised.eval()
    with torch.no_grad():
        for task in tensor_tasks:
            theta = g_supervised(task.meta.unsqueeze(0)).squeeze(0).detach().cpu()
            init_thetas_sup.append(theta)

    step6_results, step6_curve = _stage_train_f(
        stage_name="step6_g_supervised_init",
        tasks=tensor_tasks,
        spec=spec,
        config=config,
        output_dir=paths["models_g_sup"],
        init_thetas=init_thetas_sup,
    )

    step6_frame = _results_to_frame("step6_g_supervised_init", step6_results)
    step6_frame.to_csv(paths["tables"] / "step6_g_supervised_init_results.csv", index=False)
    _save_curve(step6_curve, paths["curves"] / "step6_g_supervised_init_avg.csv")

    g_dynamic, g_dynamic_hist = train_hypernet_dynamic(
        hyper_start=g_supervised,
        tasks=tensor_tasks,
        spec=spec,
        steps=config.g_dynamic_steps,
        lr=config.g_dynamic_lr,
        seed=config.seed,
    )

    torch.save(
        {
            "state_dict": g_dynamic.state_dict(),
            "meta_feature_names": META_FEATURE_NAMES,
            "meta_mean": meta_mean,
            "meta_std": meta_std,
        },
        paths["models_hyper"] / "g_dynamic.pt",
    )

    g_dynamic_hist_frame = pd.DataFrame(g_dynamic_hist)
    g_dynamic_hist_frame.to_csv(paths["tables"] / "g_dynamic_history.csv", index=False)
    plot_dynamic_curve(g_dynamic_hist_frame, paths["plots"] / "dynamic_curve.png")

    zero_shot_dyn = evaluate_hypernet_zero_shot(g_dynamic, tensor_tasks, spec=spec)

    init_thetas_dyn: list[torch.Tensor] = []
    g_dynamic.eval()
    with torch.no_grad():
        for task in tensor_tasks:
            theta = g_dynamic(task.meta.unsqueeze(0)).squeeze(0).detach().cpu()
            init_thetas_dyn.append(theta)

    step8_results, step8_curve = _stage_train_f(
        stage_name="step8_g_dynamic_init",
        tasks=tensor_tasks,
        spec=spec,
        config=config,
        output_dir=paths["models_g_dyn"],
        init_thetas=init_thetas_dyn,
    )

    step8_frame = _results_to_frame("step8_g_dynamic_init", step8_results)
    step8_frame.to_csv(paths["tables"] / "step8_g_dynamic_init_results.csv", index=False)
    _save_curve(step8_curve, paths["curves"] / "step8_g_dynamic_init_avg.csv")

    all_stage_results = pd.concat([step4_frame, step6_frame, step8_frame], axis=0, ignore_index=True)
    all_stage_results.to_csv(paths["tables"] / "all_stage_results.csv", index=False)

    stage_summary = (
        all_stage_results.groupby("stage", as_index=False)
        .agg(
            mean_final_balanced_accuracy=("final_balanced_accuracy", "mean"),
            std_final_balanced_accuracy=("final_balanced_accuracy", "std"),
            mean_initial_balanced_accuracy=("initial_balanced_accuracy", "mean"),
        )
        .sort_values("stage")
        .reset_index(drop=True)
    )
    stage_summary.to_csv(paths["tables"] / "stage_summary.csv", index=False)

    curves = {
        "random_init": step4_curve,
        "g_supervised_init": step6_curve,
        "g_dynamic_init": step8_curve,
    }
    plot_average_learning_curves(curves, paths["plots"] / "learning_curves_compare.png")
    plot_final_comparison(stage_summary, paths["plots"] / "final_comparison.png")

    report = {
        "n_datasets": int(len(tasks)),
        "device": device_name,
        "target_num_classes": int(config.target_num_classes),
        "target_num_features": int(config.target_num_features),
        "meta_features": list(META_FEATURE_NAMES),
        "step4_random_mean_final": float(step4_frame["final_balanced_accuracy"].mean()),
        "step6_g_supervised_mean_final": float(step6_frame["final_balanced_accuracy"].mean()),
        "step8_g_dynamic_mean_final": float(step8_frame["final_balanced_accuracy"].mean()),
        "zero_shot_supervised_g_mean": float(np.mean(zero_shot_sup)),
        "zero_shot_dynamic_g_mean": float(np.mean(zero_shot_dyn)),
    }
    _write_json(report, paths["out"] / "report.json")

    lab_root = Path("auto_lab3")
    try:
        artifacts_prefix = config.out_dir.relative_to(lab_root).as_posix()
    except ValueError:
        artifacts_prefix = config.out_dir.as_posix()

    _write_readme(
        readme_path=Path("auto_lab3/README.md"),
        config=config,
        report=report,
        stage_summary=stage_summary,
        device_name=device_name,
        artifacts_prefix=artifacts_prefix,
    )

    return {
        "out_dir": paths["out"],
        "report_json": paths["out"] / "report.json",
        "results_csv": paths["tables"] / "all_stage_results.csv",
        "summary_csv": paths["tables"] / "stage_summary.csv",
        "device": device_name,
        "n_datasets": int(len(tasks)),
    }
