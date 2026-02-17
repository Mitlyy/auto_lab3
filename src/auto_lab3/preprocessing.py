from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig
from .data_io import RawDataset


META_FEATURE_NAMES: tuple[str, ...] = (
    "mf_log_rows_raw",
    "mf_log_features_raw",
    "mf_missing_ratio_raw",
    "mf_numeric_ratio_raw",
    "mf_log_classes_raw",
    "mf_class_entropy_merged",
    "mf_mean_abs_corr",
    "mf_feature_std_mean",
    "mf_feature_std_std",
    "mf_pca_var_sum",
    "mf_pca_var_first",
)


@dataclass
class ProcessedTask:
    dataset_id: int
    dataset_name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta_features: np.ndarray
    summary: dict[str, int | float | str]



def _likely_regression_target(y: pd.Series, max_raw_classes: int) -> bool:
    n_unique = int(y.nunique(dropna=True))
    numeric = pd.to_numeric(y, errors="coerce")
    numeric_ratio = float(numeric.notna().mean())
    return bool(numeric_ratio > 0.95 and n_unique > max_raw_classes)



def _encode_features(
    X: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    max_categories_per_feature: int,
) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []

    if numeric_columns:
        numeric = X[numeric_columns].copy()
        for col in numeric.columns:
            col_values = pd.to_numeric(numeric[col], errors="coerce").astype(float)
            median = col_values.median()
            fill_value = 0.0 if pd.isna(median) else float(median)
            numeric[col] = col_values.fillna(fill_value)
        blocks.append(numeric.astype(float))

    for col in categorical_columns:
        series = X[col].fillna("__MISSING__").astype(str)
        top_categories = series.value_counts(dropna=False).nlargest(max_categories_per_feature).index
        series = series.where(series.isin(top_categories), "__OTHER__")
        one_hot = pd.get_dummies(series, prefix=str(col), dtype=float)
        blocks.append(one_hot)

    if not blocks:
        raise ValueError("no features after encoding")

    encoded = pd.concat(blocks, axis=1)
    encoded = encoded.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    variance = encoded.var(axis=0)
    encoded = encoded.loc[:, variance > 1e-12]
    if encoded.shape[1] == 0:
        raise ValueError("all encoded features are constant")

    return encoded



def _merge_classes(y: np.ndarray, n_classes: int) -> tuple[np.ndarray, dict[str, int], list[int]]:
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < n_classes:
        raise ValueError("not enough target classes")

    order = np.argsort(-counts)
    bin_counts = np.zeros(n_classes, dtype=int)
    mapping: dict[str, int] = {}

    for idx in order:
        label = str(labels[idx])
        count = int(counts[idx])
        target_bin = int(np.argmin(bin_counts))
        mapping[label] = target_bin
        bin_counts[target_bin] += count

    merged = np.array([mapping[str(value)] for value in y], dtype=np.int64)
    return merged, mapping, bin_counts.tolist()



def _balanced_subsample(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    samples_per_class: int,
    min_samples_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    class_indices = [np.where(y == class_id)[0] for class_id in range(n_classes)]
    min_count = min(len(indices) for indices in class_indices)
    samples = min(samples_per_class, min_count)

    if samples < min_samples_per_class:
        raise ValueError(f"too few samples per class ({samples})")

    selected = np.concatenate(
        [rng.choice(indices, size=samples, replace=False) for indices in class_indices],
        axis=0,
    )
    rng.shuffle(selected)
    return X[selected], y[selected], samples



def _mean_abs_corr(X: np.ndarray, max_features: int) -> float:
    if X.shape[1] < 2:
        return 0.0

    if X.shape[1] > max_features:
        variances = np.var(X, axis=0)
        top_idx = np.argsort(-variances)[:max_features]
        X = X[:, top_idx]

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(corr[mask])))



def _class_entropy(y: np.ndarray, n_classes: int) -> float:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    probs = counts / max(counts.sum(), 1.0)
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    normalizer = np.log(float(n_classes))
    return float(entropy / normalizer) if normalizer > 0 else 0.0



def build_task_from_raw(raw: RawDataset, config: ExperimentConfig) -> ProcessedTask:
    rng = np.random.default_rng(config.seed + max(raw.dataset_id, 0))

    n_rows_raw = int(len(raw.X))
    n_features_raw = int(raw.X.shape[1])
    n_classes_raw = int(raw.y.nunique(dropna=True))
    missing_ratio_raw = float(raw.X.isna().mean().mean())
    numeric_ratio_raw = float(len(raw.numeric_columns) / max(n_features_raw, 1))

    if n_rows_raw < config.min_rows_raw:
        raise ValueError("too few rows")
    if _likely_regression_target(raw.y, config.max_raw_classes):
        raise ValueError("likely regression target")
    if n_classes_raw < config.target_num_classes:
        raise ValueError("not enough classes")
    if n_classes_raw > config.max_raw_classes:
        raise ValueError("too many raw classes")

    if n_rows_raw > config.max_rows_raw:
        sampled_idx = rng.choice(n_rows_raw, size=config.max_rows_raw, replace=False)
        sampled_idx = np.sort(sampled_idx)
        X_raw = raw.X.iloc[sampled_idx].reset_index(drop=True)
        y_raw = raw.y.iloc[sampled_idx].reset_index(drop=True)
    else:
        X_raw = raw.X.reset_index(drop=True)
        y_raw = raw.y.reset_index(drop=True)

    encoded = _encode_features(
        X=X_raw,
        numeric_columns=raw.numeric_columns,
        categorical_columns=raw.categorical_columns,
        max_categories_per_feature=config.max_categories_per_feature,
    )

    y_merged, _, merged_counts = _merge_classes(
        y=y_raw.astype(str).to_numpy(),
        n_classes=config.target_num_classes,
    )

    X_balanced, y_balanced, samples_per_class = _balanced_subsample(
        X=encoded.to_numpy(dtype=np.float32),
        y=y_merged,
        n_classes=config.target_num_classes,
        samples_per_class=config.samples_per_class,
        min_samples_per_class=config.min_samples_per_class,
        rng=rng,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)

    n_components = min(config.target_num_features, X_scaled.shape[1], X_scaled.shape[0] - 1)
    if n_components < 1:
        raise ValueError("invalid PCA components")

    pca = PCA(n_components=n_components, random_state=config.seed)
    X_projected = pca.fit_transform(X_scaled).astype(np.float32)

    if n_components < config.target_num_features:
        pad = np.zeros((X_projected.shape[0], config.target_num_features - n_components), dtype=np.float32)
        X_projected = np.hstack([X_projected, pad])

    X_train, X_test, y_train, y_test = train_test_split(
        X_projected,
        y_balanced,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=y_balanced,
    )

    stds = np.std(X_balanced, axis=0)
    explained = pca.explained_variance_ratio_ if n_components > 0 else np.array([0.0], dtype=float)

    meta_features = np.array(
        [
            np.log1p(n_rows_raw),
            np.log1p(n_features_raw),
            missing_ratio_raw,
            numeric_ratio_raw,
            np.log1p(n_classes_raw),
            _class_entropy(y_balanced, config.target_num_classes),
            _mean_abs_corr(X_balanced, max_features=config.max_corr_features),
            float(np.mean(stds)),
            float(np.std(stds)),
            float(np.sum(explained)),
            float(explained[0]),
        ],
        dtype=np.float32,
    )

    summary: dict[str, int | float | str] = {
        "dataset_id": raw.dataset_id,
        "dataset_name": raw.dataset_name,
        "rows_raw": n_rows_raw,
        "features_raw": n_features_raw,
        "classes_raw": n_classes_raw,
        "rows_used": int(X_projected.shape[0]),
        "features_encoded": int(encoded.shape[1]),
        "features_final": int(config.target_num_features),
        "classes_final": int(config.target_num_classes),
        "samples_per_class": int(samples_per_class),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "pca_var_sum": float(np.sum(explained)),
        "pca_var_first": float(explained[0]),
    }
    for class_id, count in enumerate(merged_counts):
        summary[f"merged_count_c{class_id}"] = int(count)

    return ProcessedTask(
        dataset_id=raw.dataset_id,
        dataset_name=raw.dataset_name,
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.int64),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.int64),
        meta_features=meta_features,
        summary=summary,
    )
