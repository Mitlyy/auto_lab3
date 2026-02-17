from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff


@dataclass
class RawDataset:
    dataset_id: int
    dataset_name: str
    target_column: str
    X: pd.DataFrame
    y: pd.Series
    numeric_columns: list[str]
    categorical_columns: list[str]



def load_openml_index(index_path: Path) -> dict[int, dict[str, str]]:
    if not index_path.exists():
        return {}

    frame = pd.read_csv(index_path)
    out: dict[int, dict[str, str]] = {}
    for _, row in frame.iterrows():
        dataset_id = int(row["id"])
        out[dataset_id] = {
            "name": str(row.get("name", dataset_id)),
            "target": str(row.get("target", "")),
        }
    return out



def list_dataset_paths(data_dir: Path) -> list[Path]:
    paths = [path for path in data_dir.glob("*.arff") if path.is_file()]

    def key_fn(path: Path) -> tuple[int, str]:
        stem = path.stem
        if stem.isdigit():
            return int(stem), path.name
        return (10**12, path.name)

    return sorted(paths, key=key_fn)



def _decode_bytes(value: object) -> object:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    return value



def _decode_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(_decode_bytes)
            out[col] = out[col].replace("?", np.nan)
    return out



def _pick_target_column(
    columns: list[str],
    dataset_id: int,
    index_map: dict[int, dict[str, str]],
) -> str:
    lowered = {col.lower(): col for col in columns}

    candidates: list[str] = []
    if dataset_id in index_map:
        target_candidate = index_map[dataset_id].get("target", "")
        if target_candidate:
            candidates.append(target_candidate)

    candidates.extend(["class", "target", "label", "binaryClass"])

    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    return columns[-1]



def _is_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    converted = pd.to_numeric(series, errors="coerce")
    non_missing = int(series.notna().sum())
    if non_missing == 0:
        return True
    return (int(converted.notna().sum()) / non_missing) >= 0.98



def load_raw_dataset(path: Path, index_map: dict[int, dict[str, str]]) -> RawDataset:
    dataset_id = int(path.stem) if path.stem.isdigit() else -1

    raw_data, _ = arff.loadarff(path)
    frame = pd.DataFrame(raw_data)
    if frame.empty:
        raise ValueError("empty frame")

    frame = _decode_dataframe(frame)

    target_column = _pick_target_column(frame.columns.tolist(), dataset_id=dataset_id, index_map=index_map)
    y = frame[target_column].copy().replace("?", np.nan)
    valid_mask = y.notna()
    if int(valid_mask.sum()) < 2:
        raise ValueError("too few valid target values")

    X = frame.loc[valid_mask, frame.columns != target_column].copy().reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    if X.shape[1] == 0:
        raise ValueError("no features")

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for col in X.columns:
        if _is_numeric_like(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
            numeric_columns.append(col)
        else:
            X[col] = X[col].astype("object")
            categorical_columns.append(col)

    dataset_name = index_map.get(dataset_id, {}).get("name", path.stem)

    return RawDataset(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        target_column=target_column,
        X=X,
        y=y.astype(str),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
