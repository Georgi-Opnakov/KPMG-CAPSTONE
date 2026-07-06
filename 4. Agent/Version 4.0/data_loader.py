from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


CITY_FILE_KEYS = {
    "Madrid": "madrid",
    "Tokyo": "tokyo",
}

ML_OUTPUT_FILES = {
    "model_results": "expanded_city_model_results.csv",
    "cv_results": "expanded_city_cv_results.csv",
    "model_comparison": "expanded_city_vs_original_comparison.csv",
    "recommendation_summary": "expanded_model_recommendation_summary.csv",
    "feature_importance": "expanded_city_feature_importance.csv",
    "error_segments": "best_model_error_segments.csv",
    "listing_type_results": "listing_type_model_results.csv",
    "xgboost_tuning": "xgboost_tuning_results.csv",
    "random_forest_tuning": "random_forest_tuning_results.csv",
    "additional_model_tuning": "additional_model_tuning_results.csv",
}


def find_project_root(start: Path | None = None) -> Path:
    """Find the CAPSTONE project root from the agent folder or a child path."""
    current = Path(__file__).resolve() if start is None else Path(start).resolve()
    search_points = [current.parent, *current.parents]
    app_folder_markers = ["4. Agent", "4. Chatbot"]

    for candidate in search_points:
        has_data_folder = (candidate / "1. Data").exists()
        has_app_folder = any((candidate / marker).exists() for marker in app_folder_markers)
        if has_data_folder and has_app_folder:
            return candidate

    raise FileNotFoundError("Could not find CAPSTONE project root.")


def get_paths() -> dict[str, Path]:
    root = find_project_root()
    data_dir = root / "1. Data"
    return {
        "project_root": root,
        "data": data_dir,
        "raw": data_dir / "raw",
        "master": data_dir / "master",
        "model_ready": data_dir / "model_ready",
        "outputs": data_dir / "Outputs",
        "ml_outputs": data_dir / "Outputs" / "ml_models",
    }


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_master_data() -> dict[str, pd.DataFrame]:
    paths = get_paths()
    data: dict[str, pd.DataFrame] = {}

    for city, key in CITY_FILE_KEYS.items():
        file_path = paths["master"] / f"{key}_master_model_dataset.csv"
        data[city] = _read_csv_if_exists(file_path)

    return data


@lru_cache(maxsize=1)
def load_model_ready_data() -> dict[str, pd.DataFrame]:
    paths = get_paths()
    data: dict[str, pd.DataFrame] = {}

    for city, key in CITY_FILE_KEYS.items():
        file_path = paths["model_ready"] / f"{key}_model_ready.csv"
        data[city] = _read_csv_if_exists(file_path)

    return data


@lru_cache(maxsize=1)
def load_ml_outputs() -> dict[str, pd.DataFrame]:
    paths = get_paths()
    outputs: dict[str, pd.DataFrame] = {}

    for output_name, file_name in ML_OUTPUT_FILES.items():
        outputs[output_name] = _read_csv_if_exists(paths["ml_outputs"] / file_name)

    return outputs


@lru_cache(maxsize=1)
def load_all_data() -> dict[str, Any]:
    return {
        "paths": get_paths(),
        "master": load_master_data(),
        "model_ready": load_model_ready_data(),
        "ml_outputs": load_ml_outputs(),
    }


def available_cities(master_data: dict[str, pd.DataFrame] | None = None) -> list[str]:
    data = master_data if master_data is not None else load_master_data()
    return [city for city, df in data.items() if not df.empty]


def data_inventory() -> pd.DataFrame:
    bundle = load_all_data()
    rows: list[dict[str, Any]] = []

    for layer in ["raw", "master", "model_ready"]:
        folder = bundle["paths"][layer]
        if not folder.exists():
            continue
        for file_path in sorted(folder.glob("*.csv")):
            rows.append(
                {
                    "layer": layer,
                    "file": file_path.name,
                    "size_mb": round(file_path.stat().st_size / 1_000_000, 2),
                    "path": str(file_path),
                }
            )

    ml_folder = bundle["paths"]["ml_outputs"]
    if ml_folder.exists():
        for file_path in sorted(ml_folder.glob("*.csv")):
            rows.append(
                {
                    "layer": "ml_outputs",
                    "file": file_path.name,
                    "size_mb": round(file_path.stat().st_size / 1_000_000, 2),
                    "path": str(file_path),
                }
            )

    chatbot_output_folder = bundle["paths"]["outputs"] / "chatbot"
    if chatbot_output_folder.exists():
        for file_path in sorted(chatbot_output_folder.glob("*")):
            if not file_path.is_file():
                continue
            rows.append(
                {
                    "layer": "chatbot_outputs",
                    "file": file_path.name,
                    "size_mb": round(file_path.stat().st_size / 1_000_000, 2),
                    "path": str(file_path),
                }
            )

    return pd.DataFrame(rows)
