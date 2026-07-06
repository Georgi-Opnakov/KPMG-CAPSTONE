from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from data_loader import CITY_FILE_KEYS, get_paths


TARGET_COL = "log_price_eur"
PRICE_COL = "price_eur"
RANDOM_STATE = 42
MODEL_NAME = "XGBoost Tuned"


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def split_features(df: pd.DataFrame):
    y = df[TARGET_COL].copy()
    X = df.drop(columns=[PRICE_COL, TARGET_COL]).copy()
    numeric_cols = [column for column in X.columns if is_numeric_dtype(X[column])]
    categorical_cols = [column for column in X.columns if column not in numeric_cols]
    return X, y, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def to_euros(log_values) -> np.ndarray:
    return np.maximum(np.expm1(np.asarray(log_values)), 0)


def evaluate(y_true_log, y_pred_log) -> dict[str, float]:
    y_true = to_euros(y_true_log)
    y_pred = to_euros(y_pred_log)
    return {
        "rmse_eur": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae_eur": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def load_best_params(city: str) -> dict:
    tuning_path = get_paths()["ml_outputs"] / "xgboost_tuning_results.csv"
    tuning = pd.read_csv(tuning_path)
    row = tuning[(tuning["city"] == city) & (tuning["rank"] == 1)].iloc[0]
    return {
        "n_estimators": int(row["n_estimators"]),
        "learning_rate": float(row["learning_rate"]),
        "max_depth": int(row["max_depth"]),
        "subsample": float(row["subsample"]),
        "colsample_bytree": float(row["colsample_bytree"]),
        "reg_lambda": float(row["reg_lambda"]),
        "min_child_weight": float(row["min_child_weight"]),
    }


def feature_defaults(X: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> dict:
    defaults = {}
    for column in numeric_cols:
        defaults[column] = float(pd.to_numeric(X[column], errors="coerce").median())

    for column in categorical_cols:
        mode = X[column].dropna().astype(str).mode()
        defaults[column] = str(mode.iloc[0]) if not mode.empty else "Unknown"

    return defaults


def feature_options(X: pd.DataFrame, categorical_cols: list[str]) -> dict[str, list[str]]:
    options = {}
    for column in categorical_cols:
        values = sorted(X[column].dropna().astype(str).unique().tolist())
        options[column] = values[:500]
    return options


def train_city_model(city: str, output_dir: Path) -> dict:
    key = CITY_FILE_KEYS[city]
    model_ready_path = get_paths()["model_ready"] / f"{key}_model_ready.csv"
    master_path = get_paths()["master"] / f"{key}_master_model_dataset.csv"
    df = pd.read_csv(model_ready_path)
    master = pd.read_csv(master_path, usecols=["listing_id"])
    if len(master) != len(df):
        raise ValueError(f"{city} master and model-ready datasets do not have matching row counts.")

    X, y, numeric_cols, categorical_cols = split_features(df)
    params = load_best_params(city)

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    **params,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )
    pipeline.fit(X_train, y_train)
    holdout_metrics = evaluate(y_test, pipeline.predict(X_test))

    pipeline.fit(X, y)
    artifact = {
        "city": city,
        "model_name": MODEL_NAME,
        "pipeline": pipeline,
        "feature_columns": X.columns.tolist(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_defaults": feature_defaults(X, numeric_cols, categorical_cols),
        "feature_options": feature_options(X, categorical_cols),
        "target_col": TARGET_COL,
        "price_col": PRICE_COL,
        "training_rows": len(df),
        "holdout_metrics": holdout_metrics,
        "params": params,
        "source_file": str(model_ready_path),
        "listing_feature_file": str(output_dir / f"{key}_prediction_features.csv"),
    }

    output_path = output_dir / f"{key}_xgboost_tuned.joblib"
    feature_lookup = pd.concat([master[["listing_id"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    feature_lookup.to_csv(output_dir / f"{key}_prediction_features.csv", index=False)
    joblib.dump(artifact, output_path)
    return {
        "city": city,
        "model_name": MODEL_NAME,
        "artifact": str(output_path),
        "training_rows": len(df),
        **holdout_metrics,
        **params,
    }


def main() -> None:
    output_dir = get_paths()["outputs"] / "chatbot" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for city in CITY_FILE_KEYS:
        print(f"Training {city} {MODEL_NAME}...")
        rows.append(train_city_model(city, output_dir))

    manifest = pd.DataFrame(rows)
    manifest_path = output_dir / "price_model_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Saved manifest: {manifest_path}")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
