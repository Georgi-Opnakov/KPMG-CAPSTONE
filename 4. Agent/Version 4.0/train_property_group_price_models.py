from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from data_loader import CITY_FILE_KEYS, get_paths


TARGET_COL = "log_price_eur"
PRICE_COL = "price_eur"
SEGMENT_COL = "property_group"
RANDOM_STATE = 42


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
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


def clean_param(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, str) and value == "None":
        return None
    return value


def load_tuned_params(city: str, model_name: str) -> dict[str, Any]:
    paths = get_paths()

    if model_name == "XGBoost Tuned":
        tuning = pd.read_csv(paths["ml_outputs"] / "xgboost_tuning_results.csv")
    else:
        tuning = pd.read_csv(paths["ml_outputs"] / "additional_model_tuning_results.csv")

    match = tuning[
        (tuning["city"].astype(str) == city)
        & (tuning["model"].astype(str) == model_name if "model" in tuning.columns else True)
        & (pd.to_numeric(tuning["rank"], errors="coerce") == 1)
    ].head(1)

    if match.empty:
        return {}

    row = match.iloc[0]
    if model_name == "XGBoost Tuned":
        return {
            "n_estimators": int(row["n_estimators"]),
            "learning_rate": float(row["learning_rate"]),
            "max_depth": int(row["max_depth"]),
            "subsample": float(row["subsample"]),
            "colsample_bytree": float(row["colsample_bytree"]),
            "reg_lambda": float(row["reg_lambda"]),
            "min_child_weight": float(row["min_child_weight"]),
        }

    if model_name == "LightGBM Tuned":
        return {
            "n_estimators": int(row["n_estimators"]),
            "learning_rate": float(row["learning_rate"]),
            "max_depth": int(row["max_depth"]),
            "num_leaves": int(row["num_leaves"]),
            "min_child_samples": int(row["min_child_samples"]),
            "subsample": float(row["subsample"]),
            "colsample_bytree": float(row["colsample_bytree"]),
            "reg_lambda": float(row["reg_lambda"]),
        }

    if model_name == "CatBoost Tuned":
        return {
            "iterations": int(row["iterations"]),
            "learning_rate": float(row["learning_rate"]),
            "depth": int(row["depth"]),
            "l2_leaf_reg": float(row["l2_leaf_reg"]),
            "random_strength": float(row["random_strength"]),
        }

    if model_name in {"Random Forest Tuned", "Extra Trees Tuned"}:
        return {
            "n_estimators": int(row["n_estimators"]),
            "max_depth": clean_param(row["max_depth"]),
            "min_samples_split": int(row["min_samples_split"]),
            "min_samples_leaf": int(row["min_samples_leaf"]),
            "max_features": clean_param(row["max_features"]),
        }

    if model_name == "Decision Tree Tuned":
        return {
            "max_depth": clean_param(row["max_depth"]),
            "min_samples_split": int(row["min_samples_split"]),
            "min_samples_leaf": int(row["min_samples_leaf"]),
            "max_features": clean_param(row["max_features"]),
        }

    if model_name == "Linear Regression Tuned":
        estimator = str(row.get("linear_estimator", "Ridge"))
        params: dict[str, Any] = {"linear_estimator": estimator, "alpha": float(row.get("alpha", 1.0))}
        if estimator == "ElasticNet" and pd.notna(row.get("l1_ratio")):
            params["l1_ratio"] = float(row["l1_ratio"])
        return params

    return {}


def build_model(model_name: str, params: dict[str, Any]):
    if model_name == "XGBoost Tuned":
        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=RANDOM_STATE,
            n_jobs=1,
            **params,
        )
    if model_name == "LightGBM Tuned":
        return LGBMRegressor(
            objective="regression",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=-1,
            **params,
        )
    if model_name == "CatBoost Tuned":
        return CatBoostRegressor(
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            thread_count=1,
            verbose=False,
            **params,
        )
    if model_name == "Random Forest Tuned":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1, **params)
    if model_name == "Extra Trees Tuned":
        return ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1, **params)
    if model_name == "Decision Tree Tuned":
        return DecisionTreeRegressor(random_state=RANDOM_STATE, **params)
    if model_name == "Linear Regression Tuned":
        estimator = params.get("linear_estimator", "Ridge")
        if estimator == "Lasso":
            return Lasso(alpha=params.get("alpha", 1.0), random_state=RANDOM_STATE, max_iter=10000)
        if estimator == "ElasticNet":
            return ElasticNet(
                alpha=params.get("alpha", 1.0),
                l1_ratio=params.get("l1_ratio", 0.5),
                random_state=RANDOM_STATE,
                max_iter=10000,
            )
        return Ridge(alpha=params.get("alpha", 1.0))

    raise ValueError(f"Unsupported model for artifact export: {model_name}")


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


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown"


def train_property_group_model(city: str, property_group: str, model_name: str, output_dir: Path) -> dict:
    key = CITY_FILE_KEYS[city]
    model_ready_path = get_paths()["model_ready"] / f"{key}_model_ready.csv"
    master_path = get_paths()["master"] / f"{key}_master_model_dataset.csv"

    df = pd.read_csv(model_ready_path)
    master = pd.read_csv(master_path, usecols=["listing_id", SEGMENT_COL])
    if len(master) != len(df):
        raise ValueError(f"{city} master and model-ready datasets do not have matching row counts.")
    if SEGMENT_COL not in df.columns:
        raise ValueError(f"{model_ready_path} does not contain {SEGMENT_COL}.")

    mask = df[SEGMENT_COL].astype(str) == property_group
    segment_df = df.loc[mask].copy()
    segment_master = master.loc[mask, ["listing_id", SEGMENT_COL]].reset_index(drop=True)
    if segment_df.empty:
        raise ValueError(f"No rows found for {city} / {property_group}.")

    X, y, numeric_cols, categorical_cols = split_features(segment_df)
    params = load_tuned_params(city, model_name)
    model = build_model(model_name, params)
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", model),
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

    group_slug = slugify(property_group)
    model_slug = slugify(model_name.replace(" Tuned", ""))
    artifact_path = output_dir / f"{key}_property_group_{group_slug}_{model_slug}_tuned.joblib"
    feature_lookup_path = output_dir / f"{key}_property_group_{group_slug}_prediction_features.csv"

    feature_lookup = pd.concat([segment_master[["listing_id"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    feature_lookup.to_csv(feature_lookup_path, index=False)

    artifact = {
        "city": city,
        "segment_col": SEGMENT_COL,
        "segment_value": property_group,
        "model_name": model_name,
        "pipeline": pipeline,
        "feature_columns": X.columns.tolist(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_defaults": feature_defaults(X, numeric_cols, categorical_cols),
        "feature_options": feature_options(X, categorical_cols),
        "target_col": TARGET_COL,
        "price_col": PRICE_COL,
        "training_rows": len(segment_df),
        "holdout_metrics": holdout_metrics,
        "params": params,
        "source_file": str(model_ready_path),
        "listing_feature_file": str(feature_lookup_path),
    }
    joblib.dump(artifact, artifact_path)

    return {
        "city": city,
        "property_group": property_group,
        "model_name": model_name,
        "artifact": str(artifact_path),
        "listing_feature_file": str(feature_lookup_path),
        "training_rows": len(segment_df),
        **holdout_metrics,
        **params,
    }


def main() -> None:
    paths = get_paths()
    best_path = paths["ml_outputs"] / "property_group_best_models.csv"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing property-group best-model file: {best_path}")

    output_dir = paths["outputs"] / "chatbot" / "models" / "property_groups"
    output_dir.mkdir(parents=True, exist_ok=True)

    best = pd.read_csv(best_path)
    rows = []
    for record in best.itertuples(index=False):
        city = str(record.city)
        property_group = str(record.property_group)
        model_name = str(record.model)
        print(f"Training {city} / {property_group} with {model_name}...")
        rows.append(train_property_group_model(city, property_group, model_name, output_dir))

    manifest = pd.DataFrame(rows)
    manifest_path = output_dir / "property_group_price_model_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Saved property-group model manifest: {manifest_path}")
    print(manifest[["city", "property_group", "model_name", "training_rows", "rmse_eur", "mae_eur", "r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
