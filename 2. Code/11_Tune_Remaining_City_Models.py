from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

if importlib.util.find_spec("lightgbm"):
    from lightgbm import LGBMRegressor
else:
    LGBMRegressor = None

if importlib.util.find_spec("catboost"):
    from catboost import CatBoostRegressor
else:
    CatBoostRegressor = None


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
CV_FOLDS = 3
TEST_SIZE = 0.20
MODEL_N_JOBS = -1
TARGET_COL = "log_price_eur"
PRICE_COL = "price_eur"

TUNED_MODEL_NAMES = [
    "LightGBM Tuned",
    "Extra Trees Tuned",
    "CatBoost Tuned",
    "Decision Tree Tuned",
    "Linear Regression Tuned",
]

ORIGINAL_COMBINED_RESULTS = {
    "Linear Regression": {"rmse_eur": 43.3501, "mae_eur": 30.4450, "r2": 0.4464},
    "XGBoost": {"rmse_eur": 36.8795, "mae_eur": 25.8748, "r2": 0.5993},
}


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / "1. Data").exists() and (candidate / "2. Code").exists():
            return candidate
    raise FileNotFoundError("Could not find CAPSTONE project root.")


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "1. Data"
MODEL_READY_DIR = DATA_DIR / "model_ready"
OUTPUT_DIR = DATA_DIR / "Outputs" / "ml_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CITY_FILES = {
    "Madrid": MODEL_READY_DIR / "madrid_model_ready.csv",
    "Tokyo": MODEL_READY_DIR / "tokyo_model_ready.csv",
}


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    y = df[TARGET_COL].copy()
    X = df.drop(columns=[PRICE_COL, TARGET_COL]).copy()
    numeric_cols = [col for col in X.columns if is_numeric_dtype(X[col])]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    return X, y, numeric_cols, categorical_cols


def to_euros(log_values: pd.Series | np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(log_values))


def evaluate_on_euros(y_true_log: pd.Series, y_pred_log: np.ndarray) -> dict[str, float]:
    y_true = to_euros(y_true_log)
    y_pred = np.maximum(to_euros(y_pred_log), 0)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "rmse_eur": round(float(rmse), 4),
        "mae_eur": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def euro_rmse_score(y_true_log: pd.Series, y_pred_log: np.ndarray) -> float:
    y_true = to_euros(y_true_log)
    y_pred = np.maximum(to_euros(y_pred_log), 0)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


RMSE_SCORER = make_scorer(euro_rmse_score, greater_is_better=False)


def clean_feature_name(feature_name: str) -> str:
    return (
        feature_name
        .replace("numeric__", "")
        .replace("categorical__", "")
        .replace("_", " ")
    )


def model_search_specs(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict[str, tuple[Pipeline, dict | list[dict], int]]:
    specs: dict[str, tuple[Pipeline, dict | list[dict], int]] = {}

    specs["Decision Tree Tuned"] = (
        Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
                ("model", DecisionTreeRegressor(random_state=RANDOM_STATE)),
            ]
        ),
        {
            "model__max_depth": [6, 8, 10, 12, 16, 20, None],
            "model__min_samples_leaf": [5, 10, 20, 40, 80],
            "model__min_samples_split": [10, 20, 40, 80],
            "model__max_features": [None, "sqrt", 0.7, 0.9],
        },
        18,
    )

    specs["Extra Trees Tuned"] = (
        Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
                ("model", ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=MODEL_N_JOBS)),
            ]
        ),
        {
            "model__n_estimators": [200, 300, 450],
            "model__max_depth": [16, 24, 32, None],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["sqrt", 0.6, 0.8, 1.0],
        },
        10,
    )

    specs["Linear Regression Tuned"] = (
        Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)),
                ("model", Ridge()),
            ]
        ),
        [
            {
                "model": [Ridge()],
                "model__alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
            },
            {
                "model": [ElasticNet(max_iter=20_000, random_state=RANDOM_STATE)],
                "model__alpha": [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                "model__l1_ratio": [0.05, 0.15, 0.3, 0.5, 0.7],
            },
        ],
        12,
    )

    if LGBMRegressor is not None:
        specs["LightGBM Tuned"] = (
            Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
                    (
                        "model",
                        LGBMRegressor(
                            objective="regression",
                            random_state=RANDOM_STATE,
                            n_jobs=MODEL_N_JOBS,
                            verbose=-1,
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [350, 500, 700],
                "model__learning_rate": [0.025, 0.04, 0.06, 0.08],
                "model__num_leaves": [31, 63, 95],
                "model__max_depth": [-1, 8, 12, 16],
                "model__min_child_samples": [10, 20, 40, 80],
                "model__subsample": [0.75, 0.85, 0.95],
                "model__colsample_bytree": [0.75, 0.85, 0.95],
                "model__reg_lambda": [0.0, 0.5, 1.0, 3.0],
            },
            12,
        )

    if CatBoostRegressor is not None:
        specs["CatBoost Tuned"] = (
            Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
                    (
                        "model",
                        CatBoostRegressor(
                            loss_function="RMSE",
                            random_seed=RANDOM_STATE,
                            verbose=False,
                            allow_writing_files=False,
                            thread_count=MODEL_N_JOBS,
                        ),
                    ),
                ]
            ),
            {
                "model__iterations": [350, 500, 700],
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__depth": [4, 6, 8],
                "model__l2_leaf_reg": [1, 3, 5, 8],
                "model__random_strength": [0.5, 1.0, 2.0],
            },
            8,
        )

    return specs


def search_model(
    model_name: str,
    pipeline: Pipeline,
    param_distributions: dict | list[dict],
    n_iter: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomizedSearchCV:
    print(f"    Searching {model_name} ({n_iter} sampled settings)...", flush=True)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=RMSE_SCORER,
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search


def get_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocess"]
    names = preprocessor.get_feature_names_out()
    return [clean_feature_name(name) for name in names]


def feature_importance_rows(city: str, model_name: str, pipeline: Pipeline) -> list[dict]:
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline)

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:
        return []

    rows = []
    for feature, importance in zip(feature_names, importances):
        rows.append(
            {
                "city": city,
                "model": model_name,
                "feature": feature,
                "importance": round(float(importance), 6),
            }
        )
    return rows


def tuning_result_rows(city: str, model_name: str, search: RandomizedSearchCV) -> list[dict]:
    cv_results = pd.DataFrame(search.cv_results_)
    rows = []

    for _, row in cv_results.sort_values("rank_test_score").iterrows():
        params = row["params"]
        flat_params = {
            key.replace("model__", ""): str(value)
            for key, value in params.items()
            if key != "model"
        }
        if "model" in params:
            flat_params["linear_estimator"] = params["model"].__class__.__name__

        rows.append(
            {
                "city": city,
                "model": model_name,
                "rank": int(row["rank_test_score"]),
                "cv_rmse_eur": round(float(-row["mean_test_score"]), 4),
                "cv_rmse_std": round(float(row["std_test_score"]), 4),
                **flat_params,
            }
        )
    return rows


def prediction_frame(
    city: str,
    model_name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    y_pred_log = pipeline.predict(X_test)
    pred_df = X_test.copy()
    pred_df["city"] = city
    pred_df["model"] = model_name
    pred_df["actual_price_eur"] = to_euros(y_test)
    pred_df["predicted_price_eur"] = np.maximum(to_euros(y_pred_log), 0)
    pred_df["residual_eur"] = pred_df["actual_price_eur"] - pred_df["predicted_price_eur"]
    pred_df["abs_error_eur"] = pred_df["residual_eur"].abs()
    return pred_df


def load_existing_csv(file_name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / file_name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def remove_old_tuned_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "model" not in df.columns:
        return df
    return df[~df["model"].astype(str).isin(TUNED_MODEL_NAMES)].copy()


def recompute_comparison(results_df: pd.DataFrame) -> pd.DataFrame:
    comparison_rows = []

    for city in sorted(results_df["city"].dropna().unique()):
        city_results = results_df[results_df["city"] == city].copy()
        best_row = city_results.sort_values("rmse_eur").iloc[0]
        xgb_rows = city_results[city_results["model"] == "XGBoost"]
        xgb_row = xgb_rows.iloc[0] if not xgb_rows.empty else best_row

        for label, row in [("Best city-specific model", best_row), ("City-specific XGBoost", xgb_row)]:
            original = ORIGINAL_COMBINED_RESULTS["XGBoost"]
            comparison_rows.append(
                {
                    "city": city,
                    "comparison_scope": label,
                    "city_model": row["model"],
                    "city_rmse_eur": row["rmse_eur"],
                    "city_mae_eur": row["mae_eur"],
                    "city_r2": row["r2"],
                    "original_reference_model": "Combined-data XGBoost",
                    "original_rmse_eur": original["rmse_eur"],
                    "original_mae_eur": original["mae_eur"],
                    "original_r2": original["r2"],
                    "rmse_change_vs_original": round(float(row["rmse_eur"] - original["rmse_eur"]), 4),
                    "mae_change_vs_original": round(float(row["mae_eur"] - original["mae_eur"]), 4),
                    "r2_change_vs_original": round(float(row["r2"] - original["r2"]), 4),
                }
            )

    return pd.DataFrame(comparison_rows)


def recompute_recommendation_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    for city in sorted(results_df["city"].dropna().unique()):
        city_results = results_df[results_df["city"] == city].sort_values("rmse_eur")
        best = city_results.iloc[0]
        original = ORIGINAL_COMBINED_RESULTS["XGBoost"]
        summary_rows.append(
            {
                "city": city,
                "best_model": best["model"],
                "best_rmse_eur": best["rmse_eur"],
                "best_mae_eur": best["mae_eur"],
                "best_r2": best["r2"],
                "rmse_improvement_vs_original_xgboost": round(float(original["rmse_eur"] - best["rmse_eur"]), 4),
                "mae_improvement_vs_original_xgboost": round(float(original["mae_eur"] - best["mae_eur"]), 4),
                "r2_improvement_vs_original_xgboost": round(float(best["r2"] - original["r2"]), 4),
            }
        )

    return pd.DataFrame(summary_rows)


def recompute_best_outputs(results_df: pd.DataFrame, predictions_df: pd.DataFrame) -> None:
    segment_rows = []
    best_prediction_frames = []

    for city in sorted(results_df["city"].dropna().unique()):
        best_model_name = results_df[results_df["city"] == city].sort_values("rmse_eur").iloc[0]["model"]
        city_pred = predictions_df[
            (predictions_df["city"] == city) & (predictions_df["model"] == best_model_name)
        ].copy()
        if city_pred.empty:
            continue

        city_pred["best_model"] = best_model_name
        best_prediction_frames.append(city_pred)

        city_pred["actual_price_band"] = pd.qcut(
            city_pred["actual_price_eur"],
            q=4,
            labels=["low", "mid-low", "mid-high", "high"],
            duplicates="drop",
        )

        possible_segments = [
            "room_type",
            "property_group",
            "capacity_bucket",
            "actual_price_band",
            "neighbourhood_cleansed",
        ]

        for segment_col in possible_segments:
            if segment_col not in city_pred.columns:
                continue

            grouped = (
                city_pred
                .groupby(segment_col, dropna=False)
                .agg(
                    rows=("abs_error_eur", "size"),
                    mae_eur=("abs_error_eur", "mean"),
                    median_abs_error_eur=("abs_error_eur", "median"),
                    p90_abs_error_eur=("abs_error_eur", lambda x: np.percentile(x, 90)),
                    mean_actual_price_eur=("actual_price_eur", "mean"),
                )
                .reset_index()
                .rename(columns={segment_col: "segment"})
            )
            grouped = grouped[grouped["rows"] >= 30]
            grouped["city"] = city
            grouped["model"] = best_model_name
            grouped["segment_type"] = segment_col
            segment_rows.append(grouped)

    if best_prediction_frames:
        pd.concat(best_prediction_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "best_model_predictions.csv",
            index=False,
        )

    if segment_rows:
        error_segments_df = pd.concat(segment_rows, ignore_index=True)
        for col in ["mae_eur", "median_abs_error_eur", "p90_abs_error_eur", "mean_actual_price_eur"]:
            error_segments_df[col] = error_segments_df[col].round(4)
        error_segments_df.to_csv(OUTPUT_DIR / "best_model_error_segments.csv", index=False)


def main() -> None:
    print("Project root:", PROJECT_ROOT, flush=True)
    print("Output folder:", OUTPUT_DIR, flush=True)
    print("Loading model-ready datasets...", flush=True)

    datasets = {city: pd.read_csv(path) for city, path in CITY_FILES.items()}
    for city, df in datasets.items():
        print(f"  {city}: {df.shape[0]:,} rows, {df.shape[1]:,} columns", flush=True)

    new_results = []
    new_predictions = []
    new_feature_importance = []
    tuning_rows = []

    for city, df in datasets.items():
        print(f"\nTuning remaining models for {city}...", flush=True)
        X, y, numeric_cols, categorical_cols = split_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        for model_name, (pipeline, param_distributions, n_iter) in model_search_specs(numeric_cols, categorical_cols).items():
            search = search_model(
                model_name,
                pipeline,
                param_distributions,
                n_iter,
                X_train,
                y_train,
            )
            best_pipeline = search.best_estimator_
            y_pred_log = best_pipeline.predict(X_test)
            metrics = evaluate_on_euros(y_test, y_pred_log)

            print(
                f"      Best {model_name}: RMSE {metrics['rmse_eur']:.2f}, "
                f"MAE {metrics['mae_eur']:.2f}, R2 {metrics['r2']:.4f}",
                flush=True,
            )

            new_results.append(
                {
                    "scope": "City-specific",
                    "city": city,
                    "model": model_name,
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                    "numeric_features": len(numeric_cols),
                    "categorical_features": len(categorical_cols),
                    **metrics,
                }
            )
            new_predictions.append(prediction_frame(city, model_name, best_pipeline, X_test, y_test))
            new_feature_importance.extend(feature_importance_rows(city, model_name, best_pipeline))
            tuning_rows.extend(tuning_result_rows(city, model_name, search))

    existing_results = remove_old_tuned_rows(load_existing_csv("expanded_city_model_results.csv"))
    results_df = pd.concat([existing_results, pd.DataFrame(new_results)], ignore_index=True)
    results_df = results_df.sort_values(["city", "rmse_eur"]).reset_index(drop=True)
    results_df.to_csv(OUTPUT_DIR / "expanded_city_model_results.csv", index=False)

    existing_predictions = remove_old_tuned_rows(load_existing_csv("expanded_city_model_predictions.csv"))
    predictions_df = pd.concat([existing_predictions, *new_predictions], ignore_index=True)
    predictions_df.to_csv(OUTPUT_DIR / "expanded_city_model_predictions.csv", index=False)

    existing_importance = remove_old_tuned_rows(load_existing_csv("expanded_city_feature_importance.csv"))
    feature_importance_df = pd.concat(
        [existing_importance, pd.DataFrame(new_feature_importance)],
        ignore_index=True,
    )
    feature_importance_df.to_csv(OUTPUT_DIR / "expanded_city_feature_importance.csv", index=False)

    pd.DataFrame(tuning_rows).to_csv(OUTPUT_DIR / "additional_model_tuning_results.csv", index=False)

    comparison_df = recompute_comparison(results_df)
    comparison_df.to_csv(OUTPUT_DIR / "expanded_city_vs_original_comparison.csv", index=False)

    recommendation_df = recompute_recommendation_summary(results_df)
    recommendation_df.to_csv(OUTPUT_DIR / "expanded_model_recommendation_summary.csv", index=False)

    recompute_best_outputs(results_df, predictions_df)

    print("\nUpdated model leaderboard:", flush=True)
    print(results_df.to_string(index=False), flush=True)
    print("\nUpdated best model summary:", flush=True)
    print(recommendation_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
