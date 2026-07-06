from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from data_loader import CITY_FILE_KEYS, get_paths


def model_leaderboard(ml_outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    results = ml_outputs.get("model_results", pd.DataFrame()).copy()
    if results.empty:
        return results

    numeric_cols = ["rmse_eur", "mae_eur", "r2"]
    for column in numeric_cols:
        if column in results.columns:
            results[column] = pd.to_numeric(results[column], errors="coerce")

    return results.sort_values(["city", "rmse_eur"])


def best_models(ml_outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    leaderboard = model_leaderboard(ml_outputs)
    if leaderboard.empty:
        return leaderboard

    return leaderboard.sort_values("rmse_eur").groupby("city", as_index=False).first()


def model_summary_text(ml_outputs: dict[str, pd.DataFrame]) -> str:
    best = best_models(ml_outputs)
    if best.empty:
        return "Model results are not available yet."

    lines = ["Best current price prediction models:"]
    for _, row in best.iterrows():
        lines.append(
            f"- {row['city']}: {row['model']} with RMSE {row['rmse_eur']:.2f}, "
            f"MAE {row['mae_eur']:.2f}, R2 {row['r2']:.4f}."
        )
    return "\n".join(lines)


def top_features(
    ml_outputs: dict[str, pd.DataFrame],
    city: str,
    model: str | None = None,
    limit: int = 12,
) -> pd.DataFrame:
    features = ml_outputs.get("feature_importance", pd.DataFrame()).copy()
    if features.empty:
        return features

    features = features[features["city"].astype(str) == city].copy()
    if model:
        features = features[features["model"].astype(str) == model].copy()

    if "importance" in features.columns:
        features["importance"] = pd.to_numeric(features["importance"], errors="coerce")
        features = features.sort_values("importance", ascending=False)

    return features.head(limit)


def error_segments(
    ml_outputs: dict[str, pd.DataFrame],
    city: str,
    segment_type: str | None = None,
    limit: int = 20,
) -> pd.DataFrame:
    segments = ml_outputs.get("error_segments", pd.DataFrame()).copy()
    if segments.empty:
        return segments

    segments = segments[segments["city"].astype(str) == city].copy()
    if segment_type and "segment_type" in segments.columns:
        segments = segments[segments["segment_type"].astype(str) == segment_type]

    if "mae_eur" in segments.columns:
        segments["mae_eur"] = pd.to_numeric(segments["mae_eur"], errors="coerce")
        segments = segments.sort_values("mae_eur", ascending=False)

    return segments.head(limit)


def prediction_status() -> str:
    manifest = price_model_manifest()
    if manifest.empty:
        return (
            "Model metrics are available, but reusable price model artifacts have not been "
            "trained yet. Run `python train_price_models.py` from the chatbot folder."
        )

    cities = ", ".join(manifest["city"].astype(str).tolist())
    return (
        f"The Price Check tab can reuse saved city-level price-model artifacts for: {cities}. "
        "Cached future-price tables may use the newer property-group winners, while live fallback predictions use the saved reusable artifacts."
    )


def price_model_dir() -> Path:
    return get_paths()["outputs"] / "chatbot" / "models"


def price_model_manifest() -> pd.DataFrame:
    manifest_path = price_model_dir() / "price_model_manifest.csv"
    if not manifest_path.exists():
        return pd.DataFrame()
    return pd.read_csv(manifest_path)


@lru_cache(maxsize=4)
def load_price_model(city: str) -> dict[str, Any]:
    key = CITY_FILE_KEYS[city]
    artifact_path = price_model_dir() / f"{key}_xgboost_tuned.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Missing price model artifact for {city}: {artifact_path}. "
            "Run `python train_price_models.py` from the chatbot folder."
        )
    return joblib.load(artifact_path)


def available_price_model_cities() -> list[str]:
    manifest = price_model_manifest()
    if manifest.empty:
        return []
    return manifest["city"].dropna().astype(str).tolist()


def model_feature_options(city: str) -> dict[str, list[str]]:
    artifact = load_price_model(city)
    return artifact.get("feature_options", {})


def model_feature_defaults(city: str) -> dict:
    artifact = load_price_model(city)
    return artifact.get("feature_defaults", {})


@lru_cache(maxsize=4)
def load_listing_feature_lookup(city: str) -> pd.DataFrame:
    artifact = load_price_model(city)
    feature_file = artifact.get("listing_feature_file")
    if not feature_file:
        return pd.DataFrame()

    path = Path(feature_file)
    if not path.exists():
        local_path = price_model_dir() / path.name
        if local_path.exists():
            path = local_path
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def prepare_prediction_row(city: str, overrides: dict | pd.Series | None = None) -> pd.DataFrame:
    artifact = load_price_model(city)
    row = artifact["feature_defaults"].copy()

    if overrides is not None:
        items = overrides.items() if isinstance(overrides, dict) else overrides.to_dict().items()
        for key, value in items:
            if key in row and pd.notna(value):
                row[key] = value

    return pd.DataFrame([row], columns=artifact["feature_columns"])


def predict_price(city: str, overrides: dict | pd.Series | None = None) -> dict[str, Any]:
    artifact = load_price_model(city)
    X = prepare_prediction_row(city, overrides)
    predicted_log_price = float(artifact["pipeline"].predict(X)[0])
    predicted_price = max(float(np.expm1(predicted_log_price)), 0.0)

    return {
        "city": city,
        "model_name": artifact["model_name"],
        "predicted_nightly_price_eur": round(predicted_price, 2),
        "holdout_metrics": artifact.get("holdout_metrics", {}),
        "training_rows": artifact.get("training_rows"),
    }


def predict_listing_price(city: str, listing_row: pd.Series) -> dict[str, Any]:
    listing_id = listing_row.get("listing_id")
    if pd.notna(listing_id):
        lookup = load_listing_feature_lookup(city)
        if not lookup.empty:
            match = lookup[lookup["listing_id"] == int(listing_id)].head(1)
            if not match.empty:
                overrides = match.drop(columns=["listing_id"]).iloc[0]
                result = predict_price(city, overrides)
                result["listing_id"] = int(listing_id)
                result["feature_source"] = "model_ready_feature_lookup"
                return result

    result = predict_price(city, listing_row)
    if pd.notna(listing_id):
        result["listing_id"] = int(listing_id)
    result["feature_source"] = "master_row_with_model_defaults"
    return result


def prediction_summary_text(result: dict[str, Any], nights: int | None = None) -> str:
    price = result["predicted_nightly_price_eur"]
    metrics = result.get("holdout_metrics", {})
    text = (
        f"Predicted nightly price for {result['city']}: EUR {price:,.0f} "
        f"using {result['model_name']}."
    )

    if nights and nights > 0:
        text += f" Estimated total for {nights} nights: EUR {price * nights:,.0f}."

    if metrics:
        text += (
            f" Reference model accuracy: RMSE {metrics.get('rmse_eur')}, "
            f"MAE {metrics.get('mae_eur')}, R2 {metrics.get('r2')}."
        )

    return text
