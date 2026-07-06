from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from data_loader import CITY_FILE_KEYS, get_paths


HORIZON_MONTHS = 12
PRICE_BAND_WIDTH_EUR = 25

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)

DESCRIPTIVE_COLUMNS = [
    "listing_id",
    "city",
    "neighbourhood_cleansed",
    "property_group",
    "room_type",
    "accommodates",
    "bedrooms",
    "bathrooms",
    "distance_to_center_km",
    "review_scores_rating",
    "review_scores_value",
    "availability_30",
    "has_wifi",
    "has_kitchen",
    "has_air_conditioning",
    "has_washer",
    "has_elevator",
    "has_parking",
    "has_dedicated_workspace",
    "has_self_checkin",
    "price_eur",
]

SEASONAL_SIGNAL_COLUMNS = {
    "winter": "calendar_unavailable_winter",
    "spring": "calendar_unavailable_spring",
    "summer": "calendar_unavailable_summer",
    "autumn": "calendar_unavailable_autumn",
}


def prediction_cache_dir() -> Path:
    output_dir = get_paths()["outputs"] / "chatbot" / "prediction_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def model_dirs() -> tuple[Path, Path]:
    model_dir = get_paths()["outputs"] / "chatbot" / "models"
    return model_dir, model_dir / "property_groups"


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    manifest = pd.read_csv(path)
    for column in ["rmse_eur", "mae_eur", "r2"]:
        if column in manifest.columns:
            manifest[column] = pd.to_numeric(manifest[column], errors="coerce")
    return manifest


def load_master_city(city: str) -> pd.DataFrame:
    key = CITY_FILE_KEYS[city]
    path = get_paths()["master"] / f"{key}_master_model_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing master dataset for {city}: {path}")

    available_cols = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [column for column in DESCRIPTIVE_COLUMNS if column in available_cols]
    df = pd.read_csv(path, usecols=usecols)
    if "city" not in df.columns:
        df["city"] = city
    return df


def load_feature_lookup(path: str | Path, feature_columns: list[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction feature lookup: {path}")

    lookup = pd.read_csv(path)
    missing = [column for column in ["listing_id", *feature_columns] if column not in lookup.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return lookup


def to_price(log_values: np.ndarray | pd.Series) -> np.ndarray:
    return np.maximum(np.expm1(np.asarray(log_values, dtype=float)), 0)


def price_band(value: float | int | None, width: int = PRICE_BAND_WIDTH_EUR) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    low = int(np.floor(float(value) / width) * width)
    high = low + width
    return f"EUR {low}-{high}"


def predict_from_artifact(
    artifact: dict[str, Any],
    manifest_row: pd.Series,
    scope: str,
    fallback_segment: str | None = None,
) -> pd.DataFrame:
    feature_columns = artifact["feature_columns"]
    lookup = load_feature_lookup(artifact["listing_feature_file"], feature_columns)
    predictions = to_price(artifact["pipeline"].predict(lookup[feature_columns]))

    segment = artifact.get("segment_value") or fallback_segment or "All property groups"
    result = pd.DataFrame(
        {
            "listing_id": pd.to_numeric(lookup["listing_id"], errors="coerce").astype("Int64"),
            "predicted_nightly_price_eur": np.round(predictions, 2),
            "predicted_price_band": [price_band(value) for value in predictions],
            "prediction_model_scope": scope,
            "prediction_model_city": artifact.get("city"),
            "prediction_model_segment": segment,
            "prediction_model_name": artifact.get("model_name"),
            "prediction_rmse_eur": round(float(manifest_row.get("rmse_eur", np.nan)), 4),
            "prediction_mae_eur": round(float(manifest_row.get("mae_eur", np.nan)), 4),
            "prediction_r2": round(float(manifest_row.get("r2", np.nan)), 4),
        }
    )

    for column in SEASONAL_SIGNAL_COLUMNS.values():
        if column in lookup.columns:
            result[column] = pd.to_numeric(lookup[column], errors="coerce")

    return result


def build_property_group_predictions(property_manifest: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, row in property_manifest.iterrows():
        artifact = joblib.load(row["artifact"])
        frames.append(predict_from_artifact(artifact, row, scope="property_group"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_city_fallback_predictions(
    city_manifest: pd.DataFrame,
    already_predicted_ids: dict[str, set[int]],
) -> pd.DataFrame:
    frames = []
    for _, row in city_manifest.iterrows():
        city = str(row["city"])
        artifact = joblib.load(row["artifact"])
        predictions = predict_from_artifact(
            artifact,
            row,
            scope="city_fallback",
            fallback_segment="All property groups",
        )

        used_ids = already_predicted_ids.get(city, set())
        predictions = predictions[~predictions["listing_id"].astype(int).isin(used_ids)].copy()
        if not predictions.empty:
            frames.append(predictions)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_master_context(predictions: pd.DataFrame) -> pd.DataFrame:
    master_frames = [load_master_city(city) for city in CITY_FILE_KEYS]
    master = pd.concat(master_frames, ignore_index=True)
    master["listing_id"] = pd.to_numeric(master["listing_id"], errors="coerce").astype("Int64")

    merged = master.merge(predictions, on="listing_id", how="inner")
    merged = merged.rename(columns={"price_eur": "snapshot_price_eur"})
    merged["prediction_generated_at"] = datetime.now().isoformat(timespec="seconds")

    ordered = [
        "listing_id",
        "city",
        "neighbourhood_cleansed",
        "property_group",
        "room_type",
        "accommodates",
        "bedrooms",
        "bathrooms",
        "distance_to_center_km",
        "review_scores_rating",
        "review_scores_value",
        "availability_30",
        "has_wifi",
        "has_kitchen",
        "has_air_conditioning",
        "has_washer",
        "has_elevator",
        "has_parking",
        "has_dedicated_workspace",
        "has_self_checkin",
        "snapshot_price_eur",
        "predicted_nightly_price_eur",
        "predicted_price_band",
        "prediction_model_scope",
        "prediction_model_segment",
        "prediction_model_name",
        "prediction_rmse_eur",
        "prediction_mae_eur",
        "prediction_r2",
        *SEASONAL_SIGNAL_COLUMNS.values(),
        "prediction_generated_at",
    ]
    available = [column for column in ordered if column in merged.columns]
    return merged[available].sort_values(["city", "listing_id"]).reset_index(drop=True)


def month_season(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def availability_label(unavailable_signal: float | int | None) -> str:
    if unavailable_signal is None or pd.isna(unavailable_signal):
        return "unknown"
    signal = float(unavailable_signal)
    if signal <= 0.25:
        return "strong availability signal"
    if signal <= 0.50:
        return "moderate availability signal"
    if signal <= 0.75:
        return "limited availability signal"
    return "very limited availability signal"


def month_starts(horizon_months: int = HORIZON_MONTHS) -> pd.DatetimeIndex:
    current_month = pd.Timestamp.today().normalize().replace(day=1)
    first_future_month = current_month + pd.offsets.MonthBegin(1)
    return pd.date_range(first_future_month, periods=horizon_months, freq="MS")


def build_monthly_cache(listing_cache: pd.DataFrame, horizon_months: int = HORIZON_MONTHS) -> pd.DataFrame:
    frames = []
    base_cols = [
        "listing_id",
        "city",
        "predicted_nightly_price_eur",
        "predicted_price_band",
        "prediction_model_scope",
        "prediction_model_segment",
        "prediction_model_name",
    ]
    base_cols = [column for column in base_cols if column in listing_cache.columns]

    for month_start in month_starts(horizon_months):
        season = month_season(month_start.month)
        signal_col = SEASONAL_SIGNAL_COLUMNS[season]

        monthly = listing_cache[base_cols].copy()
        monthly["prediction_month"] = month_start.strftime("%Y-%m")
        monthly["month_start"] = month_start.date().isoformat()
        monthly["season"] = season
        monthly["month_unavailable_signal"] = pd.to_numeric(
            listing_cache.get(signal_col, pd.Series(np.nan, index=listing_cache.index)),
            errors="coerce",
        )
        monthly["month_availability_label"] = [
            availability_label(value) for value in monthly["month_unavailable_signal"]
        ]
        frames.append(monthly)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    first_cols = ["prediction_month", "month_start", "season", "city", "listing_id"]
    ordered = [column for column in first_cols if column in result.columns]
    ordered += [column for column in result.columns if column not in ordered]
    return result[ordered]


def build_prediction_cache(horizon_months: int = HORIZON_MONTHS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_dir, property_model_dir = model_dirs()
    property_manifest = read_manifest(property_model_dir / "property_group_price_model_manifest.csv")
    city_manifest = read_manifest(model_dir / "price_model_manifest.csv")

    if property_manifest.empty and city_manifest.empty:
        raise FileNotFoundError("No reusable price model artifacts were found.")

    property_predictions = build_property_group_predictions(property_manifest)
    predicted_ids: dict[str, set[int]] = {}
    if not property_predictions.empty:
        for city, group in property_predictions.groupby("prediction_model_city"):
            predicted_ids[str(city)] = set(group["listing_id"].dropna().astype(int))

    fallback_predictions = build_city_fallback_predictions(city_manifest, predicted_ids)
    predictions = pd.concat(
        [frame for frame in [property_predictions, fallback_predictions] if not frame.empty],
        ignore_index=True,
    )
    if predictions.empty:
        raise ValueError("The model artifacts loaded, but no listing predictions were produced.")

    listing_cache = add_master_context(predictions)
    monthly_cache = build_monthly_cache(listing_cache, horizon_months=horizon_months)

    manifest = pd.DataFrame(
        [
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "horizon_months": horizon_months,
                "listing_cache_rows": len(listing_cache),
                "monthly_cache_rows": len(monthly_cache),
                "property_group_artifacts": len(property_manifest),
                "city_fallback_artifacts": len(city_manifest),
                "property_group_prediction_rows": len(property_predictions),
                "city_fallback_prediction_rows": len(fallback_predictions),
                "listing_cache_file": str(prediction_cache_dir() / "listing_price_prediction_cache.csv"),
                "monthly_cache_file": str(prediction_cache_dir() / "monthly_price_prediction_cache.csv"),
                "notes": (
                    "Property-group models are used where available. City-level XGBoost artifacts fill any "
                    "remaining listings. Monthly rows reuse the static model price and add seasonal calendar "
                    "availability signals; they are not live Airbnb quotes."
                ),
            }
        ]
    )
    return listing_cache, monthly_cache, manifest


def main() -> None:
    output_dir = prediction_cache_dir()
    listing_cache, monthly_cache, manifest = build_prediction_cache()

    listing_path = output_dir / "listing_price_prediction_cache.csv"
    monthly_path = output_dir / "monthly_price_prediction_cache.csv"
    manifest_path = output_dir / "prediction_cache_manifest.csv"

    listing_cache.to_csv(listing_path, index=False)
    monthly_cache.to_csv(monthly_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    print(f"Saved listing prediction cache: {listing_path}")
    print(f"Saved monthly prediction cache: {monthly_path}")
    print(f"Saved prediction cache manifest: {manifest_path}")
    print(manifest.to_string(index=False))
    print()
    print(
        listing_cache[
            [
                "city",
                "prediction_model_scope",
                "prediction_model_segment",
                "prediction_model_name",
                "predicted_nightly_price_eur",
            ]
        ]
        .groupby(["city", "prediction_model_scope", "prediction_model_segment", "prediction_model_name"])
        .agg(rows=("predicted_nightly_price_eur", "size"), median_predicted_price=("predicted_nightly_price_eur", "median"))
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
