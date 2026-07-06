from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from data_loader import get_paths


MONTH_NAME_TO_NUMBER = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

SEASON_TO_MONTHS = {
    "Winter": "December to February",
    "Spring": "March to May",
    "Summer": "June to August",
    "Autumn": "September to November",
}

MONTH_TO_SEASON = {
    "January": "Winter",
    "February": "Winter",
    "March": "Spring",
    "April": "Spring",
    "May": "Spring",
    "June": "Summer",
    "July": "Summer",
    "August": "Summer",
    "September": "Autumn",
    "October": "Autumn",
    "November": "Autumn",
    "December": "Winter",
}


def prediction_cache_dir() -> Path:
    return get_paths()["outputs"] / "chatbot" / "prediction_cache"


def listing_cache_path() -> Path:
    return prediction_cache_dir() / "listing_price_prediction_cache.csv"


def monthly_cache_path() -> Path:
    return prediction_cache_dir() / "monthly_price_prediction_cache.csv"


def prediction_cache_available() -> bool:
    return listing_cache_path().exists() and monthly_cache_path().exists()


@lru_cache(maxsize=1)
def load_listing_prediction_cache() -> pd.DataFrame:
    path = listing_cache_path()
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "listing_id" in df.columns:
        df["listing_id"] = pd.to_numeric(df["listing_id"], errors="coerce").astype("Int64")
    return df


@lru_cache(maxsize=1)
def load_monthly_prediction_cache() -> pd.DataFrame:
    path = monthly_cache_path()
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "listing_id" in df.columns:
        df["listing_id"] = pd.to_numeric(df["listing_id"], errors="coerce").astype("Int64")
    if "month_start" in df.columns:
        df["month_start"] = pd.to_datetime(df["month_start"], errors="coerce")
    return df


def enrich_with_prediction_cache(df: pd.DataFrame, city: str) -> pd.DataFrame:
    cache = load_listing_prediction_cache()
    if cache.empty or df.empty or "listing_id" not in df.columns:
        return df

    cache_cols = [
        "listing_id",
        "predicted_nightly_price_eur",
        "predicted_price_band",
        "prediction_model_scope",
        "prediction_model_segment",
        "prediction_model_name",
        "prediction_rmse_eur",
        "prediction_mae_eur",
        "prediction_r2",
    ]
    cache_cols = [column for column in cache_cols if column in cache.columns]
    city_cache = cache[cache["city"].astype(str) == city][cache_cols].copy()

    enriched = df.copy()
    enriched["listing_id"] = pd.to_numeric(enriched["listing_id"], errors="coerce").astype("Int64")
    duplicate_cols = [column for column in cache_cols if column != "listing_id" and column in enriched.columns]
    if duplicate_cols:
        enriched = enriched.drop(columns=duplicate_cols)
    return enriched.merge(city_cache, on="listing_id", how="left")


def _normalise_seasons(seasons: list[str] | None) -> list[str]:
    if not seasons:
        return []
    labels = []
    for season in seasons:
        clean = str(season).strip().title()
        if clean == "Fall":
            clean = "Autumn"
        if clean in SEASON_TO_MONTHS:
            labels.append(clean)
    return list(dict.fromkeys(labels))


def _normalise_months(months: list[str] | None) -> list[str]:
    if not months:
        return []
    labels = []
    for month in months:
        clean = str(month).strip().title()
        if clean in MONTH_NAME_TO_NUMBER:
            labels.append(clean)
    return list(dict.fromkeys(labels))


def prediction_window_summary(
    city: str,
    listing_ids: pd.Series | list[int],
    *,
    months: list[str] | None = None,
    seasons: list[str] | None = None,
) -> pd.DataFrame:
    monthly = load_monthly_prediction_cache()
    if monthly.empty:
        return pd.DataFrame()

    ids = pd.Series(listing_ids).dropna()
    if ids.empty:
        return pd.DataFrame()

    ids = set(pd.to_numeric(ids, errors="coerce").dropna().astype("int64").tolist())
    window = monthly[
        (monthly["city"].astype(str) == city)
        & (monthly["listing_id"].astype("int64").isin(ids))
    ].copy()
    if window.empty:
        return pd.DataFrame()

    month_labels = _normalise_months(months)
    season_labels = _normalise_seasons(seasons)

    if month_labels:
        month_numbers = [MONTH_NAME_TO_NUMBER[month] for month in month_labels]
        month_window = window[window["month_start"].dt.month.isin(month_numbers)].copy()
        month_window["_window_label"] = month_window["month_start"].dt.month.map(
            {number: month for month, number in MONTH_NAME_TO_NUMBER.items()}
        )
        month_seasons = {MONTH_TO_SEASON[month] for month in month_labels}
        season_labels = [season for season in season_labels if season not in month_seasons]
        if season_labels:
            season_window = window[window["season"].str.title().isin(season_labels)].copy()
            season_window["_window_label"] = season_window["season"].str.title()
            window = pd.concat([month_window, season_window], ignore_index=True)
        else:
            window = month_window
        window_order = [*month_labels, *season_labels]
    else:
        if season_labels:
            window = window[window["season"].str.title().isin(season_labels)].copy()
            window_order = season_labels
        else:
            window_order = list(SEASON_TO_MONTHS)
        window["_window_label"] = window["season"].str.title()

    if window.empty:
        return pd.DataFrame()

    summary = (
        window.groupby("_window_label", as_index=False)
        .agg(
            count=("listing_id", "nunique"),
            mean_price=("predicted_nightly_price_eur", "mean"),
            median_price=("predicted_nightly_price_eur", "median"),
            mean_unavailable=("month_unavailable_signal", "mean"),
        )
        .rename(columns={"_window_label": "window"})
    )

    month_lookup = {month: month for month in MONTH_NAME_TO_NUMBER}
    months_lookup = {**SEASON_TO_MONTHS, **month_lookup}
    summary["months"] = summary["window"].map(months_lookup).fillna(summary["window"])
    summary["price_source"] = "precomputed model prediction cache"
    order = {label: idx for idx, label in enumerate(window_order)}
    summary["_order"] = summary["window"].map(order).fillna(999)
    return summary.sort_values(["mean_price", "_order"]).drop(columns="_order").reset_index(drop=True)
