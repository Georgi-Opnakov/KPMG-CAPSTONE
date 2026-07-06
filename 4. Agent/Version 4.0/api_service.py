from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any

import pandas as pd
import requests


AIRROI_BASE_URL = "https://api.airroi.com"
AIRROI_API_KEY_ENV = "AIRROI_API_KEY"


class ApiNotConfiguredError(RuntimeError):
    pass


class ApiRequestError(RuntimeError):
    pass


def _normalise_date(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return pd.to_datetime(value).date().isoformat()


def airroi_api_key() -> str | None:
    key = os.getenv(AIRROI_API_KEY_ENV)
    return key.strip() if key and key.strip() else None


def airroi_is_configured() -> bool:
    return airroi_api_key() is not None


def api_status() -> dict[str, Any]:
    return {
        "provider": "AirROI",
        "configured": airroi_is_configured(),
        "api_key_env": AIRROI_API_KEY_ENV,
        "base_url": AIRROI_BASE_URL,
        "future_rates_endpoint": "/listings/future/rates",
    }


def get_airroi_future_rates(listing_id: int, currency: str = "native", timeout: int = 20) -> dict[str, Any]:
    key = airroi_api_key()
    if not key:
        raise ApiNotConfiguredError(
            f"AirROI API key not configured. Set the {AIRROI_API_KEY_ENV} environment variable."
        )

    response = requests.get(
        f"{AIRROI_BASE_URL}/listings/future/rates",
        headers={"x-api-key": key},
        params={"id": int(listing_id), "currency": currency},
        timeout=timeout,
    )

    if response.status_code >= 400:
        raise ApiRequestError(f"AirROI request failed with HTTP {response.status_code}: {response.text[:500]}")

    return response.json()


def _extract_daily_rates(payload: dict[str, Any]) -> pd.DataFrame:
    """Handle likely future-rate response shapes without assuming one exact schema."""
    candidates = [
        payload.get("rates"),
        payload.get("future_rates"),
        payload.get("calendar"),
        payload.get("entries"),
        payload.get("data"),
    ]

    records = next((item for item in candidates if isinstance(item, list)), None)
    if records is None and isinstance(payload, list):
        records = payload
    if records is None:
        return pd.DataFrame()

    rates = pd.DataFrame(records)
    if rates.empty:
        return rates

    rename_map = {}
    for column in rates.columns:
        lowered = column.lower()
        if lowered in {"day", "date"}:
            rename_map[column] = "date"
        elif lowered in {"rate", "price", "nightly_rate", "daily_rate", "amount"}:
            rename_map[column] = "nightly_rate"
        elif lowered in {"available", "is_available", "availability"}:
            rename_map[column] = "available"
        elif lowered in {"minimum_nights", "min_nights", "min_stay"}:
            rename_map[column] = "minimum_nights"

    rates = rates.rename(columns=rename_map)
    if "date" in rates.columns:
        rates["date"] = pd.to_datetime(rates["date"], errors="coerce").dt.date.astype(str)
    if "nightly_rate" in rates.columns:
        rates["nightly_rate"] = pd.to_numeric(rates["nightly_rate"], errors="coerce")
    if "available" in rates.columns:
        rates["available"] = rates["available"].astype(str).str.lower().isin(["true", "t", "1", "yes", "available"])

    return rates


def summarize_airroi_stay(
    listing_id: int,
    check_in: str | date | datetime,
    check_out: str | date | datetime,
    currency: str = "native",
) -> dict[str, Any]:
    payload = get_airroi_future_rates(listing_id, currency=currency)
    rates = _extract_daily_rates(payload)

    start = _normalise_date(check_in)
    end = _normalise_date(check_out)
    nights = max((pd.to_datetime(end).date() - pd.to_datetime(start).date()).days, 0)

    if rates.empty or "date" not in rates.columns:
        return {
            "provider": "AirROI",
            "listing_id": int(listing_id),
            "check_in": start,
            "check_out": end,
            "nights": nights,
            "status": "AirROI returned a response, but daily rates could not be parsed.",
            "raw_payload": payload,
            "rates": rates,
        }

    stay_rates = rates[(rates["date"] >= start) & (rates["date"] < end)].copy()
    has_rate = "nightly_rate" in stay_rates.columns and stay_rates["nightly_rate"].notna().all()
    dates_available = True
    if "available" in stay_rates.columns:
        dates_available = bool(stay_rates["available"].all())

    total = float(stay_rates["nightly_rate"].sum()) if has_rate else None
    average = float(stay_rates["nightly_rate"].mean()) if has_rate else None
    complete_range = len(stay_rates) == nights

    if complete_range and dates_available and has_rate:
        status = "Live API returned date-specific rates and availability."
    elif complete_range and has_rate:
        status = "Live API returned date-specific rates, but at least one night may be unavailable."
    elif complete_range:
        status = "Live API returned dates, but no parseable nightly rates."
    else:
        status = "Live API did not return a complete date range for this stay."

    return {
        "provider": "AirROI",
        "listing_id": int(listing_id),
        "check_in": start,
        "check_out": end,
        "nights": nights,
        "complete_range": complete_range,
        "is_available": dates_available if complete_range else False,
        "has_live_price": has_rate and complete_range,
        "average_nightly_rate": round(average, 2) if average is not None else None,
        "total_rate": round(total, 2) if total is not None else None,
        "status": status,
        "rates": stay_rates,
    }
