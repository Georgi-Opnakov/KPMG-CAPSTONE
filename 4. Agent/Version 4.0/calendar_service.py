from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from data_loader import CITY_FILE_KEYS, get_paths


CALENDAR_COLUMNS = [
    "listing_id",
    "date",
    "available",
    "price",
    "adjusted_price",
    "minimum_nights",
    "maximum_nights",
]


def calendar_cache_path() -> Path:
    output_dir = get_paths()["outputs"] / "chatbot"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "calendar_snapshot.sqlite"


def calendar_cache_exists() -> bool:
    return calendar_cache_path().exists()


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(calendar_cache_path())


def _normalise_date(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return pd.to_datetime(value).date().isoformat()


def _nights(check_in: str | date | datetime, check_out: str | date | datetime) -> int:
    start = pd.to_datetime(check_in).date()
    end = pd.to_datetime(check_out).date()
    return max((end - start).days, 0)


def initialise_calendar_db(force: bool = False) -> Path:
    """Build a compact SQLite cache of available calendar dates from raw calendars."""
    db_path = calendar_cache_path()
    if db_path.exists() and not force:
        return db_path

    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute(
            """
            CREATE TABLE calendar_available (
                city TEXT NOT NULL,
                listing_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                minimum_nights INTEGER,
                maximum_nights INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE calendar_meta (
                city TEXT PRIMARY KEY,
                source_file TEXT NOT NULL,
                min_date TEXT,
                max_date TEXT,
                total_rows INTEGER,
                available_rows INTEGER,
                price_rows INTEGER,
                adjusted_price_rows INTEGER,
                built_at TEXT NOT NULL
            )
            """
        )
        conn.commit()

        raw_dir = get_paths()["raw"]
        for city, key in CITY_FILE_KEYS.items():
            source_file = raw_dir / f"{key}_calendar.csv"
            if not source_file.exists():
                continue

            total_rows = 0
            available_rows = 0
            price_rows = 0
            adjusted_price_rows = 0
            min_date: str | None = None
            max_date: str | None = None

            for chunk in pd.read_csv(source_file, usecols=CALENDAR_COLUMNS, chunksize=500_000):
                total_rows += len(chunk)
                price_rows += int(chunk["price"].notna().sum())
                adjusted_price_rows += int(chunk["adjusted_price"].notna().sum())

                chunk_min = str(chunk["date"].min())
                chunk_max = str(chunk["date"].max())
                min_date = chunk_min if min_date is None or chunk_min < min_date else min_date
                max_date = chunk_max if max_date is None or chunk_max > max_date else max_date

                available = chunk.loc[
                    chunk["available"].astype(str).str.lower().eq("t"),
                    ["listing_id", "date", "minimum_nights", "maximum_nights"],
                ].copy()
                if available.empty:
                    continue

                available.insert(0, "city", city)
                available_rows += len(available)
                available.to_sql("calendar_available", conn, if_exists="append", index=False)

            conn.execute(
                """
                INSERT INTO calendar_meta (
                    city,
                    source_file,
                    min_date,
                    max_date,
                    total_rows,
                    available_rows,
                    price_rows,
                    adjusted_price_rows,
                    built_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    city,
                    str(source_file),
                    min_date,
                    max_date,
                    total_rows,
                    available_rows,
                    price_rows,
                    adjusted_price_rows,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            conn.commit()

        conn.execute("CREATE INDEX idx_calendar_city_listing_date ON calendar_available(city, listing_id, date)")
        conn.execute("CREATE INDEX idx_calendar_city_date_listing ON calendar_available(city, date, listing_id)")
        conn.commit()
    finally:
        conn.close()

    return db_path


def calendar_metadata() -> pd.DataFrame:
    if not calendar_cache_exists():
        return pd.DataFrame()

    with _connect() as conn:
        return pd.read_sql_query("SELECT * FROM calendar_meta ORDER BY city", conn)


def date_bounds(city: str | None = None) -> tuple[date | None, date | None]:
    metadata = calendar_metadata()
    if metadata.empty:
        return None, None

    if city and city in set(metadata["city"]):
        metadata = metadata[metadata["city"] == city]

    min_date = pd.to_datetime(metadata["min_date"].min()).date()
    max_date = pd.to_datetime(metadata["max_date"].max()).date()
    return min_date, max_date


def check_listing_availability(
    city: str,
    listing_id: int,
    check_in: str | date | datetime,
    check_out: str | date | datetime,
) -> dict:
    nights = _nights(check_in, check_out)
    if nights <= 0:
        return {
            "city": city,
            "listing_id": listing_id,
            "nights": nights,
            "available_nights": 0,
            "is_available": False,
            "stay_length_ok": False,
            "status": "Check-out must be after check-in.",
        }

    if not calendar_cache_exists():
        return {
            "city": city,
            "listing_id": listing_id,
            "nights": nights,
            "available_nights": 0,
            "is_available": False,
            "stay_length_ok": False,
            "status": "Calendar cache has not been built yet.",
        }

    start = _normalise_date(check_in)
    end = _normalise_date(check_out)
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(DISTINCT date) AS available_nights,
                MAX(minimum_nights) AS minimum_nights_required,
                MIN(maximum_nights) AS maximum_nights_allowed
            FROM calendar_available
            WHERE city = ?
              AND listing_id = ?
              AND date >= ?
              AND date < ?
            """,
            (city, int(listing_id), start, end),
        ).fetchone()

    available_nights = int(row[0] or 0)
    minimum_nights_required = int(row[1] or 0)
    maximum_nights_allowed = int(row[2] or 0)
    dates_available = available_nights >= nights
    minimum_ok = minimum_nights_required == 0 or nights >= minimum_nights_required
    maximum_ok = maximum_nights_allowed == 0 or nights <= maximum_nights_allowed
    stay_length_ok = minimum_ok and maximum_ok
    is_available = dates_available and stay_length_ok

    if is_available:
        status = "Available in the local calendar snapshot."
    elif not dates_available:
        status = f"Only {available_nights} of {nights} nights are available in the local calendar snapshot."
    elif not minimum_ok:
        status = f"Stay is shorter than the listing minimum of {minimum_nights_required} nights."
    else:
        status = f"Stay is longer than the listing maximum of {maximum_nights_allowed} nights."

    return {
        "city": city,
        "listing_id": int(listing_id),
        "check_in": start,
        "check_out": end,
        "nights": nights,
        "available_nights": available_nights,
        "minimum_nights_required": minimum_nights_required,
        "maximum_nights_allowed": maximum_nights_allowed,
        "is_available": is_available,
        "stay_length_ok": stay_length_ok,
        "has_calendar_price": False,
        "status": status,
    }


def availability_for_listings(
    city: str,
    listing_ids: Iterable[int],
    check_in: str | date | datetime,
    check_out: str | date | datetime,
) -> pd.DataFrame:
    ids = [int(listing_id) for listing_id in listing_ids if pd.notna(listing_id)]
    ids = list(dict.fromkeys(ids))
    nights = _nights(check_in, check_out)

    if not ids or nights <= 0 or not calendar_cache_exists():
        return pd.DataFrame(
            columns=[
                "listing_id",
                "available_nights",
                "minimum_nights_required",
                "maximum_nights_allowed",
                "is_available",
                "calendar_status",
            ]
        )

    start = _normalise_date(check_in)
    end = _normalise_date(check_out)
    frames = []

    with _connect() as conn:
        for offset in range(0, len(ids), 800):
            batch = ids[offset : offset + 800]
            placeholders = ",".join("?" for _ in batch)
            query = f"""
                SELECT
                    listing_id,
                    COUNT(DISTINCT date) AS available_nights,
                    MAX(minimum_nights) AS minimum_nights_required,
                    MIN(maximum_nights) AS maximum_nights_allowed
                FROM calendar_available
                WHERE city = ?
                  AND date >= ?
                  AND date < ?
                  AND listing_id IN ({placeholders})
                GROUP BY listing_id
            """
            frames.append(pd.read_sql_query(query, conn, params=[city, start, end, *batch]))

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        result = pd.DataFrame({"listing_id": ids})

    result = pd.DataFrame({"listing_id": ids}).merge(result, on="listing_id", how="left")
    result["available_nights"] = result["available_nights"].fillna(0).astype(int)
    result["minimum_nights_required"] = result["minimum_nights_required"].fillna(0).astype(int)
    result["maximum_nights_allowed"] = result["maximum_nights_allowed"].fillna(0).astype(int)

    dates_available = result["available_nights"] >= nights
    minimum_ok = (result["minimum_nights_required"] == 0) | (nights >= result["minimum_nights_required"])
    maximum_ok = (result["maximum_nights_allowed"] == 0) | (nights <= result["maximum_nights_allowed"])
    result["is_available"] = dates_available & minimum_ok & maximum_ok
    result["calendar_status"] = np_status(result, nights)
    return result


def np_status(result: pd.DataFrame, nights: int) -> list[str]:
    statuses = []
    for _, row in result.iterrows():
        if bool(row["is_available"]):
            statuses.append("Available in local calendar snapshot")
        elif int(row["available_nights"]) < nights:
            statuses.append(f"Only {int(row['available_nights'])} of {nights} nights available")
        elif int(row["minimum_nights_required"]) and nights < int(row["minimum_nights_required"]):
            statuses.append(f"Minimum stay is {int(row['minimum_nights_required'])} nights")
        elif int(row["maximum_nights_allowed"]) and nights > int(row["maximum_nights_allowed"]):
            statuses.append(f"Maximum stay is {int(row['maximum_nights_allowed'])} nights")
        else:
            statuses.append("Unavailable for requested stay")
    return statuses
