from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


AMENITY_COLUMNS = {
    "WiFi": "has_wifi",
    "Kitchen": "has_kitchen",
    "Air conditioning": "has_air_conditioning",
    "Washer": "has_washer",
    "Dedicated workspace": "has_dedicated_workspace",
    "TV": "has_tv",
    "Parking": "has_parking",
    "Elevator": "has_elevator",
    "Heating": "has_heating",
    "Self check-in": "has_self_checkin",
}

LUXURY_AMENITY_COLUMNS = [
    "has_air_conditioning",
    "has_washer",
    "has_dedicated_workspace",
    "has_tv",
    "has_parking",
    "has_elevator",
    "has_heating",
    "has_self_checkin",
]

DISPLAY_COLUMNS = [
    "listing_id",
    "city",
    "neighbourhood_cleansed",
    "room_type",
    "property_group",
    "price_eur",
    "accommodates",
    "bedrooms",
    "bathrooms",
    "review_scores_rating",
    "review_scores_value",
    "number_of_reviews",
    "availability_30",
    "distance_to_center_km",
    "amenities_count",
    "value_score",
    "recommendation_reason",
]


def format_eur(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"EUR {float(value):,.0f}"


def _existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _rank_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.5, index=series.index)

    values = values.fillna(values.median())
    ranks = values.rank(pct=True, method="average")
    return ranks if higher_is_better else 1 - ranks


def _luxury_amenity_count(df: pd.DataFrame) -> pd.Series:
    columns = [column for column in LUXURY_AMENITY_COLUMNS if column in df.columns]
    if not columns:
        return pd.Series(0, index=df.index)

    luxury_flags = df[columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    return luxury_flags.sum(axis=1)


def add_value_score(df: pd.DataFrame) -> pd.DataFrame:
    """Create a simple explainable value score for holiday recommendations."""
    if df.empty:
        return df.copy()

    scored = df.copy()
    price_score = _rank_score(_numeric(scored, "price_eur"), higher_is_better=False)
    rating_score = _rank_score(_numeric(scored, "review_scores_rating", 4.0), higher_is_better=True)
    value_rating_score = _rank_score(_numeric(scored, "review_scores_value", 4.0), higher_is_better=True)
    location_score = _rank_score(_numeric(scored, "distance_to_center_km"), higher_is_better=False)
    amenity_score = _rank_score(_numeric(scored, "amenities_count"), higher_is_better=True)
    availability_score = _rank_score(_numeric(scored, "availability_30"), higher_is_better=True)

    scored["value_score"] = (
        0.35 * price_score
        + 0.20 * rating_score
        + 0.15 * value_rating_score
        + 0.15 * location_score
        + 0.10 * amenity_score
        + 0.05 * availability_score
    )
    scored["value_score"] = (scored["value_score"] * 100).round(1)
    return scored


def build_recommendation_reason(row: pd.Series) -> str:
    reasons: list[str] = []

    if pd.notna(row.get("price_eur")):
        reasons.append(f"{format_eur(row['price_eur'])} nightly price")
    rating = row.get("review_scores_rating")
    review_count = row.get("number_of_reviews")
    if pd.notna(rating) and pd.notna(review_count):
        reasons.append(f"{float(rating):.2f} rating from {int(review_count)} reviews")
    elif pd.notna(rating):
        reasons.append(f"{float(rating):.2f} rating")
    if pd.notna(row.get("distance_to_center_km")):
        reasons.append(f"{row['distance_to_center_km']:.1f} km from centre")
    if pd.notna(row.get("availability_30")):
        reasons.append(f"{int(row['availability_30'])} available days in next 30")

    return "; ".join(reasons[:4])


def _recommendation_style_sort(scored: pd.DataFrame, style: str, limit: int) -> pd.DataFrame:
    """Rank recommendations for a simple traveler intent: Budget, Value, or Luxury.

    The K-means segmentation work showed clear Value, Standard, Premium, and
    Luxury listing profiles. This helper applies those ideas directly to the
    current filtered listing pool without relying on a fragile row-order join.
    """
    if scored.empty:
        return scored

    style_key = str(style or "Value").strip().lower()
    ranked = scored.copy()

    price_score = _rank_score(_numeric(ranked, "price_eur"), higher_is_better=False)
    premium_price_score = _rank_score(_numeric(ranked, "price_eur"), higher_is_better=True)
    rating_score = _rank_score(_numeric(ranked, "review_scores_rating", 4.0), higher_is_better=True)
    value_rating_score = _rank_score(_numeric(ranked, "review_scores_value", 4.0), higher_is_better=True)
    location_score = _rank_score(_numeric(ranked, "distance_to_center_km"), higher_is_better=False)
    amenity_score = _rank_score(_numeric(ranked, "amenities_count"), higher_is_better=True)
    availability_score = _rank_score(_numeric(ranked, "availability_30"), higher_is_better=True)

    if style_key == "budget":
        prices = _numeric(ranked, "price_eur")
        cutoff = prices.quantile(0.65) if prices.notna().sum() >= max(limit, 8) else None
        if cutoff is not None and pd.notna(cutoff):
            narrowed = ranked[prices <= cutoff].copy()
            if len(narrowed) >= min(limit, 3):
                ranked = narrowed
                price_score = price_score.loc[ranked.index]
                rating_score = rating_score.loc[ranked.index]
                value_rating_score = value_rating_score.loc[ranked.index]
                location_score = location_score.loc[ranked.index]
                availability_score = availability_score.loc[ranked.index]

        ranked["recommendation_style_score"] = (
            0.45 * price_score
            + 0.20 * rating_score
            + 0.15 * location_score
            + 0.10 * value_rating_score
            + 0.10 * availability_score
        )
        return ranked.sort_values(
            ["recommendation_style_score", "value_score", "review_scores_rating", "price_eur"],
            ascending=[False, False, False, True],
        )

    if style_key == "value":
        ranked["recommendation_style_score"] = (
            0.30 * value_rating_score
            + 0.22 * rating_score
            + 0.18 * amenity_score
            + 0.15 * location_score
            + 0.10 * availability_score
            + 0.05 * premium_price_score
        )
        return ranked.sort_values(
            ["recommendation_style_score", "review_scores_value", "review_scores_rating", "price_eur"],
            ascending=[False, False, False, True],
        )

    if style_key == "luxury":
        if "room_type" in ranked.columns:
            entire_home = ranked["room_type"].astype(str).str.lower().eq("entire home/apt")
            narrowed = ranked[entire_home].copy()
            if len(narrowed) >= min(limit, 3):
                ranked = narrowed

        ranked["recommendation_style_score"] = (
            0.20 * _rank_score(_numeric(ranked, "review_scores_rating", 4.0), higher_is_better=True)
            + 0.16 * _rank_score(_numeric(ranked, "review_scores_value", 4.0), higher_is_better=True)
            + 0.18 * _rank_score(_luxury_amenity_count(ranked), higher_is_better=True)
            + 0.12 * _rank_score(_numeric(ranked, "amenities_count"), higher_is_better=True)
            + 0.14 * _rank_score(_numeric(ranked, "accommodates"), higher_is_better=True)
            + 0.08 * _rank_score(_numeric(ranked, "bedrooms"), higher_is_better=True)
            + 0.04 * _rank_score(_numeric(ranked, "availability_30"), higher_is_better=True)
            + 0.08 * _rank_score(_numeric(ranked, "price_eur"), higher_is_better=True)
        )
        return ranked.sort_values(
            ["recommendation_style_score", "review_scores_rating", "amenities_count", "price_eur"],
            ascending=[False, False, False, False],
        )

    ranked["recommendation_style_score"] = ranked["value_score"]
    return ranked.sort_values(["value_score", "review_scores_rating", "price_eur"], ascending=[False, False, True])


def filter_listings(
    df: pd.DataFrame,
    budget_max: float | None = None,
    min_price_eur: float | None = None,
    guests: int | None = None,
    room_type: str | None = None,
    neighbourhood: str | None = None,
    amenities: list[str] | None = None,
    min_rating: float | None = None,
    max_distance_km: float | None = None,
    min_available_days_30: int | None = None,
    min_reviews: int | None = None,
    min_amenities_count: int | None = None,
    min_luxury_amenities: int | None = None,
) -> pd.DataFrame:
    filtered = df.copy()

    if budget_max and "price_eur" in filtered.columns:
        filtered = filtered[_numeric(filtered, "price_eur") <= budget_max]

    if min_price_eur and "price_eur" in filtered.columns:
        filtered = filtered[_numeric(filtered, "price_eur") >= min_price_eur]

    if guests and "accommodates" in filtered.columns:
        filtered = filtered[_numeric(filtered, "accommodates") >= guests]

    if room_type and room_type != "Any" and "room_type" in filtered.columns:
        filtered = filtered[filtered["room_type"].astype(str) == room_type]

    if neighbourhood and neighbourhood != "Any" and "neighbourhood_cleansed" in filtered.columns:
        filtered = filtered[filtered["neighbourhood_cleansed"].astype(str) == neighbourhood]

    if min_rating and "review_scores_rating" in filtered.columns:
        rating = _numeric(filtered, "review_scores_rating")
        filtered = filtered[(rating >= min_rating) | rating.isna()]

    if max_distance_km and "distance_to_center_km" in filtered.columns:
        filtered = filtered[_numeric(filtered, "distance_to_center_km") <= max_distance_km]

    if min_available_days_30 and "availability_30" in filtered.columns:
        filtered = filtered[_numeric(filtered, "availability_30") >= min_available_days_30]

    if min_reviews and "number_of_reviews" in filtered.columns:
        filtered = filtered[_numeric(filtered, "number_of_reviews") >= min_reviews]

    if min_amenities_count and "amenities_count" in filtered.columns:
        filtered = filtered[_numeric(filtered, "amenities_count") >= min_amenities_count]

    if min_luxury_amenities:
        filtered = filtered[_luxury_amenity_count(filtered) >= min_luxury_amenities]

    for amenity in amenities or []:
        column = AMENITY_COLUMNS.get(amenity)
        if column in filtered.columns:
            filtered = filtered[_numeric(filtered, column) == 1]

    return filtered


def get_recommendations(
    df: pd.DataFrame,
    limit: int = 10,
    **filters,
) -> pd.DataFrame:
    recommendation_style = filters.pop("recommendation_style", "Value")
    filtered = filter_listings(df, **filters)
    scored = add_value_score(filtered)

    if scored.empty:
        return scored

    scored["recommendation_reason"] = scored.apply(build_recommendation_reason, axis=1)
    scored = _recommendation_style_sort(scored, recommendation_style, limit=limit)
    columns = _existing_columns(scored, DISPLAY_COLUMNS)
    return scored.head(limit)[columns]


def recommendation_price_floor(
    df: pd.DataFrame,
    filters: dict,
    quantile: float = 0.20,
    min_candidates: int = 30,
) -> tuple[float | None, int]:
    """Return a data-driven floor that avoids promoting very cheap outliers.

    The floor is calculated from the already-filtered candidate pool. It keeps
    value recommendations affordable, but stops the shortlist being dominated
    by the lowest-price tail of the data.
    """
    if "price_eur" not in df.columns:
        return None, 0

    candidate_filters = {key: value for key, value in filters.items() if key != "min_price_eur"}
    candidates = filter_listings(df, **candidate_filters)
    prices = pd.to_numeric(candidates.get("price_eur"), errors="coerce").dropna()
    if len(prices) < min_candidates:
        return None, len(prices)

    floor = float(prices.quantile(quantile))
    if not pd.notna(floor) or floor <= 0:
        return None, len(prices)

    return floor, len(prices)


def market_summary(df: pd.DataFrame) -> dict[str, float | int | str]:
    if df.empty:
        return {"listings": 0}

    price = _numeric(df, "price_eur")
    rating = _numeric(df, "review_scores_rating")
    availability = _numeric(df, "availability_30")
    distance = _numeric(df, "distance_to_center_km")

    return {
        "listings": int(len(df)),
        "median_price_eur": round(float(price.median()), 2),
        "mean_price_eur": round(float(price.mean()), 2),
        "mean_rating": round(float(rating.mean()), 2),
        "median_availability_30": round(float(availability.median()), 1),
        "median_distance_to_center_km": round(float(distance.median()), 2),
    }


def neighbourhood_summary(df: pd.DataFrame, min_rows: int = 25) -> pd.DataFrame:
    if df.empty or "neighbourhood_cleansed" not in df.columns:
        return pd.DataFrame()

    grouped = (
        df.groupby("neighbourhood_cleansed", dropna=False)
        .agg(
            listings=("price_eur", "size"),
            median_price_eur=("price_eur", "median"),
            mean_price_eur=("price_eur", "mean"),
            mean_rating=("review_scores_rating", "mean"),
            mean_value_rating=("review_scores_value", "mean"),
            median_distance_km=("distance_to_center_km", "median"),
            median_availability_30=("availability_30", "median"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["listings"] >= min_rows].copy()

    for column in [
        "median_price_eur",
        "mean_price_eur",
        "mean_rating",
        "mean_value_rating",
        "median_distance_km",
        "median_availability_30",
    ]:
        grouped[column] = pd.to_numeric(grouped[column], errors="coerce").round(2)

    grouped["neighbourhood_value_score"] = (
        0.45 * _rank_score(grouped["median_price_eur"], higher_is_better=False)
        + 0.30 * _rank_score(grouped["mean_rating"], higher_is_better=True)
        + 0.15 * _rank_score(grouped["mean_value_rating"], higher_is_better=True)
        + 0.10 * _rank_score(grouped["median_distance_km"], higher_is_better=False)
    )
    grouped["neighbourhood_value_score"] = (grouped["neighbourhood_value_score"] * 100).round(1)
    return grouped.sort_values("neighbourhood_value_score", ascending=False)


def compare_cities(master_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for city, df in master_data.items():
        if df.empty:
            continue
        summary = market_summary(df)
        rows.append({"city": city, **summary})
    return pd.DataFrame(rows)


def answer_from_data(question: str, master_data: dict[str, pd.DataFrame], city: str | None = None) -> str:
    """Rule-based first response layer. RAG/LLM can be added behind this later."""
    q = question.lower()
    selected_cities = [city] if city in master_data else list(master_data.keys())

    if any(word in q for word in ["neighbourhood", "neighborhood", "area", "district"]):
        lines = ["Top neighbourhoods by value score:"]
        for city_name in selected_cities:
            areas = neighbourhood_summary(master_data[city_name]).head(5)
            lines.append(f"{city_name}:")
            for _, row in areas.iterrows():
                lines.append(
                    f"- {row['neighbourhood_cleansed']}: median {format_eur(row['median_price_eur'])}, "
                    f"rating {row['mean_rating']:.2f}, score {row['neighbourhood_value_score']}"
                )
        return "\n".join(lines)

    if any(word in q for word in ["recommend", "best value", "good value", "where should", "holiday"]):
        lines = ["Here are the strongest value-led options from the current filters:"]
        for city_name in selected_cities:
            recs = get_recommendations(master_data[city_name], limit=3)
            if recs.empty:
                lines.append(f"{city_name}: no recommendations found with the current data.")
                continue
            lines.append(f"{city_name}:")
            for _, row in recs.iterrows():
                lines.append(
                    f"- {row.get('neighbourhood_cleansed', 'Unknown area')}: "
                    f"{format_eur(row.get('price_eur'))}, {row.get('room_type', 'listing')}, "
                    f"value score {row.get('value_score', 'n/a')}"
                )
        return "\n".join(lines)

    if any(word in q for word in ["price", "budget", "cheap", "expensive", "cost"]):
        lines = ["Current price summary:"]
        for city_name in selected_cities:
            summary = market_summary(master_data[city_name])
            lines.append(
                f"- {city_name}: median {format_eur(summary.get('median_price_eur'))}, "
                f"mean {format_eur(summary.get('mean_price_eur'))}, "
                f"{summary.get('listings', 0):,} listings."
            )
        return "\n".join(lines)

    return (
        "I can help compare Madrid and Tokyo, suggest good-value stays, summarize prices, "
        "rank neighbourhoods, and explain the ML model outputs. Try asking for best value, "
        "neighbourhoods, budget, availability, or model performance."
    )
