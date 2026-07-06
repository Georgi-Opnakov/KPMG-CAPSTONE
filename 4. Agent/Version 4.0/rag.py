from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from analytics import compare_cities, format_eur, market_summary, neighbourhood_summary
from data_loader import get_paths
from model_service import best_models, error_segments, model_leaderboard, top_features


@dataclass(frozen=True)
class RagDocument:
    title: str
    content: str
    source: str
    category: str
    city: str = "All"


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return str(value).replace("\n", " ").strip()


def _doc(title: str, content: str, source: str, category: str, city: str = "All") -> RagDocument:
    return RagDocument(
        title=_clean_text(title),
        content=_clean_text(content),
        source=_clean_text(source),
        category=_clean_text(category),
        city=_clean_text(city),
    )


def _safe_read_csv(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _city_market_documents(master_data: dict[str, pd.DataFrame]) -> list[RagDocument]:
    documents: list[RagDocument] = []
    comparison = compare_cities(master_data)

    if not comparison.empty:
        rows = []
        for _, row in comparison.iterrows():
            rows.append(
                f"{row['city']}: {int(row['listings']):,} listings, "
                f"median price {format_eur(row['median_price_eur'])}, "
                f"mean price {format_eur(row['mean_price_eur'])}, "
                f"mean rating {row['mean_rating']}, "
                f"median availability next 30 days {row['median_availability_30']}."
            )
        documents.append(
            _doc(
                "Madrid and Tokyo market comparison",
                " ".join(rows),
                "master city datasets",
                "market_summary",
            )
        )

    for city, df in master_data.items():
        if df.empty:
            continue

        summary = market_summary(df)
        documents.append(
            _doc(
                f"{city} market overview",
                (
                    f"{city} has {summary['listings']:,} listings in the cleaned master dataset. "
                    f"The median nightly price is {format_eur(summary['median_price_eur'])}; "
                    f"the mean nightly price is {format_eur(summary['mean_price_eur'])}. "
                    f"The average rating is {summary['mean_rating']}. "
                    f"Median availability over the next 30 days is {summary['median_availability_30']} days. "
                    f"Median distance to the city centre is {summary['median_distance_to_center_km']} km."
                ),
                f"{city.lower()}_master_model_dataset.csv",
                "market_summary",
                city,
            )
        )

        if "room_type" in df.columns:
            room_summary = (
                df.groupby("room_type", dropna=False)
                .agg(
                    listings=("price_eur", "size"),
                    median_price_eur=("price_eur", "median"),
                    mean_rating=("review_scores_rating", "mean"),
                    median_availability_30=("availability_30", "median"),
                )
                .reset_index()
                .sort_values("listings", ascending=False)
            )
            for _, row in room_summary.iterrows():
                documents.append(
                    _doc(
                        f"{city} room type: {row['room_type']}",
                        (
                            f"In {city}, room type {row['room_type']} has {int(row['listings']):,} listings. "
                            f"Median nightly price is {format_eur(row['median_price_eur'])}; "
                            f"mean rating is {row['mean_rating']:.2f}; "
                            f"median availability next 30 days is {row['median_availability_30']:.1f}."
                        ),
                        f"{city.lower()}_master_model_dataset.csv",
                        "room_type",
                        city,
                    )
                )

        areas = neighbourhood_summary(df)
        for rank, (_, row) in enumerate(areas.head(40).iterrows(), start=1):
            value_label = f"Top value neighbourhood rank {rank}. " if rank <= 10 else ""
            documents.append(
                _doc(
                    f"{city} neighbourhood: {row['neighbourhood_cleansed']}",
                    (
                        f"{value_label}{row['neighbourhood_cleansed']} in {city} has {int(row['listings']):,} listings. "
                        f"Median nightly price is {format_eur(row['median_price_eur'])}; "
                        f"mean rating is {row['mean_rating']:.2f}; "
                        f"mean value rating is {row['mean_value_rating']:.2f}; "
                        f"median distance to centre is {row['median_distance_km']} km; "
                        f"median availability next 30 days is {row['median_availability_30']} days; "
                        f"neighbourhood value score is {row['neighbourhood_value_score']}."
                    ),
                    f"{city.lower()}_master_model_dataset.csv",
                    "neighbourhood",
                    city,
                )
            )

    return documents


def _model_documents(ml_outputs: dict[str, pd.DataFrame]) -> list[RagDocument]:
    documents: list[RagDocument] = []

    leaderboard = model_leaderboard(ml_outputs)
    if not leaderboard.empty:
        for _, row in leaderboard.iterrows():
            documents.append(
                _doc(
                    f"{row['city']} model result: {row['model']}",
                    (
                        f"{row['model']} for {row['city']} produced RMSE {row['rmse_eur']}, "
                        f"MAE {row['mae_eur']}, and R2 {row['r2']} on the holdout test split."
                    ),
                    "expanded_city_model_results.csv",
                    "model_result",
                    row["city"],
                )
            )

    best = best_models(ml_outputs)
    if not best.empty:
        for _, row in best.iterrows():
            documents.append(
                _doc(
                    f"{row['city']} best model",
                    (
                        f"The current best model for {row['city']} is {row['model']}. "
                        f"It achieved RMSE {row['rmse_eur']}, MAE {row['mae_eur']}, and R2 {row['r2']}."
                    ),
                    "expanded_model_recommendation_summary.csv",
                    "model_summary",
                    row["city"],
                )
            )

    for city in ["Madrid", "Tokyo"]:
        best_row = best[best["city"] == city] if not best.empty and "city" in best.columns else pd.DataFrame()
        model_name = best_row["model"].iloc[0] if not best_row.empty else None

        features = top_features(ml_outputs, city=city, model=model_name, limit=12)
        if not features.empty:
            feature_text = "; ".join(
                f"{row['feature']} importance {row['importance']}" for _, row in features.iterrows()
            )
            documents.append(
                _doc(
                    f"{city} top model features",
                    f"Important features affecting {city}'s price prediction model: {feature_text}.",
                    "expanded_city_feature_importance.csv",
                    "feature_importance",
                    city,
                )
            )

        segments = error_segments(ml_outputs, city=city, limit=12)
        if not segments.empty:
            segment_text = "; ".join(
                f"{row['segment_type']} {row['segment']} MAE {row['mae_eur']}" for _, row in segments.iterrows()
            )
            documents.append(
                _doc(
                    f"{city} model error segments",
                    f"The model struggles most in these segments for {city}: {segment_text}.",
                    "best_model_error_segments.csv",
                    "model_error",
                    city,
                )
            )

    comparison = ml_outputs.get("model_comparison", pd.DataFrame())
    if not comparison.empty:
        for _, row in comparison.iterrows():
            documents.append(
                _doc(
                    f"{row['city']} versus original model",
                    (
                        f"{row['comparison_scope']} for {row['city']} used {row['city_model']}. "
                        f"Compared with the original combined-data XGBoost, RMSE changed by "
                        f"{row['rmse_change_vs_original']}, MAE changed by {row['mae_change_vs_original']}, "
                        f"and R2 changed by {row['r2_change_vs_original']}."
                    ),
                    "expanded_city_vs_original_comparison.csv",
                    "model_comparison",
                    row["city"],
                )
            )

    return documents


def _data_dictionary_documents() -> list[RagDocument]:
    paths = get_paths()
    documents: list[RagDocument] = []
    dictionary_files = [
        paths["master"] / "master_model_attribute_dictionary.csv",
        paths["model_ready"] / "model_ready_feature_dictionary.csv",
        paths["master"] / "recommended_model_features_by_city.csv",
        paths["model_ready"] / "model_ready_feature_decisions.csv",
    ]

    for file_path in dictionary_files:
        df = _safe_read_csv(file_path)
        if df.empty:
            continue

        for _, row in df.head(120).iterrows():
            parts = [f"{column}: {_clean_text(value)}" for column, value in row.items()]
            documents.append(
                _doc(
                    f"{file_path.stem}: {_clean_text(row.iloc[0])}",
                    "; ".join(parts),
                    file_path.name,
                    "data_dictionary",
                )
            )

    return documents


def build_documents(master_data: dict[str, pd.DataFrame], ml_outputs: dict[str, pd.DataFrame]) -> list[RagDocument]:
    documents: list[RagDocument] = []
    documents.extend(
        [
            _doc(
                "Project data layers",
                (
                    "The chatbot uses three project data layers: raw source files from Inside Airbnb; "
                    "master city datasets with richer cleaned attributes for planning and chatbot context; "
                    "model_ready datasets for machine learning; and ml_outputs with model results, "
                    "feature importance, predictions, tuning results, and error analysis."
                ),
                "1. Data folder structure",
                "project_context",
            ),
            _doc(
                "Snapshot data limitation",
                (
                    "The project uses cleaned snapshot data, not live Airbnb availability. "
                    "Recommendations are data-backed from the project datasets but should not be presented "
                    "as live booking availability or live prices."
                ),
                "project methodology",
                "project_context",
            ),
        ]
    )
    documents.extend(_city_market_documents(master_data))
    documents.extend(_model_documents(ml_outputs))
    documents.extend(_data_dictionary_documents())
    return documents


def build_rag_index(master_data: dict[str, pd.DataFrame], ml_outputs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    documents = build_documents(master_data, ml_outputs)
    doc_rows = [
        {
            "title": document.title,
            "content": document.content,
            "source": document.source,
            "category": document.category,
            "city": document.city,
            "search_text": f"{document.title} {document.category} {document.city} {document.content}",
        }
        for document in documents
    ]
    docs_df = pd.DataFrame(doc_rows)

    if docs_df.empty:
        return {"documents": docs_df, "vectorizer": None, "matrix": None}

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(docs_df["search_text"])
    return {"documents": docs_df, "vectorizer": vectorizer, "matrix": matrix}


def retrieve_context(
    question: str,
    rag_index: dict[str, Any],
    city: str | None = None,
    top_k: int = 6,
    min_score: float = 0.03,
) -> pd.DataFrame:
    docs_df = rag_index.get("documents", pd.DataFrame())
    vectorizer = rag_index.get("vectorizer")
    matrix = rag_index.get("matrix")

    if docs_df.empty or vectorizer is None or matrix is None:
        return pd.DataFrame(columns=["title", "content", "source", "category", "city", "score"])

    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix).ravel()
    results = docs_df.copy()
    results["score"] = scores
    results["score"] = results["score"] + results["category"].map(_category_boosts(question)).fillna(0.0)
    results["score"] = results["score"] - results["category"].map(_category_penalties(question)).fillna(0.0)
    results["score"] = np.maximum(results["score"], 0)

    if city:
        results = results[(results["city"] == city) | (results["city"] == "All")]

    results = results[results["score"] >= min_score]
    return results.sort_values("score", ascending=False).head(top_k)[
        ["title", "content", "source", "category", "city", "score"]
    ]


def _category_boosts(question: str) -> dict[str, float]:
    q = question.lower()
    boosts: dict[str, float] = {}

    if any(word in q for word in ["neighbourhood", "neighborhood", "area", "district", "where"]):
        boosts.update({"neighbourhood": 0.25, "market_summary": 0.08, "room_type": 0.04})

    if any(word in q for word in ["recommend", "best value", "good value", "holiday", "stay", "budget"]):
        boosts.update({"neighbourhood": 0.14, "market_summary": 0.08, "room_type": 0.06})

    if any(word in q for word in ["model", "prediction", "xgboost", "random forest", "rmse", "mae", "r2"]):
        boosts.update(
            {
                "model_summary": 0.24,
                "model_result": 0.18,
                "model_comparison": 0.18,
                "feature_importance": 0.12,
                "model_error": 0.08,
            }
        )

    asks_about_features = any(word in q for word in ["feature", "features", "attribute", "column", "data dictionary"])
    asks_about_model = any(word in q for word in ["model", "prediction", "xgboost", "random forest", "rmse", "mae", "r2"])
    asks_what_affects_price = "affect" in q and "price" in q

    if asks_about_features and (asks_about_model or asks_what_affects_price):
        boosts.update({"feature_importance": 0.34, "model_summary": 0.10, "data_dictionary": 0.04})
    elif asks_about_features:
        boosts.update({"data_dictionary": 0.18, "feature_importance": 0.10})

    if any(word in q for word in ["availability", "available", "calendar", "season"]):
        boosts.update({"market_summary": 0.12, "neighbourhood": 0.08, "data_dictionary": 0.06})

    return boosts


def _category_penalties(question: str) -> dict[str, float]:
    q = question.lower()
    asks_about_place = any(word in q for word in ["neighbourhood", "neighborhood", "area", "district", "where"])
    asks_about_model = any(word in q for word in ["model", "prediction", "xgboost", "random forest", "rmse", "mae", "r2"])

    if asks_about_place and not asks_about_model:
        return {
            "model_summary": 0.18,
            "model_result": 0.18,
            "model_comparison": 0.18,
            "model_error": 0.12,
            "feature_importance": 0.08,
        }

    return {}


def format_sources(retrieved: pd.DataFrame) -> list[str]:
    if retrieved.empty:
        return []

    sources = []
    for _, row in retrieved.iterrows():
        sources.append(f"{row['title']} ({row['source']})")
    return sources


def context_text(retrieved: pd.DataFrame, max_chars: int = 1800) -> str:
    if retrieved.empty:
        return ""

    chunks = []
    used_chars = 0
    for _, row in retrieved.iterrows():
        chunk = f"- {row['title']}: {row['content']} Source: {row['source']}."
        if used_chars + len(chunk) > max_chars:
            break
        chunks.append(chunk)
        used_chars += len(chunk)

    return "\n".join(chunks)


def generate_rag_response(
    question: str,
    base_answer: str,
    rag_index: dict[str, Any],
    city: str | None = None,
    top_k: int = 5,
) -> tuple[str, list[str], pd.DataFrame]:
    retrieved = retrieve_context(question, rag_index, city=city, top_k=top_k)

    if retrieved.empty:
        return base_answer, [], retrieved

    return base_answer, format_sources(retrieved), retrieved
