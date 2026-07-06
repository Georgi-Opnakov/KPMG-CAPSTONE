from __future__ import annotations

import base64
from datetime import date, timedelta
from html import escape
from pathlib import Path
import re
import threading

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pydeck as pdk
import streamlit.components.v1 as components

from analytics import (
    add_value_score,
    answer_from_data,
    build_recommendation_reason,
    compare_cities,
    filter_listings,
    format_eur,
    get_recommendations,
    market_summary,
    neighbourhood_summary,
    recommendation_price_floor,
)
from api_service import ApiNotConfiguredError, ApiRequestError, api_status, summarize_airroi_stay
from calendar_service import (
    availability_for_listings,
    calendar_cache_exists,
    calendar_metadata,
    check_listing_availability,
    date_bounds,
)
from data_loader import available_cities, data_inventory, load_all_data
from model_service import (
    available_price_model_cities,
    best_models,
    model_feature_defaults,
    model_feature_options,
    model_leaderboard,
    model_summary_text,
    predict_listing_price,
    predict_price,
    prediction_summary_text,
    prediction_status,
    top_features,
)
from prediction_cache_service import load_monthly_prediction_cache, prediction_window_summary
from prompts import APP_NAME, EXAMPLE_QUESTIONS
from rag import build_rag_index, context_text, format_sources, retrieve_context
from llm_service import stream_polished_answer_with_llm, warm_up_ollama_model


APP_DIR = Path(__file__).resolve().parent
BANNER_PATH = APP_DIR / "assets" / "banner.png"
FAVICON_PATH = APP_DIR / "assets" / "favicon_lilly.png"
PAGE_ICON = FAVICON_PATH if FAVICON_PATH.exists() else None
SNAPSHOT_DISCLAIMER = (
    "This demo uses cleaned snapshot data. Recommendations are data-backed, "
    "but they do not represent live Airbnb booking availability unless live API integration is enabled."
)

st.set_page_config(page_title=APP_NAME, page_icon=PAGE_ICON, layout="wide")


@st.cache_resource(show_spinner=False)
def start_llm_warmup() -> bool:
    """Start one non-blocking Ollama warm-up for this Streamlit server process."""
    thread = threading.Thread(target=warm_up_ollama_model, daemon=True)
    thread.start()
    return True


@st.cache_data(show_spinner=False)
def image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def apply_custom_styles() -> None:
    """KPMG-inspired visual system for the Streamlit demo.

    Uses a professional blue/cyan palette inspired by KPMG-style consulting dashboards
    without using any trademarked logo assets.
    """
    st.markdown("""
    <style>
    :root {
        --kpmg-blue: #00338D;
        --kpmg-blue-dark: #001F5B;
        --kpmg-cobalt: #005EB8;
        --kpmg-cyan: #00A3E0;
        --kpmg-sky: #EAF4FF;
        --text: #111827;
        --muted: #64748B;
        --border: #D8E3F0;
        --surface: #FFFFFF;
        --background: #F4F7FB;
        --warning-bg: #FFF7D6;
        --warning-text: #6B4E00;
    }

    .stApp {
        background: linear-gradient(180deg, #F8FBFF 0%, var(--background) 100%);
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 8rem;
        max-width: 1500px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F4F8FE 100%);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--kpmg-blue-dark);
    }

    h1, h2, h3 {
        color: var(--kpmg-blue-dark);
        letter-spacing: -0.025em;
    }

    p, li, label, span {
        color: var(--text);
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid #F5D56C;
    }

    .custom-nav {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 18px;
        width: 100%;
        margin: 4px auto 22px auto;
        position: relative;
        z-index: 30;
    }

    .custom-nav a,
    .custom-nav button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 42px;
        padding: 0 16px;
        border-radius: 10px 10px 0 0;
        color: var(--kpmg-blue-dark) !important;
        background: transparent;
        border: 0;
        font-size: 14px;
        font-weight: 700;
        text-decoration: none !important;
        cursor: pointer;
        transition: background 0.15s ease, color 0.15s ease, box-shadow 0.15s ease;
        white-space: nowrap;
        list-style: none;
    }

    .custom-nav a:hover,
    .custom-nav button:hover {
        background: #EAF4FF;
        color: var(--kpmg-blue) !important;
    }

    .custom-nav .nav-link.active,
    .custom-nav .nav-popover-trigger.active,
    .custom-nav .nav-popover-trigger:hover {
        background: linear-gradient(135deg, var(--kpmg-blue-dark) 0%, var(--kpmg-blue) 70%, var(--kpmg-cyan) 100%);
        color: #FFFFFF !important;
        border-bottom: 3px solid var(--kpmg-cyan);
        box-shadow: 0 8px 18px rgba(0, 51, 141, 0.14);
    }

    .nav-dropdown-menu {
        margin: 0;
        min-width: 230px;
        padding: 8px;
        border-radius: 12px;
        background: #FFFFFF;
        border: 1px solid var(--border);
        box-shadow: 0 16px 34px rgba(0, 31, 91, 0.18);
    }

    .nav-dropdown-menu:popover-open {
        position: fixed;
        inset: auto auto auto auto;
        top: 170px;
        left: 50%;
        transform: translateX(160px);
    }

    .nav-dropdown-menu::backdrop {
        background: transparent;
    }

    .nav-dropdown-menu a {
        display: flex;
        width: 100%;
        justify-content: flex-start;
        min-height: 38px;
        padding: 0 12px;
        border-radius: 9px;
        border-bottom: none;
        box-shadow: none;
        color: var(--kpmg-blue-dark) !important;
        text-decoration: none !important;
    }

    .nav-dropdown-menu a:hover,
    .nav-dropdown-menu a.active {
        background: #EAF4FF;
        color: var(--kpmg-blue-dark) !important;
        font-weight: 900;
    }

    div[data-testid="stTabs"] div[role="tablist"],
    div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
        justify-content: center;
        gap: 12px;
        width: 100%;
    }

    div[data-testid="stTabs"] div[role="tablist"] button[data-baseweb="tab"] {
        flex: 0 0 auto;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        font-size: 14px;
        font-weight: 700;
        color: var(--kpmg-blue-dark);
        padding: 10px 14px;
        border-radius: 10px 10px 0 0;
        transition: all 0.15s ease-in-out;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
        background: #EAF4FF;
        color: var(--kpmg-blue) !important;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: linear-gradient(135deg, var(--kpmg-blue-dark) 0%, var(--kpmg-blue) 70%, var(--kpmg-cyan) 100%) !important;
        color: #FFFFFF !important;
        border-bottom: 3px solid var(--kpmg-cyan) !important;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] p,
    div[data-testid="stTabs"] button[aria-selected="true"] span {
        color: #FFFFFF !important;
    }

    div[data-testid="metric-container"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        padding: 16px;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0, 51, 141, 0.06);
    }

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border);
        box-shadow: 0 2px 10px rgba(0, 51, 141, 0.04);
    }

    .advisor-hero {
        background: linear-gradient(135deg, var(--kpmg-blue-dark) 0%, var(--kpmg-blue) 62%, var(--kpmg-cyan) 100%);
        color: #FFFFFF !important;
        border-radius: 22px;
        padding: 34px 36px;
        margin-bottom: 18px;
        box-shadow: 0 14px 34px rgba(0, 51, 141, 0.22);
        border: 1px solid rgba(255,255,255,0.25);
    }

    .advisor-hero,
    .advisor-hero * {
        color: #FFFFFF !important;
    }

    .advisor-hero h1 {
        color: #FFFFFF !important;
        font-size: 42px;
        margin-bottom: 8px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.18);
    }

    .advisor-hero p {
        color: #F3F8FF !important;
        font-size: 16px;
        margin-bottom: 4px;
    }

    .advisor-pill {
        display: inline-block;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.32);
        color: #FFFFFF;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 12px;
        letter-spacing: .02em;
    }

    .advisor-banner {
        position: relative;
        border-radius: 22px;
        overflow: hidden;
        width: min(100%, 1500px);
        height: auto;
        margin: 0 auto 10px auto;
        background: #EAF4FF;
        border: 1px solid rgba(0, 51, 141, 0.12);
        box-shadow: 0 12px 24px rgba(0, 51, 141, 0.14);
    }

    .advisor-banner img {
        display: block;
        width: 100%;
        height: auto;
    }

    .disclaimer-help-wrap {
        position: absolute;
        top: 16px;
        right: 18px;
        z-index: 4;
    }

    .disclaimer-help {
        width: 34px;
        height: 34px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #FFF7D6;
        border: 1px solid #F5D56C;
        color: var(--warning-text) !important;
        font-weight: 900;
        box-shadow: 0 6px 18px rgba(0, 31, 91, 0.18);
        cursor: help;
    }

    .disclaimer-tooltip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        top: 44px;
        right: 0;
        width: min(360px, 72vw);
        padding: 12px 14px;
        border-radius: 12px;
        background: #FFF7D6;
        border: 1px solid #F5D56C;
        color: var(--warning-text) !important;
        font-weight: 650;
        line-height: 1.35;
        text-align: left;
        box-shadow: 0 12px 28px rgba(0, 31, 91, 0.18);
        transition: opacity 0.12s ease, visibility 0.12s ease;
    }

    .disclaimer-tooltip::before {
        content: "";
        position: absolute;
        top: -6px;
        right: 11px;
        width: 12px;
        height: 12px;
        transform: rotate(45deg);
        background: #FFF7D6;
        border-left: 1px solid #F5D56C;
        border-top: 1px solid #F5D56C;
    }

    .disclaimer-help-wrap:hover .disclaimer-tooltip,
    .disclaimer-help-wrap:focus-within .disclaimer-tooltip {
        visibility: visible;
        opacity: 1;
    }

    .section-card {
        background-color: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 20px 24px;
        margin: 14px 0 20px 0;
        box-shadow: 0 4px 18px rgba(0, 51, 141, 0.05);
    }

    .kpi-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-left: 5px solid var(--kpmg-cyan);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 5px 18px rgba(0, 51, 141, 0.08);
        min-height: 118px;
    }

    .kpi-label {
        color: var(--muted);
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }

    .kpi-value {
        color: var(--kpmg-blue-dark);
        font-size: 30px;
        font-weight: 800;
        line-height: 1.1;
    }

    .kpi-note {
        color: var(--muted);
        font-size: 12px;
        margin-top: 6px;
    }

    .recommendation-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-left: 6px solid var(--kpmg-blue);
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 6px 18px rgba(0, 51, 141, 0.07);
    }

    .recommendation-title {
        color: var(--kpmg-blue-dark);
        font-size: 18px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .recommendation-meta {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 12px;
    }

    .badge {
        display: inline-block;
        background: var(--kpmg-sky);
        color: var(--kpmg-blue);
        border: 1px solid #C7E7FF;
        border-radius: 999px;
        padding: 5px 9px;
        font-size: 12px;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .badge-strong {
        background: var(--kpmg-blue);
        color: #FFFFFF;
        border-color: var(--kpmg-blue);
    }

    .small-muted {
        color: var(--muted);
        font-size: 14px;
    }

    div.st-key-recommendation_compare_city_a div[data-baseweb="select"] > div,
    div.st-key-recommendation_compare_city_b div[data-baseweb="select"] > div {
        background: #FFFFFF !important;
        color: var(--kpmg-blue-dark) !important;
        border: 1px solid var(--border) !important;
        border-left: 6px solid var(--kpmg-blue) !important;
        border-radius: 14px !important;
        min-height: 86px !important;
        position: relative !important;
        display: flex !important;
        align-items: center !important;
        padding: 0 62px !important;
        box-shadow: 0 6px 18px rgba(0, 51, 141, 0.06) !important;
    }

    div.st-key-recommendation_compare_city_a div[data-baseweb="select"] > div > div:first-child,
    div.st-key-recommendation_compare_city_b div[data-baseweb="select"] > div > div:first-child {
        flex: 1 1 auto !important;
        width: 100% !important;
        justify-content: center !important;
        align-items: center !important;
        min-width: 0 !important;
    }

    div.st-key-recommendation_compare_city_a div[data-baseweb="select"] > div > div:first-child *,
    div.st-key-recommendation_compare_city_b div[data-baseweb="select"] > div > div:first-child * {
        color: var(--kpmg-blue-dark) !important;
        font-size: 25px !important;
        font-weight: 850 !important;
        text-align: center !important;
    }

    div.st-key-recommendation_compare_city_a div[data-baseweb="select"] > div > div:last-child,
    div.st-key-recommendation_compare_city_b div[data-baseweb="select"] > div > div:last-child {
        position: absolute !important;
        right: 18px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        background: #EAF4FF !important;
        border: 2px solid var(--kpmg-blue) !important;
        border-radius: 8px !important;
        height: 31px !important;
        width: 31px !important;
        min-width: 31px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        box-shadow: 0 3px 8px rgba(0, 51, 141, 0.12) !important;
    }

    div.st-key-recommendation_compare_city_a div[data-baseweb="select"] svg,
    div.st-key-recommendation_compare_city_b div[data-baseweb="select"] svg {
        color: var(--kpmg-blue) !important;
        fill: var(--kpmg-blue) !important;
        display: block !important;
        height: 18px !important;
        width: 18px !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    .prediction-result-hero {
        background: linear-gradient(135deg, var(--kpmg-blue-dark) 0%, var(--kpmg-blue) 72%, var(--kpmg-cyan) 100%);
        color: #FFFFFF !important;
        border-radius: 16px;
        padding: 24px 28px;
        margin: 18px 0 14px 0;
        box-shadow: 0 10px 28px rgba(0, 51, 141, 0.18);
    }

    .prediction-result-hero,
    .prediction-result-hero * {
        color: #FFFFFF !important;
    }

    .prediction-result-label {
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.86;
        margin-bottom: 6px;
    }

    .prediction-result-price {
        font-size: 40px;
        line-height: 1.05;
        font-weight: 900;
        margin-bottom: 8px;
    }

    .prediction-result-subtitle {
        font-size: 15px;
        opacity: 0.92;
    }

    .price-source-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
        gap: 12px;
        margin: 14px 0 16px 0;
    }

    .price-source-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-left: 5px solid var(--kpmg-cobalt);
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 5px 16px rgba(0, 51, 141, 0.08);
    }

    .price-source-card.price-source-good {
        border-left-color: var(--kpmg-cyan);
        background: #F4FAFF;
    }

    .price-source-card.price-source-warning {
        border-left-color: #F59E0B;
        background: #FFFBEB;
    }

    .price-source-title {
        color: var(--muted);
        font-size: 12px;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }

    .price-source-value {
        color: var(--kpmg-blue-dark);
        font-size: 18px;
        font-weight: 900;
        line-height: 1.18;
        margin-bottom: 4px;
    }

    .price-source-detail {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.35;
    }

    .prediction-location-card {
        background: linear-gradient(135deg, var(--kpmg-blue-dark) 0%, var(--kpmg-blue) 100%);
        border: 1px solid rgba(0, 163, 224, 0.65);
        border-radius: 10px;
        min-height: 42px;
        padding: 0 13px;
        box-shadow: 0 3px 10px rgba(0, 51, 141, 0.10);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
    }

    .prediction-location-label {
        color: rgba(255, 255, 255, 0.82);
        font-size: 12px;
        font-weight: 800;
        line-height: 1.15;
        text-align: right;
        max-width: 48%;
    }

    .prediction-location-value {
        color: #FFFFFF;
        font-size: 18px;
        font-weight: 850;
        line-height: 1.15;
        white-space: nowrap;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input,
    div[data-testid="stDateInput"] input,
    div[data-testid="stTextArea"] textarea {
        background: linear-gradient(135deg, var(--kpmg-blue-dark) 0%, var(--kpmg-blue) 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(0, 163, 224, 0.65) !important;
        border-radius: 10px !important;
        box-shadow: 0 3px 10px rgba(0, 51, 141, 0.10);
    }

    div[data-baseweb="select"] *,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input,
    div[data-testid="stDateInput"] input,
    div[data-testid="stTextArea"] textarea {
        color: #FFFFFF !important;
    }

    div[data-baseweb="select"] svg,
    div[data-testid="stNumberInput"] svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    div[data-baseweb="popover"] div[role="listbox"] {
        background: var(--kpmg-blue-dark) !important;
        border: 1px solid rgba(0, 163, 224, 0.55) !important;
    }

    div[role="option"] {
        color: #FFFFFF !important;
        background: var(--kpmg-blue-dark) !important;
    }

    div[role="option"]:hover,
    div[role="option"][aria-selected="true"] {
        background: var(--kpmg-cobalt) !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stNumberInput"] button {
        background: var(--kpmg-blue-dark) !important;
        border-color: rgba(0, 163, 224, 0.45) !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stSlider"] div[role="slider"] {
        background: var(--kpmg-cyan) !important;
        border-color: #FFFFFF !important;
        box-shadow: 0 2px 10px rgba(0, 51, 141, 0.18);
    }

    div[data-testid="stSlider"] [data-testid="stThumbValue"] {
        color: var(--kpmg-blue-dark) !important;
        font-weight: 800;
    }

    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    div[data-testid="stBottom"],
    div[data-testid="stBottomBlockContainer"],
    div[data-testid="stChatFloatingInputContainer"] {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    div[data-testid="stBottom"] > div,
    div[data-testid="stBottomBlockContainer"] > div,
    div[data-testid="stChatFloatingInputContainer"] > div {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    div[data-testid="stChatInput"] {
        position: fixed !important;
        left: 50% !important;
        right: auto !important;
        bottom: 22px !important;
        transform: translateX(-50%) !important;
        z-index: 1000 !important;
        width: min(92vw, 1380px) !important;
        max-width: calc(100vw - 72px) !important;
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        padding: 0 !important;
        backdrop-filter: none !important;
    }

    div[data-testid="stChatInput"] > div {
        background: #FFFFFF !important;
        border: 1px solid #9CCBFF !important;
        border-radius: 18px !important;
        box-shadow: 0 14px 36px rgba(0, 51, 141, 0.16) !important;
        padding: 8px 10px !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }

    div[data-testid="stChatInput"] > div > div,
    div[data-testid="stChatInput"] div[data-baseweb="textarea"],
    div[data-testid="stChatInput"] div[data-baseweb="textarea"] *,
    div[data-testid="stChatInput"] [data-baseweb="base-input"],
    div[data-testid="stChatInput"] [data-baseweb="base-input"] *,
    div[data-testid="stChatInput"] [data-testid="stChatInputTextArea"],
    div[data-testid="stChatInput"] [data-testid="stChatInputTextArea"] * {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }

    div[data-testid="stChatInput"] div[data-baseweb="textarea"] {
        flex: 1 1 auto !important;
        background: transparent !important;
        border: none !important;
        border-radius: 14px !important;
        box-shadow: none !important;
    }

    div[data-testid="stChatInput"] textarea {
        min-height: 42px !important;
        background: transparent !important;
        color: var(--kpmg-blue-dark) !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        padding: 8px 4px !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #64748B !important;
        opacity: 1 !important;
    }

    div[data-testid="stChatInput"] button {
        background: #EAF4FF !important;
        border: 1px solid #BFE4FF !important;
        color: var(--kpmg-blue-dark) !important;
        border-radius: 12px !important;
        width: 38px !important;
        height: 38px !important;
        min-width: 38px !important;
        min-height: 38px !important;
        flex: 0 0 38px !important;
        margin: 0 !important;
    }

    div[data-testid="stChatInput"] button svg {
        color: var(--kpmg-blue-dark) !important;
        fill: var(--kpmg-blue-dark) !important;
    }

    details[data-testid="stExpander"],
    div[data-testid="stExpander"] details {
        background: #FFFFFF !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 3px 12px rgba(0, 51, 141, 0.06);
    }

    details[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] details summary,
    div[data-testid="stExpander"] > details > summary {
        background: linear-gradient(135deg, var(--kpmg-blue-dark) 0%, var(--kpmg-blue) 100%) !important;
        background-color: var(--kpmg-blue) !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        font-weight: 800 !important;
    }

    details[data-testid="stExpander"][open] summary,
    div[data-testid="stExpander"] details[open] summary,
    div[data-testid="stExpander"] > details[open] > summary {
        border-radius: 10px 10px 0 0 !important;
        border-bottom: 1px solid rgba(0, 163, 224, 0.35) !important;
    }

    details[data-testid="stExpander"] summary *,
    details[data-testid="stExpander"] summary p,
    details[data-testid="stExpander"] summary span,
    details[data-testid="stExpander"] summary svg,
    div[data-testid="stExpander"] summary *,
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary span,
    div[data-testid="stExpander"] summary svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    details[data-testid="stExpander"] div[data-testid="stExpanderDetails"],
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"],
    div[data-testid="stExpanderDetails"] {
        background: #FFFFFF !important;
        color: var(--text) !important;
        padding: 14px 18px 18px 18px !important;
    }

    details[data-testid="stExpander"] div[data-testid="stExpanderDetails"] *,
    details[data-testid="stExpander"] div[data-testid="stExpanderDetails"] p,
    details[data-testid="stExpander"] div[data-testid="stExpanderDetails"] li,
    details[data-testid="stExpander"] div[data-testid="stExpanderDetails"] span,
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] *,
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] p,
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] li,
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] span {
        color: var(--text) !important;
    }

    .prediction-table-wrap {
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
        margin-top: 12px;
        box-shadow: 0 4px 16px rgba(0, 51, 141, 0.07);
        background: #FFFFFF;
    }

    .prediction-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }

    .prediction-table thead th {
        background: linear-gradient(135deg, var(--kpmg-blue-dark) 0%, var(--kpmg-blue) 100%);
        color: #FFFFFF !important;
        padding: 12px 14px;
        text-align: left;
        font-weight: 800;
    }

    .prediction-table tbody td {
        padding: 11px 14px;
        border-bottom: 1px solid #E4ECF5;
        color: var(--text) !important;
        background: #FFFFFF;
    }

    .prediction-table tbody tr:nth-child(even) td {
        background: #F4F8FE;
    }

    .prediction-table tbody tr:last-child td {
        border-bottom: none;
    }

    .prediction-table tbody td:first-child {
        color: var(--kpmg-blue-dark) !important;
        font-weight: 800;
        width: 22%;
    }

    .overview-table-wrap {
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow-x: auto;
        margin: 10px 0 24px 0;
        background: #FFFFFF;
        box-shadow: 0 4px 16px rgba(0, 51, 141, 0.06);
    }

    .overview-table {
        width: 100%;
        min-width: 760px;
        border-collapse: collapse;
        font-size: 14px;
    }

    .overview-table thead th {
        background: linear-gradient(135deg, var(--kpmg-blue-dark) 0%, var(--kpmg-blue) 100%);
        color: #FFFFFF !important;
        text-align: left;
        padding: 12px 14px;
        font-weight: 800;
        white-space: nowrap;
    }

    .overview-table tbody td {
        padding: 11px 14px;
        border-bottom: 1px solid #DDE8F4;
        color: var(--text) !important;
        background: #FFFFFF;
    }

    .overview-table tbody tr:nth-child(even) td {
        background: #F4F8FE;
    }

    .overview-table tbody tr:hover td {
        background: #EAF4FF;
    }

    .overview-table tbody tr.best-model-row td {
        background: #EAF4FF !important;
        font-weight: 850;
        border-top: 2px solid #9CD0FF;
        border-bottom: 2px solid #9CD0FF;
    }

    .overview-table tbody tr.best-model-row td:first-child {
        border-left: 6px solid var(--kpmg-blue);
    }

    .overview-table tbody tr:last-child td {
        border-bottom: none;
    }

    .overview-table tbody td:first-child {
        color: var(--kpmg-blue-dark) !important;
        font-weight: 800;
    }

    .knowledge-table-wrap {
        overflow: visible;
    }

    .knowledge-table {
        min-width: 720px;
    }

    .knowledge-info-cell {
        width: 90px;
        text-align: center;
        position: relative;
    }

    .kb-help-wrap {
        position: relative;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }

    .kb-help {
        width: 24px;
        height: 24px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #EAF4FF;
        border: 1px solid #9CCBFF;
        color: var(--kpmg-blue-dark) !important;
        font-weight: 900;
        cursor: help;
        box-shadow: 0 2px 8px rgba(0, 51, 141, 0.10);
    }

    .kb-help-tooltip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        z-index: 20;
        top: 50%;
        right: 34px;
        transform: translateY(-50%);
        width: min(340px, 70vw);
        padding: 10px 12px;
        border-radius: 10px;
        background: var(--kpmg-blue-dark);
        color: #FFFFFF !important;
        font-weight: 600;
        line-height: 1.35;
        text-align: left;
        white-space: normal;
        box-shadow: 0 10px 28px rgba(0, 31, 91, 0.24);
        transition: opacity 0.12s ease, visibility 0.12s ease;
    }

    .kb-help-tooltip::before {
        content: "";
        position: absolute;
        top: 50%;
        right: -6px;
        transform: translateY(-50%) rotate(45deg);
        width: 12px;
        height: 12px;
        background: var(--kpmg-blue-dark);
    }

    .kb-help-wrap:hover .kb-help-tooltip,
    .kb-help-wrap:focus-within .kb-help-tooltip {
        visibility: visible;
        opacity: 1;
    }


    .blue-insight-card {
        background: linear-gradient(135deg, var(--kpmg-blue-dark) 0%, var(--kpmg-blue) 100%);
        color: #FFFFFF !important;
        border-radius: 16px;
        padding: 18px 20px;
        margin: 14px 0 20px 0;
        box-shadow: 0 8px 22px rgba(0, 51, 141, 0.16);
    }

    .blue-insight-card,
    .blue-insight-card * {
        color: #FFFFFF !important;
    }

    .accent-line {
        height: 4px;
        width: 72px;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--kpmg-cyan), var(--kpmg-blue));
        margin: 4px 0 18px 0;
    }

    .stButton > button,
    div[data-testid="stButton"] button,
    div[data-testid="stFormSubmitButton"] button {
        border-radius: 10px;
        border: 1px solid var(--kpmg-blue);
        color: #FFFFFF !important;
        background: linear-gradient(135deg, var(--kpmg-blue-dark) 0%, var(--kpmg-blue) 100%) !important;
        font-weight: 700;
    }

    .stButton > button *,
    div[data-testid="stButton"] button *,
    div[data-testid="stFormSubmitButton"] button *,
    div[data-testid="stFormSubmitButton"] button p,
    div[data-testid="stFormSubmitButton"] button span,
    button[kind="primary"] *,
    button[kind="secondary"] *,
    button[kind="formSubmit"] * {
        color: #FFFFFF !important;
    }

    .stButton > button:hover,
    div[data-testid="stButton"] button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background: var(--kpmg-cyan);
        border-color: var(--kpmg-cyan);
        color: #FFFFFF !important;
    }

    div[data-testid="stButtonGroup"] {
        margin: 8px 0 18px 0;
    }

    div[data-testid="stButtonGroup"] div[data-baseweb="button-group"] {
        gap: 8px !important;
        flex-wrap: nowrap !important;
        overflow-x: auto;
        padding-bottom: 2px;
    }

    div[data-testid="stButtonGroup"] button[kind="pills"],
    div[data-testid="stButtonGroup"] button[kind="pillsActive"] {
        background: #EAF4FF !important;
        border: 1px solid #B7DAFF !important;
        color: var(--kpmg-blue-dark) !important;
        border-radius: 999px !important;
        box-shadow: 0 2px 8px rgba(0, 51, 141, 0.06);
        min-height: 34px !important;
        padding: 7px 14px !important;
        font-weight: 800 !important;
    }

    div[data-testid="stButtonGroup"] button[kind="pills"] *,
    div[data-testid="stButtonGroup"] button[kind="pillsActive"] * {
        color: var(--kpmg-blue-dark) !important;
    }

    div[data-testid="stButtonGroup"] button[kind="pills"]:hover,
    div[data-testid="stButtonGroup"] button[kind="pills"]:focus-visible,
    div[data-testid="stButtonGroup"] button[kind="pillsActive"],
    div[data-testid="stButtonGroup"] button[kind="pillsActive"]:hover,
    div[data-testid="stButtonGroup"] button[kind="pillsActive"]:focus-visible {
        background: linear-gradient(135deg, var(--kpmg-blue-dark) 0%, var(--kpmg-blue) 100%) !important;
        border-color: var(--kpmg-blue) !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stButtonGroup"] button[kind="pills"]:hover *,
    div[data-testid="stButtonGroup"] button[kind="pills"]:focus-visible *,
    div[data-testid="stButtonGroup"] button[kind="pillsActive"] *,
    div[data-testid="stButtonGroup"] button[kind="pillsActive"]:hover *,
    div[data-testid="stButtonGroup"] button[kind="pillsActive"]:focus-visible * {
        color: #FFFFFF !important;
    }

    div.st-key-recommendation_style div[data-baseweb="button-group"],
    div.st-key-recommendation_rank_mode div[data-baseweb="button-group"] {
        gap: 10px !important;
        overflow: visible !important;
        flex-wrap: nowrap !important;
        margin-top: 0 !important;
    }

    div.st-key-recommendation_rank_mode div[data-baseweb="button-group"] {
        justify-content: center !important;
        width: 100% !important;
    }

    div.st-key-recommendation_distance_cap {
        width: 148px !important;
        margin-left: auto !important;
    }

    div.st-key-recommendation_style button[kind="pills"],
    div.st-key-recommendation_style button[kind="pillsActive"] {
        min-width: 94px !important;
        padding-left: 18px !important;
        padding-right: 18px !important;
    }

    div.st-key-recommendation_rank_mode button[kind="pills"],
    div.st-key-recommendation_rank_mode button[kind="pillsActive"] {
        min-width: 138px !important;
        padding-left: 18px !important;
        padding-right: 18px !important;
    }

    div.st-key-recommendation_distance_cap div[data-baseweb="select"] > div {
        background: #EAF4FF !important;
        color: var(--kpmg-blue-dark) !important;
        border: 1px solid #B7DAFF !important;
        border-radius: 999px !important;
        min-height: 36px !important;
        height: 36px !important;
        width: 148px !important;
        min-width: 148px !important;
        max-width: 148px !important;
        padding-left: 14px !important;
        padding-right: 8px !important;
        box-shadow: 0 2px 8px rgba(0, 51, 141, 0.06) !important;
    }

    div.st-key-recommendation_distance_cap div[data-baseweb="select"] *,
    div.st-key-recommendation_distance_cap div[data-baseweb="select"] input {
        color: var(--kpmg-blue-dark) !important;
        font-weight: 800 !important;
        font-size: 14px !important;
    }

    div.st-key-recommendation_distance_cap div[data-baseweb="select"] svg {
        color: var(--kpmg-blue) !important;
        fill: var(--kpmg-blue) !important;
    }

    div[class*="st-key-main_nav_pills_"] button[kind="pills"],
    div[class*="st-key-main_nav_pills_"] button[kind="pillsActive"] {
        background: linear-gradient(135deg, #4361EE 0%, var(--kpmg-blue) 100%) !important;
        border: 1px solid transparent !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(0, 51, 141, 0.12) !important;
        width: 166px !important;
        min-width: 166px !important;
        max-width: 166px !important;
        min-height: 40px !important;
        height: 40px !important;
        padding: 0 14px !important;
        justify-content: center !important;
    }

    div[class*="st-key-main_nav_pills_"] div[data-baseweb="button-group"] {
        justify-content: center !important;
        align-items: center !important;
        gap: 10px !important;
        overflow: visible !important;
    }

    div[class*="st-key-main_nav_pills_"] button[kind="pills"] *,
    div[class*="st-key-main_nav_pills_"] button[kind="pillsActive"] * {
        color: #FFFFFF !important;
    }

    div[class*="st-key-main_nav_pills_"] button[kind="pills"]:hover,
    div[class*="st-key-main_nav_pills_"] button[kind="pillsActive"],
    div[class*="st-key-main_nav_pills_"] button[kind="pillsActive"]:hover {
        background: linear-gradient(135deg, var(--kpmg-blue-dark) 0%, var(--kpmg-blue) 100%) !important;
        border-color: transparent !important;
        color: #FFFFFF !important;
        transform: translateY(-1px);
    }

    div[class*="st-key-nav_button_"] button {
        width: 100% !important;
        min-width: 0 !important;
        max-width: none !important;
        height: 46px !important;
        min-height: 46px !important;
        padding: 0 18px !important;
        border-radius: 10px !important;
        background: linear-gradient(135deg, #4361EE 0%, var(--kpmg-blue) 100%) !important;
        border: 1px solid transparent !important;
        color: #FFFFFF !important;
        box-shadow: 0 7px 18px rgba(0, 51, 141, 0.16) !important;
        font-weight: 850 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    div[class*="st-key-nav_button_"] button[kind="primary"],
    div[class*="st-key-nav_button_"] button:hover {
        background: linear-gradient(135deg, var(--kpmg-blue-dark) 0%, var(--kpmg-blue) 100%) !important;
        transform: translateY(-1px);
    }

    div[class*="st-key-nav_button_"] button * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    div[class*="st-key-quick_question_"] button[kind="pills"],
    div[class*="st-key-quick_question_"] button[kind="pillsActive"] {
        background: #F4FAFF !important;
        border: 2px solid #0EA5FF !important;
        border-radius: 999px !important;
        color: #008CFF !important;
        box-shadow: 0 2px 8px rgba(14, 165, 255, 0.09) !important;
        min-height: 36px !important;
        padding: 8px 18px !important;
    }

    div[class*="st-key-quick_question_"] button[kind="pills"] *,
    div[class*="st-key-quick_question_"] button[kind="pillsActive"] * {
        color: #008CFF !important;
    }

    div[class*="st-key-quick_question_"] button[kind="pills"]:hover,
    div[class*="st-key-quick_question_"] button[kind="pills"]:focus-visible,
    div[class*="st-key-quick_question_"] button[kind="pillsActive"],
    div[class*="st-key-quick_question_"] button[kind="pillsActive"]:hover {
        background: #FFFFFF !important;
        border-color: #0099FF !important;
        color: #008CFF !important;
    }

    div[class*="st-key-quick_question_"] button[kind="pills"]:hover *,
    div[class*="st-key-quick_question_"] button[kind="pills"]:focus-visible *,
    div[class*="st-key-quick_question_"] button[kind="pillsActive"] *,
    div[class*="st-key-quick_question_"] button[kind="pillsActive"]:hover * {
        color: #008CFF !important;
    }

    div[data-testid="stPopover"] button {
        background: linear-gradient(135deg, #4361EE 0%, var(--kpmg-blue) 100%) !important;
        border: 1px solid transparent !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(0, 51, 141, 0.12) !important;
        min-height: 46px !important;
        height: 46px !important;
        padding: 0 14px !important;
        line-height: 1 !important;
        font-weight: 800 !important;
        width: 100% !important;
        min-width: 0 !important;
        max-width: none !important;
        justify-content: center !important;
    }

    div[data-testid="stPopover"] button *,
    div[data-testid="stPopover"] button svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    div[data-testid="stPopover"] {
        margin: 8px 0 18px 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    div[data-testid="stPopover"] button *,
    div[data-testid="stPopover"] button[kind="secondary"] * {
        color: #FFFFFF !important;
    }

    div[data-testid="stPopover"] button:hover,
    div[data-testid="stPopover"] button:focus-visible,
    div[data-testid="stPopover"] button[kind="primary"] {
        background: linear-gradient(135deg, var(--kpmg-blue-dark) 0%, var(--kpmg-blue) 100%) !important;
        border-color: var(--kpmg-blue) !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stPopover"] button:hover *,
    div[data-testid="stPopover"] button:focus-visible *,
    div[data-testid="stPopover"] button[kind="primary"] * {
        color: #FFFFFF !important;
    }

    div[data-testid="stPopoverBody"],
    div[data-testid="stPopoverContent"] {
        background: #FFFFFF !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        box-shadow: 0 16px 34px rgba(0, 31, 91, 0.16) !important;
        padding: 10px !important;
    }

    div[data-testid="stPopoverBody"] *,
    div[data-testid="stPopoverContent"] * {
        color: var(--text) !important;
    }

    div[data-baseweb="popover"]:not(:has(div[role="listbox"])),
    div[data-baseweb="popover"]:not(:has(div[role="listbox"])) > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    div[data-baseweb="popover"]:not(:has(div[role="listbox"])) div[data-testid="stPopoverBody"],
    div[data-baseweb="popover"]:not(:has(div[role="listbox"])) div[data-testid="stPopoverContent"],
    div[data-baseweb="popover"]:not(:has(div[role="listbox"])) div[data-testid="stVerticalBlock"] {
        background: #FFFFFF !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        box-shadow: 0 16px 34px rgba(0, 31, 91, 0.16) !important;
        padding: 8px !important;
    }

    div[data-baseweb="popover"]:not(:has(div[role="listbox"])) div[data-testid="stButton"] button {
        background: #EAF4FF !important;
        border: 1px solid #B7DAFF !important;
        color: var(--kpmg-blue-dark) !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(0, 51, 141, 0.06) !important;
        min-height: 38px !important;
        font-weight: 800 !important;
    }

    div[data-baseweb="popover"]:not(:has(div[role="listbox"])) div[data-testid="stButton"] button *,
    div[data-baseweb="popover"]:not(:has(div[role="listbox"])) div[data-testid="stButton"] button[kind="secondary"] * {
        color: var(--kpmg-blue-dark) !important;
    }

    div[data-baseweb="popover"]:not(:has(div[role="listbox"])) div[data-testid="stButton"] button:hover,
    div[data-baseweb="popover"]:not(:has(div[role="listbox"])) div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, var(--kpmg-blue-dark) 0%, var(--kpmg-blue) 100%) !important;
        border-color: var(--kpmg-blue) !important;
        color: #FFFFFF !important;
    }

    div[data-baseweb="popover"]:not(:has(div[role="listbox"])) div[data-testid="stButton"] button:hover *,
    div[data-baseweb="popover"]:not(:has(div[role="listbox"])) div[data-testid="stButton"] button[kind="primary"] * {
        color: #FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_styles()


@st.cache_data(show_spinner=False)
def get_bundle():
    return load_all_data()


@st.cache_resource(show_spinner=False)
def get_rag_index():
    bundle = get_bundle()
    return build_rag_index(bundle["master"], bundle["ml_outputs"])


def _recommendation_filter_key(filters: dict) -> tuple[tuple[str, object], ...]:
    """Convert recommendation filters into a stable cache key."""
    key_parts = []
    for key, value in sorted(filters.items()):
        if isinstance(value, list):
            value = tuple(value)
        elif isinstance(value, set):
            value = tuple(sorted(value))
        key_parts.append((key, value))
    return tuple(key_parts)


def _recommendation_filters_from_key(filter_key: tuple[tuple[str, object], ...]) -> dict:
    filters = dict(filter_key)
    amenities = filters.get("amenities")
    if isinstance(amenities, tuple):
        filters["amenities"] = list(amenities)
    return filters


@st.cache_data(show_spinner=False, max_entries=400)
def cached_recommendation_price_floor(
    city: str,
    filter_key: tuple[tuple[str, object], ...],
    quantile: float = 0.20,
    min_candidates: int = 30,
) -> tuple[float | None, int]:
    df = get_bundle()["master"].get(city, pd.DataFrame())
    if df.empty:
        return None, 0
    return recommendation_price_floor(
        df,
        _recommendation_filters_from_key(filter_key),
        quantile=quantile,
        min_candidates=min_candidates,
    )


@st.cache_data(show_spinner=False, max_entries=400)
def cached_recommendations(
    city: str,
    filter_key: tuple[tuple[str, object], ...],
    limit: int,
) -> pd.DataFrame:
    df = get_bundle()["master"].get(city, pd.DataFrame())
    if df.empty:
        return pd.DataFrame()
    return get_recommendations(df, limit=limit, **_recommendation_filters_from_key(filter_key))


@st.cache_data(show_spinner=False, max_entries=300)
def cached_filtered_listings(city: str, filter_key: tuple[tuple[str, object], ...]) -> pd.DataFrame:
    df = get_bundle()["master"].get(city, pd.DataFrame())
    if df.empty:
        return pd.DataFrame()
    return filter_listings(df, **_recommendation_filters_from_key(filter_key))


@st.cache_data(show_spinner=False, max_entries=16)
def cached_market_summary(city: str) -> dict:
    df = get_bundle()["master"].get(city, pd.DataFrame())
    return market_summary(df)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_neighbourhood_summary(city: str, min_rows: int = 25) -> pd.DataFrame:
    df = get_bundle()["master"].get(city, pd.DataFrame())
    return neighbourhood_summary(df, min_rows=min_rows)


@st.cache_data(show_spinner=False)
def cached_city_comparison() -> pd.DataFrame:
    return compare_cities(get_bundle()["master"])


@st.cache_data(show_spinner=False, max_entries=300)
def cached_retrieved_context(question: str, city: str | None, top_k: int = 5) -> pd.DataFrame:
    return retrieve_context(question, get_rag_index(), city=city, top_k=top_k)


def city_dataframe(master_data: dict[str, pd.DataFrame], city_choice: str) -> pd.DataFrame:
    if city_choice == "Compare both":
        frames = [df for df in master_data.values() if not df.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return master_data.get(city_choice, pd.DataFrame())


def default_filters(master_data: dict[str, pd.DataFrame]) -> dict:
    cities = available_cities(master_data)
    city_choice = "Compare both" if len(cities) > 1 else (cities[0] if cities else "Compare both")

    combined = city_dataframe(master_data, city_choice)
    price_max = int(max(50, min(500, combined["price_eur"].quantile(0.95)))) if not combined.empty else 250

    budget_max = min(150, max(price_max, 100))
    guests = 2
    room_type = "Any"
    neighbourhood = "Any"
    amenities = []
    min_rating = 4.0
    max_distance_km = 15
    min_available_days_30 = 0

    min_prediction_date, max_prediction_date = _price_prediction_date_bounds()
    fallback_start = date.today() + timedelta(days=30)
    if min_prediction_date and max_prediction_date:
        default_check_in = max(fallback_start, min_prediction_date)
        if default_check_in >= max_prediction_date:
            default_check_in = min_prediction_date
        default_check_out = min(default_check_in + timedelta(days=3), max_prediction_date)
    else:
        default_check_in = fallback_start
        default_check_out = fallback_start + timedelta(days=3)

    return {
        "city_choice": city_choice,
        "budget_max": budget_max,
        "guests": guests,
        "room_type": room_type,
        "neighbourhood": neighbourhood,
        "amenities": amenities,
        "min_rating": min_rating,
        "max_distance_km": max_distance_km,
        "min_available_days_30": min_available_days_30,
        "check_calendar_dates": False,
        "check_in": default_check_in,
        "check_out": default_check_out,
    }


@st.cache_resource(show_spinner=False)
def precompute_common_recommendation_cache() -> bool:
    """Warm common recommendation cache entries used by demo questions and tabs."""
    master_data = get_bundle()["master"]
    if not master_data:
        return False

    filters = default_filters(master_data)
    base_filters = {
        "budget_max": filters["budget_max"],
        "guests": filters["guests"],
        "room_type": "Any",
        "neighbourhood": "Any",
        "amenities": [],
        "min_rating": filters["min_rating"],
        "max_distance_km": filters["max_distance_km"],
        "min_available_days_30": 1,
        "min_reviews": 5,
    }
    common_profiles = [
        {},
        {"budget_max": 120},
        {"max_distance_km": 5, "amenities": ["Air conditioning"]},
    ]

    for city, df in master_data.items():
        if df.empty:
            continue
        for profile in common_profiles:
            recommendation_filters = {**base_filters, **profile}
            price_floor, _ = cached_recommendation_price_floor(
                city,
                _recommendation_filter_key(recommendation_filters),
                quantile=0.20,
            )
            if price_floor:
                recommendation_filters["min_price_eur"] = price_floor
            cached_recommendations(
                city,
                _recommendation_filter_key(recommendation_filters),
                limit=10,
            )

    return True


def metric_row(summary: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏠 Listings", f"{summary.get('listings', 0):,}")
    col2.metric("💰 Median price", format_eur(summary.get("median_price_eur")))
    col3.metric("⭐ Mean rating", summary.get("mean_rating", "n/a"))
    col4.metric("📅 Median availability", summary.get("median_availability_30", "n/a"))




def render_kpi_card(label: str, value: str | int | float, note: str = "", icon: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{icon} {label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_value(row: pd.Series, column: str, default: str = "n/a") -> str:
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return str(value)


RECOMMENDATION_STYLE_CONFIG = {
    "Budget": {
        "price_floor_quantile": 0.10,
        "min_reviews": 5,
        "min_rating": None,
        "note": "Budget mode keeps the lowest sensible price floor, then ranks by affordability, rating, and location.",
    },
    "In Style": {
        "price_floor_quantile": 0.40,
        "min_reviews": 10,
        "min_rating": 4.4,
        "note": "In Style mode avoids the cheapest tail and prioritises value rating, overall rating, amenities, and location.",
    },
    "Luxury": {
        "price_floor_quantile": 0.75,
        "min_reviews": 20,
        "min_rating": 4.7,
        "room_type": "Entire home/apt",
        "min_luxury_amenities": 4,
        "amenity_floor_quantile": 0.75,
        "note": "Luxury mode focuses on entire properties with stronger review confidence and upper-tier amenity coverage.",
    },
}


def recommendation_style_config(style: str) -> dict:
    return RECOMMENDATION_STYLE_CONFIG.get(style, RECOMMENDATION_STYLE_CONFIG["In Style"])


def parse_recommendation_distance_cap(label: str | None) -> int | None:
    if not label or label == "Any":
        return None
    try:
        return int(str(label).split()[0])
    except (TypeError, ValueError):
        return None


def prepare_recommendation_display(
    recs: pd.DataFrame,
    rank_mode: str,
    distance_cap_km: int | None,
) -> tuple[pd.DataFrame, bool]:
    """Apply lightweight display-only ordering and distance filtering."""
    if recs.empty:
        return recs, False

    display = recs.copy()
    distance_filtered = False

    if distance_cap_km is not None and "distance_to_center_km" in display.columns:
        distances = pd.to_numeric(display["distance_to_center_km"], errors="coerce")
        within_distance = display[distances <= float(distance_cap_km)].copy()
        if not within_distance.empty:
            display = within_distance
            distance_filtered = True
        else:
            rank_mode = "Closest centre"

    if rank_mode == "Lowest price" and "price_eur" in display.columns:
        display["_sort_price_eur"] = pd.to_numeric(display["price_eur"], errors="coerce")
        display["_sort_value_score"] = pd.to_numeric(display.get("value_score"), errors="coerce")
        display = display.sort_values(
            ["_sort_price_eur", "_sort_value_score"],
            ascending=[True, False],
            na_position="last",
        )
    elif rank_mode == "Closest centre" and "distance_to_center_km" in display.columns:
        display["_sort_distance_km"] = pd.to_numeric(display["distance_to_center_km"], errors="coerce")
        display["_sort_value_score"] = pd.to_numeric(display.get("value_score"), errors="coerce")
        display = display.sort_values(
            ["_sort_distance_km", "_sort_value_score"],
            ascending=[True, False],
            na_position="last",
        )

    return display.drop(columns=[col for col in display.columns if col.startswith("_sort_")], errors="ignore"), distance_filtered


def render_recommendation_controls(style_note: str) -> None:
    control_left, control_mid, control_right = st.columns([1.1, 1.8, 1.1], gap="medium", vertical_alignment="top")
    with control_left:
        st.pills(
            "Recommendation category",
            ["Budget", "In Style", "Luxury"],
            selection_mode="single",
            key="recommendation_style",
        )
    with control_mid:
        st.pills(
            "Rank by",
            ["Best match", "Lowest price", "Closest centre"],
            selection_mode="single",
            key="recommendation_rank_mode",
        )
    with control_right:
        st.selectbox(
            "Distance",
            ["Any", "1 km", "2 km", "3 km", "4 km", "5 km"],
            key="recommendation_distance_cap",
        )
    st.caption(style_note)


def category_amenity_floor(df: pd.DataFrame, filters: dict, style_config: dict) -> int | None:
    quantile = style_config.get("amenity_floor_quantile")
    if quantile is None or "amenities_count" not in df.columns:
        return None

    candidate_filters = {
        key: value
        for key, value in filters.items()
        if key not in {"min_price_eur", "min_amenities_count", "recommendation_style"}
    }
    candidates = filter_listings(df, **candidate_filters)
    amenities = pd.to_numeric(candidates.get("amenities_count"), errors="coerce").dropna()
    if len(amenities) < 30:
        return None

    floor = amenities.quantile(float(quantile))
    if not pd.notna(floor):
        return None
    return max(1, int(round(float(floor))))


def neighbourhood_average_distance(
    city_df: pd.DataFrame,
    neighbourhood: str | None,
    fallback_distance: float,
) -> tuple[float, int]:
    if city_df.empty or "distance_to_center_km" not in city_df.columns:
        return fallback_distance, 0

    scope = city_df
    if neighbourhood and "neighbourhood_cleansed" in city_df.columns:
        scope = city_df[city_df["neighbourhood_cleansed"].astype(str) == str(neighbourhood)]

    distances = pd.to_numeric(scope.get("distance_to_center_km"), errors="coerce").dropna()
    if distances.empty:
        return fallback_distance, 0

    return float(distances.mean()), int(len(distances))


def render_neighbourhood_distance_card(distance_km: float, listing_count: int, neighbourhood: str | None) -> None:
    st.markdown(
        f"""
        <div class="prediction-location-card">
            <div class="prediction-location-value">{distance_km:.1f} km</div>
            <div class="prediction-location-label">Avg. distance to centre</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_recommendation_card(row: pd.Series, rank: int) -> None:
    """Render compact polished recommendation cards using Streamlit components.

    This keeps the previous elegant compact-card style while avoiding raw HTML
    rendering issues from st.markdown in some Streamlit/browser combinations.
    """
    neighbourhood = _safe_value(row, "neighbourhood_cleansed", "Unknown neighbourhood")
    room_type = _safe_value(row, "room_type", "Listing")
    property_group = _safe_value(row, "property_group", "Property")
    city = _safe_value(row, "city", "")

    price = format_eur(row.get("price_eur"))

    rating = row.get("review_scores_rating")
    rating_text = f"{float(rating):.2f}" if pd.notna(rating) else "n/a"

    value_score = row.get("value_score")
    value_text = f"{float(value_score):.1f}" if pd.notna(value_score) else "n/a"

    listing_id_raw = row.get("listing_id")
    listing_id_text = ""
    if pd.notna(listing_id_raw):
        if isinstance(listing_id_raw, float) and listing_id_raw.is_integer():
            listing_id_text = f"{listing_id_raw:.0f}"
        else:
            listing_id_text = str(listing_id_raw).strip()
        listing_id_text = listing_id_text.replace(",", "")
        if listing_id_text.endswith(".0"):
            listing_id_text = listing_id_text[:-2]

    distance = row.get("distance_to_center_km")
    distance_text = f"{float(distance):.1f} km from centre" if pd.notna(distance) else "Distance n/a"

    availability = row.get("availability_30")
    availability_text = f"{int(availability)} days next 30" if pd.notna(availability) else "Availability n/a"

    reviews = row.get("number_of_reviews")
    reviews_text = f"{int(reviews)} reviews" if pd.notna(reviews) else "Reviews n/a"

    reason = _safe_value(
        row,
        "recommendation_reason",
        "Strong balance of price, rating, location, amenities, and availability.",
    )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
                background: transparent;
            }}

            .card {{
                background: #FFFFFF;
                border: 1px solid #D8E3F0;
                border-left: 7px solid #00338D;
                border-radius: 18px;
                padding: 24px 28px;
                box-shadow: 0 7px 20px rgba(0, 51, 141, 0.08);
                min-height: 214px;
                box-sizing: border-box;
            }}

            .card-header {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 16px;
                margin-bottom: 14px;
            }}

            .title-block {{
                min-width: 0;
                flex: 1 1 auto;
            }}

            .title {{
                color: #001F5B;
                font-size: 22px;
                font-weight: 850;
                margin-bottom: 8px;
            }}

            .meta {{
                color: #64748B;
                font-size: 14px;
                margin-bottom: 0;
            }}

            .listing-id {{
                color: #64748B;
                font-size: 14px;
                font-weight: 650;
                text-align: right;
                white-space: normal;
                max-width: 46%;
                line-height: 1.35;
                box-sizing: border-box;
            }}

            .listing-id span {{
                color: #64748B;
                background: transparent;
                font-size: 14px;
                font-weight: 650;
                word-break: break-all;
            }}

            .badges {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-bottom: 14px;
            }}

            .badge {{
                display: inline-flex;
                align-items: center;
                gap: 5px;
                background: #EAF4FF;
                color: #00338D;
                border: 1px solid #BFE4FF;
                border-radius: 999px;
                padding: 8px 13px;
                font-size: 13px;
                font-weight: 800;
                line-height: 1;
                white-space: nowrap;
            }}

            .badge.primary {{
                background: linear-gradient(135deg, #00338D 0%, #005EB8 100%);
                color: #FFFFFF;
                border-color: #00338D;
            }}

            .reason {{
                color: #475569;
                font-size: 15px;
                line-height: 1.45;
                margin-top: 6px;
            }}

            .reason strong {{
                color: #001F5B;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="card-header">
                <div class="title-block">
                    <div class="title">#{rank} · 📍 {neighbourhood}</div>
                    <div class="meta">{city} · {room_type} · {property_group}</div>
                </div>
                <div class="listing-id">Listing ID <span>{escape(listing_id_text) if listing_id_text else "n/a"}</span></div>
            </div>

            <div class="badges">
                <span class="badge primary">Value score {value_text}</span>
                <span class="badge">💰 {price} / night</span>
                <span class="badge">⭐ {rating_text}</span>
                <span class="badge">{reviews_text}</span>
                <span class="badge">📍 {distance_text}</span>
                <span class="badge">📅 {availability_text}</span>
            </div>

            <div class="reason">
                <strong>Why recommended:</strong> {reason}
            </div>
        </div>
    </body>
    </html>
    """

    components.html(html, height=292, scrolling=False)

def recommendation_view(master_data: dict[str, pd.DataFrame], filters: dict) -> None:
    st.subheader("Recommendations & Destination Comparison")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        The advisor ranks listings using an explainable value score based on
        **price, rating, value rating, distance to centre, amenities, and availability**.
        Use this view to compare two possible holiday destinations side by side.
        """
    )

    city_options = [city for city, df in master_data.items() if not df.empty]
    if not city_options:
        st.warning("No city datasets are available for recommendations.")
        return

    if st.session_state.get("recommendation_compare_city_a") not in city_options:
        st.session_state["recommendation_compare_city_a"] = city_options[0]
    default_city_b = city_options[1] if len(city_options) > 1 else city_options[0]
    if st.session_state.get("recommendation_compare_city_b") not in city_options:
        st.session_state["recommendation_compare_city_b"] = default_city_b

    st.markdown("### Top Recommendations by City")

    destination_cols = st.columns(2, gap="medium")
    with destination_cols[0]:
        compare_city_a = st.selectbox(
            "Destination 1",
            city_options,
            key="recommendation_compare_city_a",
            label_visibility="collapsed",
        )
    with destination_cols[1]:
        compare_city_b = st.selectbox(
            "Destination 2",
            city_options,
            key="recommendation_compare_city_b",
            label_visibility="collapsed",
        )

    if st.session_state.get("recommendation_style") == "Value":
        st.session_state["recommendation_style"] = "In Style"
    if st.session_state.get("recommendation_style") not in RECOMMENDATION_STYLE_CONFIG:
        st.session_state["recommendation_style"] = "In Style"
    if st.session_state.get("recommendation_rank_mode") not in {"Best match", "Lowest price", "Closest centre"}:
        st.session_state["recommendation_rank_mode"] = "Best match"
    if st.session_state.get("recommendation_distance_cap") not in {"Any", "1 km", "2 km", "3 km", "4 km", "5 km"}:
        st.session_state["recommendation_distance_cap"] = "Any"

    recommendation_style = str(st.session_state["recommendation_style"])
    rank_mode = str(st.session_state["recommendation_rank_mode"])
    distance_cap_label = str(st.session_state["recommendation_distance_cap"])
    distance_cap_km = parse_recommendation_distance_cap(distance_cap_label)
    style_config = recommendation_style_config(recommendation_style)

    selected_cities = list(dict.fromkeys([compare_city_a, compare_city_b]))
    if len(selected_cities) == 1 and len(city_options) > 1:
        st.info("Select two different destinations to compare them side by side.")

    city_recommendations: dict[str, pd.DataFrame] = {}
    city_quality_notes: dict[str, str] = {}

    for city in selected_cities:
        df = master_data.get(city, pd.DataFrame())
        if df.empty:
            continue

        recommendation_filters = {
            "budget_max": filters["budget_max"],
            "guests": filters["guests"],
            "room_type": style_config.get("room_type", filters["room_type"]),
            "neighbourhood": filters["neighbourhood"] if len(selected_cities) == 1 else "Any",
            "amenities": filters["amenities"],
            "min_rating": max(
                float(filters["min_rating"] or 0),
                float(style_config["min_rating"] or 0),
            ),
            "max_distance_km": filters["max_distance_km"],
            "min_available_days_30": filters["min_available_days_30"],
            "min_reviews": int(style_config["min_reviews"]),
        }
        if style_config.get("min_luxury_amenities"):
            recommendation_filters["min_luxury_amenities"] = int(style_config["min_luxury_amenities"])

        if not filters["check_calendar_dates"]:
            recommendation_filters["min_available_days_30"] = max(
                int(recommendation_filters["min_available_days_30"] or 0),
                1,
            )

        amenity_floor = category_amenity_floor(df, recommendation_filters, style_config)
        if amenity_floor:
            recommendation_filters["min_amenities_count"] = amenity_floor

        recommendation_filter_key = _recommendation_filter_key(recommendation_filters)
        price_floor, candidate_count = cached_recommendation_price_floor(
            city,
            recommendation_filter_key,
            quantile=float(style_config["price_floor_quantile"]),
        )
        if price_floor:
            recommendation_filters["min_price_eur"] = price_floor

        styled_recommendation_filters = {
            **recommendation_filters,
            "recommendation_style": recommendation_style,
        }
        recommendation_limit = 200 if filters["check_calendar_dates"] else 10
        recs = cached_recommendations(
            city,
            _recommendation_filter_key(styled_recommendation_filters),
            limit=recommendation_limit,
        )

        if recs.empty:
            fallback_filters = {
                key: value
                for key, value in styled_recommendation_filters.items()
                if key not in {"min_price_eur", "min_reviews", "min_amenities_count", "min_luxury_amenities"}
            }
            recs = cached_recommendations(
                city,
                _recommendation_filter_key(fallback_filters),
                limit=recommendation_limit,
            )
            city_quality_notes[city] = "Quality checks were relaxed because they removed every matching listing."
        else:
            floor_text = format_eur(price_floor) if price_floor else "not applied"
            amenity_text = (
                f", minimum amenities {amenity_floor}"
                if amenity_floor and recommendation_style == "Luxury"
                else ""
            )
            room_text = ", entire home/apt only" if recommendation_style == "Luxury" else ""
            city_quality_notes[city] = (
                f"{recommendation_style} category selected. Quality checks applied: at least "
                f"{style_config['min_reviews']} reviews"
                + (", at least 1 available day in the next 30" if not filters["check_calendar_dates"] else "")
                + room_text
                + amenity_text
                + f", price floor {floor_text} from the filtered candidate pool."
            )

        if filters["check_calendar_dates"] and calendar_cache_exists() and not recs.empty:
            availability = availability_for_listings(
                city=city,
                listing_ids=recs["listing_id"],
                check_in=filters["check_in"],
                check_out=filters["check_out"],
            )
            recs = recs.merge(availability, on="listing_id", how="left")
            recs = recs[recs["is_available"].fillna(False)].copy()
            nights = max((filters["check_out"] - filters["check_in"]).days, 0)
            recs["estimated_stay_total_eur"] = (pd.to_numeric(recs["price_eur"], errors="coerce") * nights).round(2)
            recs["nightly_price_source"] = "listing snapshot price; calendar price blank"

            if not recs.empty and city in available_price_model_cities():
                predicted_prices = []
                for _, row in recs.iterrows():
                    prediction = predict_listing_price(city, row)
                    predicted_prices.append(prediction["predicted_nightly_price_eur"])
                recs["predicted_nightly_price_eur"] = predicted_prices
                recs["predicted_stay_total_eur"] = (recs["predicted_nightly_price_eur"] * nights).round(2)
            recs = recs.head(10)

        city_recommendations[city] = recs

    recommendations = pd.concat(city_recommendations.values(), ignore_index=True) if city_recommendations else pd.DataFrame()

    if recommendations.empty:
        st.warning("No listings matched the current filters. Try widening the budget, distance, rating, or availability filters.")
        return

    if filters["check_calendar_dates"]:
        st.info(
            "Recommendations are filtered using the local calendar snapshot. "
            "The calendar confirms availability, but its future price columns are blank, "
            "so displayed prices use the listing snapshot price."
        )

    if len(city_recommendations) > 1:
        prepared_city_recommendations: dict[str, dict[str, object]] = {}
        city_columns = st.columns(len(city_recommendations))
        for city_column, (city, recs) in zip(city_columns, city_recommendations.items()):
            with city_column:
                display_recs, distance_filter_applied = prepare_recommendation_display(
                    recs,
                    rank_mode=rank_mode,
                    distance_cap_km=distance_cap_km,
                )
                shown_recs = display_recs.head(5)
                market_df = master_data.get(city, pd.DataFrame())
                recommended_median = pd.to_numeric(shown_recs.get("price_eur"), errors="coerce").median()
                market_median = pd.to_numeric(market_df.get("price_eur"), errors="coerce").median()
                avg_score = pd.to_numeric(shown_recs.get("value_score"), errors="coerce").mean()
                prepared_city_recommendations[city] = {
                    "shown_recs": shown_recs,
                    "avg_score": avg_score,
                    "distance_filter_applied": distance_filter_applied,
                }

                metric_cols = st.columns(3)
                with metric_cols[0]:
                    render_kpi_card("Shown", f"{len(shown_recs):,}", "Recommendation cards", "🏠")
                with metric_cols[1]:
                    render_kpi_card("Shortlist median", format_eur(recommended_median), "Recommended set", "💰")
                with metric_cols[2]:
                    render_kpi_card("Market median", format_eur(market_median), "Full city market", "📊")

        render_recommendation_controls(style_config["note"])

        card_columns = st.columns(len(city_recommendations))
        for city_column, city in zip(card_columns, city_recommendations.keys()):
            with city_column:
                prepared = prepared_city_recommendations.get(city, {})
                shown_recs = prepared.get("shown_recs", pd.DataFrame())
                avg_score = prepared.get("avg_score")
                distance_filter_applied = bool(prepared.get("distance_filter_applied", False))
                st.caption(
                    f"Average value score: {avg_score:.1f}. The recommendation median is expected to sit below "
                    "the full market median because the list is value-ranked under the active filters."
                    if pd.notna(avg_score)
                    else "The recommendation median is calculated from the value-ranked shortlist under the active filters."
                )
                st.caption(city_quality_notes.get(city, ""))

                if distance_cap_km is not None and not distance_filter_applied:
                    st.caption(
                        f"No {city} shortlist cards were available within {distance_cap_km} km, "
                        "so the closest available recommendations are shown."
                    )

                for idx, (_, row) in enumerate(shown_recs.iterrows(), start=1):
                    render_recommendation_card(row, idx)

        return

    recommendations, distance_filter_applied = prepare_recommendation_display(
        recommendations,
        rank_mode=rank_mode,
        distance_cap_km=distance_cap_km,
    )
    shown_recommendations = recommendations.head(8)

    col1, col2, col3 = st.columns(3)
    with col1:
        render_kpi_card("Shown listings", f"{len(shown_recommendations):,}", "Recommendation cards", "🏠")
    with col2:
        median_price = pd.to_numeric(shown_recommendations.get("price_eur"), errors="coerce").median()
        render_kpi_card("Shortlist median", format_eur(median_price), "Recommended set", "💰")
    with col3:
        avg_score = pd.to_numeric(shown_recommendations.get("value_score"), errors="coerce").mean()
        render_kpi_card("Avg. value score", f"{avg_score:.1f}" if pd.notna(avg_score) else "n/a", "Higher is better", "📈")

    render_recommendation_controls(style_config["note"])

    if distance_cap_km is not None and not distance_filter_applied:
        st.caption(
            f"No shortlist cards were available within {distance_cap_km} km, "
            "so the closest available recommendations are shown."
        )

    st.markdown("### 🏆 Top Recommendations")
    for idx, (_, row) in enumerate(shown_recommendations.iterrows(), start=1):
        render_recommendation_card(row, idx)


def render_calendar_cache_coverage(metadata: pd.DataFrame) -> None:
    """Render calendar cache metadata without the default dark dataframe styling."""
    rows = []
    for _, row in metadata.iterrows():
        city = escape(str(row.get("city", "n/a")))
        source_file = escape(Path(str(row.get("source_file", ""))).name or "n/a")
        min_date = escape(str(row.get("min_date", "n/a")))
        max_date = escape(str(row.get("max_date", "n/a")))
        total_rows = pd.to_numeric(pd.Series([row.get("total_rows")]), errors="coerce").iloc[0]
        available_rows = pd.to_numeric(pd.Series([row.get("available_rows")]), errors="coerce").iloc[0]
        price_rows = pd.to_numeric(pd.Series([row.get("price_rows")]), errors="coerce").iloc[0]
        built_at = escape(str(row.get("built_at", "n/a")))

        rows.append(
            "<tr>"
            f"<td>{city}</td>"
            f"<td>{source_file}</td>"
            f"<td>{min_date} to {max_date}</td>"
            f"<td>{int(total_rows):,}</td>"
            f"<td>{int(available_rows):,}</td>"
            f"<td>{int(price_rows):,}</td>"
            f"<td>{built_at}</td>"
            "</tr>"
        )

    st.markdown(
        f"""
        <div class="overview-table-wrap">
            <table class="overview-table">
                <thead>
                    <tr>
                        <th>City</th>
                        <th>Source</th>
                        <th>Date coverage</th>
                        <th>Calendar rows</th>
                        <th>Available rows</th>
                        <th>Price rows</th>
                        <th>Built at</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def calendar_view(master_data: dict[str, pd.DataFrame], filters: dict) -> None:
    st.subheader("Local Future Calendar Check")

    metadata = calendar_metadata()
    if metadata.empty:
        st.warning("Calendar cache has not been built yet. Run `python build_calendar_cache.py` from the chatbot folder.")
        return

    city_options = list(master_data.keys())
    default_city = filters["city_choice"] if filters["city_choice"] in city_options else city_options[0]
    default_city_index = city_options.index(default_city)

    control_cols = st.columns([1.1, 1.35, 1, 1, 0.95])
    with control_cols[0]:
        city = st.selectbox("City to check", city_options, index=default_city_index)

    city_df = master_data.get(city, pd.DataFrame())
    default_listing_id = int(city_df["listing_id"].iloc[0]) if not city_df.empty and "listing_id" in city_df.columns else 0

    with control_cols[1]:
        listing_id_text = st.text_input("Listing ID", value=str(default_listing_id), key="calendar_listing_id")
    with control_cols[2]:
        check_in = st.date_input("Check-in", value=filters["check_in"], key="calendar_check_in")
    with control_cols[3]:
        check_out = st.date_input("Check-out", value=filters["check_out"], key="calendar_check_out")
    with control_cols[4]:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_calendar_check = st.button("Check availability", use_container_width=True)

    if run_calendar_check:
        try:
            listing_id = int(str(listing_id_text).strip())
        except ValueError:
            st.error("Please enter a valid numeric listing ID.")
            listing_id = None

    if run_calendar_check and listing_id is not None:
        result = check_listing_availability(city, listing_id, check_in, check_out)
        listing = city_df[city_df["listing_id"] == int(listing_id)].head(1) if not city_df.empty else pd.DataFrame()

        st.write(result["status"])
        metric_cols = st.columns(4)
        metric_cols[0].metric("Requested nights", result.get("nights", 0))
        metric_cols[1].metric("Available nights", result.get("available_nights", 0))
        metric_cols[2].metric("Calendar available", "Yes" if result.get("is_available") else "No")
        metric_cols[3].metric("Calendar price present", "No")

        if result.get("available_nights", 0) >= result.get("nights", 0) and not result.get("is_available"):
            st.info(
                "The requested dates are present in the local calendar snapshot, "
                "but the listing's minimum or maximum stay rule prevents this exact booking."
            )

        if not listing.empty:
            nightly_price = pd.to_numeric(listing["price_eur"], errors="coerce").iloc[0]
            total = nightly_price * result.get("nights", 0)
            st.markdown(
                f"Snapshot nightly price for this listing: **{format_eur(nightly_price)}**. "
                f"Estimated stay total using snapshot price: **{format_eur(total)}**."
            )

            render_overview_table(
                listing,
                {
                    "listing_id": "Listing ID",
                    "city": "City",
                    "neighbourhood_cleansed": "Neighbourhood",
                    "room_type": "Room type",
                    "property_group": "Property group",
                    "price_eur": "Snapshot nightly price",
                    "accommodates": "Guests",
                    "review_scores_rating": "Rating",
                    "availability_30": "Days available next 30",
                    "availability_365": "Days available next 365",
                },
            )

        st.caption(
            "The local calendar snapshot does not contain date-specific prices in the raw price fields. "
            "This checker verifies availability first; use Price Check or the AI Advisor for model-backed future price estimates."
        )

def live_api_view(master_data: dict[str, pd.DataFrame], filters: dict) -> None:
    st.subheader("Optional Live API Check")

    status = api_status()
    col1, col2, col3 = st.columns(3)
    col1.metric("Provider", status["provider"])
    col2.metric("Configured", "Yes" if status["configured"] else "No")
    col3.metric("Endpoint", status["future_rates_endpoint"])

    st.caption(
        "This connector is scaffolded for AirROI's future-rates endpoint. "
        "It only runs when an API key is available, so the rest of the app remains fully local."
    )

    if not status["configured"]:
        st.info(
            "Set an `AIRROI_API_KEY` environment variable to enable live future-rate checks. "
            "Until then, the app uses local calendar availability plus model prediction."
        )

    city_options = list(master_data.keys())
    default_city = filters["city_choice"] if filters["city_choice"] in city_options else city_options[0]
    city = st.selectbox("API check city", city_options, index=city_options.index(default_city))
    city_df = master_data.get(city, pd.DataFrame())

    default_listing_id = int(city_df["listing_id"].iloc[0]) if not city_df.empty and "listing_id" in city_df.columns else 0
    listing_id = st.number_input("Airbnb listing ID", min_value=0, value=default_listing_id, step=1, key="api_listing_id")

    col_a, col_b = st.columns(2)
    with col_a:
        check_in = st.date_input("API check-in", value=filters["check_in"], key="api_check_in")
    with col_b:
        check_out = st.date_input("API check-out", value=filters["check_out"], key="api_check_out")

    currency = st.selectbox("Currency", ["native", "usd"], index=0)

    if st.button("Run live API check"):
        try:
            live = summarize_airroi_stay(int(listing_id), check_in, check_out, currency=currency)
        except ApiNotConfiguredError as exc:
            st.warning(str(exc))
            return
        except ApiRequestError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Live API check failed: {exc}")
            return

        st.write(live["status"])
        metric_cols = st.columns(4)
        metric_cols[0].metric("Live available", "Yes" if live.get("is_available") else "No")
        metric_cols[1].metric("Has live price", "Yes" if live.get("has_live_price") else "No")
        metric_cols[2].metric("Avg nightly", live.get("average_nightly_rate") or "n/a")
        metric_cols[3].metric("Stay total", live.get("total_rate") or "n/a")

        rates = live.get("rates")
        if isinstance(rates, pd.DataFrame) and not rates.empty:
            st.dataframe(rates, use_container_width=True, hide_index=True)


def _safe_float(value, fallback: float) -> float:
    try:
        if pd.isna(value):
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_int(value, fallback: int, minimum: int, maximum: int) -> int:
    try:
        if pd.isna(value):
            cleaned = fallback
        else:
            cleaned = int(round(float(value)))
    except (TypeError, ValueError):
        cleaned = fallback
    return max(minimum, min(maximum, cleaned))


def _option_index(options: list[str], default_value: object | None) -> int:
    if default_value in options:
        return options.index(default_value)
    return 0


def _default_bool(defaults: dict, column: str, fallback: bool) -> bool:
    value = defaults.get(column, int(fallback))
    try:
        if pd.isna(value):
            return fallback
        return bool(round(float(value)))
    except (TypeError, ValueError):
        return fallback


def _metric_text(value: object, decimals: int = 2) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "n/a"


def render_overview_table(df: pd.DataFrame, columns: dict[str, str]) -> None:
    display_df = df[list(columns.keys())].rename(columns=columns).copy()

    def format_value(value: object) -> str:
        if pd.isna(value):
            return "n/a"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if abs(number - round(number)) < 0.005:
                return f"{number:,.0f}"
            return f"{number:,.2f}"
        return str(value)

    header_cells = "".join(f"<th>{escape(str(column))}</th>" for column in display_df.columns)
    body_rows = "\n".join(
        "<tr>"
        + "".join(f"<td>{escape(format_value(value))}</td>" for value in row)
        + "</tr>"
        for row in display_df.itertuples(index=False, name=None)
    )

    st.markdown(
        f"""
        <div class="overview-table-wrap">
            <table class="overview-table">
                <thead><tr>{header_cells}</tr></thead>
                <tbody>{body_rows}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _pretty_column_label(column: str) -> str:
    labels = {
        "scope": "Scope",
        "city": "City",
        "rank_within_city": "Rank within city",
        "model": "Model",
        "train_rows": "Train rows",
        "test_rows": "Test rows",
        "numeric_features": "Numeric features",
        "categorical_features": "Categorical features",
        "rmse_eur": "RMSE EUR",
        "mae_eur": "MAE EUR",
        "r2": "R2",
        "feature": "Feature",
        "importance": "Importance",
        "segment": "Segment",
        "rows": "Rows",
        "median_abs_error_eur": "Median abs error EUR",
        "p90_abs_error_eur": "P90 abs error EUR",
        "mean_actual_price_eur": "Mean actual price EUR",
        "segment_type": "Segment type",
    }
    return labels.get(column, column.replace("_", " ").title())


def render_theme_table(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    if df.empty:
        return

    selected_columns = [column for column in (columns or list(df.columns)) if column in df.columns]
    if not selected_columns:
        return

    render_overview_table(
        df[selected_columns].copy(),
        {column: _pretty_column_label(column) for column in selected_columns},
    )


def render_city_model_leaderboard(leaderboard: pd.DataFrame, limit_per_city: int = 3) -> None:
    if leaderboard.empty:
        return

    display = leaderboard.copy()
    for column in ["rank_within_city", "rmse_eur", "mae_eur", "r2"]:
        if column in display.columns:
            display[column] = pd.to_numeric(display[column], errors="coerce")

    sort_columns = ["city"]
    sort_ascending = [True]
    if "rank_within_city" in display.columns:
        sort_columns.append("rank_within_city")
        sort_ascending.append(True)
    if "rmse_eur" in display.columns:
        sort_columns.append("rmse_eur")
        sort_ascending.append(True)
    display = display.sort_values(sort_columns, ascending=sort_ascending)

    if "rank_within_city" not in display.columns:
        display["rank_within_city"] = display.groupby("city").cumcount() + 1
    display["rank_within_city"] = display["rank_within_city"].fillna(
        display.groupby("city").cumcount() + 1
    ).astype(int)

    display = display.groupby("city", group_keys=False).head(limit_per_city).copy()

    columns = [
        "city",
        "rank_within_city",
        "model",
        "rmse_eur",
        "mae_eur",
        "r2",
        "train_rows",
        "test_rows",
        "numeric_features",
        "categorical_features",
        "scope",
    ]
    selected_columns = [column for column in columns if column in display.columns]

    def format_model_value(column: str, value: object) -> str:
        if pd.isna(value):
            return "n/a"
        if column == "rank_within_city":
            rank = int(value)
            return "#1 Best" if rank == 1 else f"#{rank}"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if column in {"rmse_eur", "mae_eur", "r2"}:
                return f"{number:,.2f}"
            if abs(number - round(number)) < 0.005:
                return f"{number:,.0f}"
            return f"{number:,.2f}"
        return str(value)

    header_cells = "".join(f"<th>{escape(_pretty_column_label(column))}</th>" for column in selected_columns)
    body_rows = []
    for _, row in display.iterrows():
        row_class = "best-model-row" if int(row.get("rank_within_city", 0)) == 1 else ""
        cells = "".join(
            f"<td>{escape(format_model_value(column, row[column]))}</td>"
            for column in selected_columns
        )
        body_rows.append(f'<tr class="{row_class}">{cells}</tr>')

    st.markdown(
        f"""
        <div class="overview-table-wrap">
            <table class="overview-table">
                <thead><tr>{header_cells}</tr></thead>
                <tbody>{''.join(body_rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_table(recommendations: pd.DataFrame) -> None:
    if recommendations.empty:
        st.info("No recommendations to display.")
        return

    columns = {
        "listing_id": "Listing ID",
        "city": "City",
        "neighbourhood_cleansed": "Neighbourhood",
        "room_type": "Room type",
        "property_group": "Property group",
        "price_eur": "Nightly price",
        "accommodates": "Guests",
        "bedrooms": "Bedrooms",
        "bathrooms": "Bathrooms",
        "review_scores_rating": "Rating",
        "review_scores_value": "Value rating",
        "number_of_reviews": "Reviews",
        "availability_30": "Days next 30",
        "distance_to_center_km": "Km to centre",
        "value_score": "Value score",
        "recommendation_reason": "Why recommended",
    }
    available_columns = {column: label for column, label in columns.items() if column in recommendations.columns}
    render_overview_table(recommendations, available_columns)


def render_price_source_cards(cards: list[dict[str, str]]) -> None:
    if not cards:
        return

    card_html = []
    for card in cards:
        status = str(card.get("status", "neutral")).lower()
        if status not in {"good", "warning", "neutral"}:
            status = "neutral"
        title = escape(str(card.get("title", "")))
        value = escape(str(card.get("value", "")))
        detail = escape(str(card.get("detail", "")))
        card_html.append(
            f'<div class="price-source-card price-source-{status}">'
            f'<div class="price-source-title">{title}</div>'
            f'<div class="price-source-value">{value}</div>'
            f'<div class="price-source-detail">{detail}</div>'
            "</div>"
        )

    st.markdown(f'<div class="price-source-grid">{"".join(card_html)}</div>', unsafe_allow_html=True)


def _requested_month_starts(check_in: date, check_out: date) -> list[pd.Timestamp]:
    last_night = check_out - timedelta(days=1)
    if last_night < check_in:
        return []
    return list(pd.period_range(pd.Timestamp(check_in), pd.Timestamp(last_night), freq="M").to_timestamp())


def _format_month_window(months: list[pd.Timestamp]) -> str:
    if not months:
        return "requested period"
    labels = [month.strftime("%B %Y") for month in months]
    if len(labels) == 1:
        return labels[0]
    return f"{labels[0]} to {labels[-1]}"


def _calendar_period_status(city: str, check_in: date, check_out: date) -> tuple[bool, str]:
    min_date, max_date = date_bounds(city)
    if min_date is None or max_date is None:
        return False, "No local calendar snapshot is available for this city."

    last_night = check_out - timedelta(days=1)
    if check_in < min_date or last_night > max_date:
        return (
            False,
            f"Local calendar coverage for {city} runs from {min_date.isoformat()} to {max_date.isoformat()}.",
        )
    return True, "Requested dates are inside the local calendar snapshot."


def _calendar_result_for_listing(city: str, listing_id: int, check_in: date, check_out: date) -> dict:
    within_calendar, coverage_text = _calendar_period_status(city, check_in, check_out)
    nights = max((check_out - check_in).days, 0)
    if not within_calendar:
        return {
            "city": city,
            "listing_id": int(listing_id),
            "nights": nights,
            "available_nights": 0,
            "is_available": False,
            "stay_length_ok": False,
            "calendar_checked": False,
            "status": "Requested dates are outside the local calendar snapshot.",
            "coverage_text": coverage_text,
        }

    result = check_listing_availability(city, int(listing_id), check_in, check_out)
    result["calendar_checked"] = True
    result["coverage_text"] = coverage_text
    return result


def _calendar_result_for_candidates(
    city: str,
    listing_ids: pd.Series,
    check_in: date,
    check_out: date,
) -> dict:
    ids = pd.to_numeric(listing_ids, errors="coerce").dropna().astype("int64").drop_duplicates()
    nights = max((check_out - check_in).days, 0)
    if ids.empty:
        return {
            "calendar_checked": False,
            "candidate_count": 0,
            "available_count": 0,
            "available_ids": [],
            "nights": nights,
            "is_available": False,
            "status": "No matching listings were available for a calendar check.",
        }

    within_calendar, coverage_text = _calendar_period_status(city, check_in, check_out)
    if not within_calendar:
        return {
            "calendar_checked": False,
            "candidate_count": int(len(ids)),
            "available_count": 0,
            "available_ids": [],
            "nights": nights,
            "is_available": False,
            "status": "Requested dates are outside the local calendar snapshot.",
            "coverage_text": coverage_text,
        }

    availability = availability_for_listings(city, ids.tolist(), check_in, check_out)
    available_ids = (
        availability.loc[availability["is_available"].fillna(False), "listing_id"]
        .dropna()
        .astype("int64")
        .tolist()
    )
    available_count = len(available_ids)
    return {
        "calendar_checked": True,
        "candidate_count": int(len(ids)),
        "available_count": available_count,
        "available_ids": available_ids,
        "nights": nights,
        "is_available": available_count > 0,
        "status": (
            f"{available_count:,} of {len(ids):,} matching listings are available for the requested stay."
            if available_count
            else "No matching listings pass the exact calendar availability and stay-length rules."
        ),
        "coverage_text": coverage_text,
    }


def _precomputed_prediction_estimate(
    city: str,
    listing_ids: pd.Series | list[int],
    check_in: date,
    check_out: date,
) -> dict | None:
    monthly = load_monthly_prediction_cache()
    months = _requested_month_starts(check_in, check_out)
    if monthly.empty or not months:
        return None

    ids = pd.to_numeric(pd.Series(listing_ids), errors="coerce").dropna().astype("int64").drop_duplicates()
    if ids.empty:
        return None

    window = monthly[
        (monthly["city"].astype(str) == city)
        & (monthly["listing_id"].astype("int64").isin(ids.tolist()))
        & (monthly["month_start"].isin(months))
    ].copy()
    if window.empty:
        return None

    covered_months = set(window["month_start"].dropna())
    if not all(month in covered_months for month in months):
        return None

    price = pd.to_numeric(window["predicted_nightly_price_eur"], errors="coerce").median()
    if pd.isna(price):
        return None

    model_name = (
        window["prediction_model_name"].dropna().astype(str).mode().iloc[0]
        if "prediction_model_name" in window.columns and not window["prediction_model_name"].dropna().empty
        else "Precomputed model"
    )
    price_band = (
        window["predicted_price_band"].dropna().astype(str).mode().iloc[0]
        if "predicted_price_band" in window.columns and not window["predicted_price_band"].dropna().empty
        else "n/a"
    )
    segment = (
        window["prediction_model_segment"].dropna().astype(str).mode().iloc[0]
        if "prediction_model_segment" in window.columns and not window["prediction_model_segment"].dropna().empty
        else "n/a"
    )

    return {
        "city": city,
        "model_name": model_name,
        "predicted_nightly_price_eur": round(float(price), 2),
        "price_source": "precomputed_future_estimate",
        "price_source_label": "Precomputed future estimate",
        "price_band": price_band,
        "prediction_model_segment": segment,
        "cache_month_window": _format_month_window(months),
        "cache_listing_count": int(window["listing_id"].nunique()),
        "cache_row_count": int(len(window)),
    }


def _snapshot_price_estimate(
    city: str,
    price: object,
    *,
    label: str,
    model_name: str,
    listing_count: int | None = None,
) -> dict | None:
    numeric_price = pd.to_numeric(pd.Series([price]), errors="coerce").iloc[0]
    if pd.isna(numeric_price):
        return None

    result = {
        "city": city,
        "model_name": model_name,
        "predicted_nightly_price_eur": round(float(numeric_price), 2),
        "price_source": "snapshot_price",
        "price_source_label": label,
    }
    if listing_count is not None:
        result["snapshot_listing_count"] = int(listing_count)
    return result


def _custom_price_candidates(
    city_df: pd.DataFrame,
    *,
    accommodates: int,
    bedrooms: int,
    bathrooms: int,
    room_type: str,
    property_group: str,
    neighbourhood: str | None,
    selected_amenities: list[str],
    min_rating: float,
    min_available_days_30: int,
) -> pd.DataFrame:
    candidates = filter_listings(
        city_df,
        guests=accommodates,
        room_type=room_type,
        neighbourhood=neighbourhood,
        amenities=selected_amenities,
        min_rating=min_rating,
        min_available_days_30=min_available_days_30 if min_available_days_30 > 0 else None,
    )

    if property_group and "property_group" in candidates.columns:
        candidates = candidates[candidates["property_group"].astype(str) == str(property_group)].copy()
    if bedrooms > 0 and "bedrooms" in candidates.columns:
        candidates = candidates[pd.to_numeric(candidates["bedrooms"], errors="coerce").fillna(0) >= bedrooms].copy()
    if bathrooms > 0 and "bathrooms" in candidates.columns:
        candidates = candidates[pd.to_numeric(candidates["bathrooms"], errors="coerce").fillna(0) >= bathrooms].copy()
    return candidates


def _price_prediction_date_bounds() -> tuple[date | None, date | None]:
    calendar_min, calendar_max = date_bounds()
    monthly = load_monthly_prediction_cache()

    cache_min = cache_max = None
    if not monthly.empty and "month_start" in monthly.columns:
        cache_min = pd.to_datetime(monthly["month_start"], errors="coerce").min()
        cache_max = pd.to_datetime(monthly["month_start"], errors="coerce").max()
        if pd.notna(cache_min):
            cache_min = cache_min.date()
        if pd.notna(cache_max):
            cache_max = (cache_max + pd.offsets.MonthEnd(1)).date()

    min_candidates = [value for value in [calendar_min, cache_min] if value is not None]
    max_candidates = [value for value in [calendar_max, cache_max, date.today() + timedelta(days=730)] if value is not None]
    return (min(min_candidates) if min_candidates else None, max(max_candidates) if max_candidates else None)


def render_price_prediction_result(
    prediction: dict,
    nights: int,
    scenario_rows: list[dict[str, str]],
    availability: dict | None = None,
    source_cards: list[dict[str, str]] | None = None,
) -> None:
    price = _safe_float(prediction.get("predicted_nightly_price_eur"), 0.0)
    model_name = str(prediction.get("model_name", "Price model"))
    city = str(prediction.get("city", "Selected city"))
    result_label = str(prediction.get("price_source_label", "Live model estimate"))
    total = price * nights if nights and nights > 0 else None
    metrics = prediction.get("holdout_metrics", {}) or {}

    total_text = f"Estimated stay total: {format_eur(total)}" if total is not None else "Select travel dates to estimate a stay total"
    subtitle = f"{escape(city)} using {escape(model_name)}. {escape(total_text)}."

    st.markdown(
        f"""
        <div class="prediction-result-hero">
            <div class="prediction-result-label">{escape(result_label)}</div>
            <div class="prediction-result-price">{escape(format_eur(price))}</div>
            <div class="prediction-result-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_price_source_cards(source_cards or [])

    rows: list[dict[str, str]] = [
        {"Category": "Prediction", "Item": "Nightly estimate", "Value": format_eur(price)},
        {"Category": "Prediction", "Item": "Requested nights", "Value": f"{nights:,}" if nights else "No stay length selected"},
        {"Category": "Prediction", "Item": "Estimated stay total", "Value": format_eur(total) if total is not None else "n/a"},
        {"Category": "Prediction", "Item": "Price source", "Value": result_label},
        {"Category": "Model", "Item": "Model used", "Value": model_name},
    ]

    if prediction.get("price_band"):
        rows.append({"Category": "Prediction", "Item": "Price band", "Value": str(prediction["price_band"])})
    if prediction.get("cache_month_window"):
        rows.append({"Category": "Prediction", "Item": "Prediction window", "Value": str(prediction["cache_month_window"])})
    if prediction.get("cache_listing_count"):
        rows.append({"Category": "Prediction", "Item": "Cached listings used", "Value": f"{int(prediction['cache_listing_count']):,}"})
    if prediction.get("snapshot_listing_count"):
        rows.append({"Category": "Prediction", "Item": "Snapshot listings used", "Value": f"{int(prediction['snapshot_listing_count']):,}"})
    if prediction.get("prediction_model_segment"):
        rows.append({"Category": "Model", "Item": "Model segment", "Value": str(prediction["prediction_model_segment"])})

    if prediction.get("training_rows"):
        rows.append({"Category": "Model", "Item": "Training rows", "Value": f"{int(prediction['training_rows']):,}"})

    if metrics:
        rows.extend([
            {"Category": "Model reference", "Item": "RMSE", "Value": _metric_text(metrics.get("rmse_eur"))},
            {"Category": "Model reference", "Item": "MAE", "Value": _metric_text(metrics.get("mae_eur"))},
            {"Category": "Model reference", "Item": "R2", "Value": _metric_text(metrics.get("r2"), decimals=4)},
        ])

    if availability:
        if "available_count" in availability:
            rows.extend([
                {"Category": "Calendar", "Item": "Any verified match", "Value": "Yes" if availability.get("is_available") else "No"},
                {"Category": "Calendar", "Item": "Listings checked", "Value": f"{availability.get('candidate_count', 0):,}"},
                {"Category": "Calendar", "Item": "Calendar-verified matches", "Value": f"{availability.get('available_count', 0):,}"},
                {"Category": "Calendar", "Item": "Status", "Value": str(availability.get("status", "n/a"))},
            ])
        else:
            rows.extend([
                {"Category": "Calendar", "Item": "Available", "Value": "Yes" if availability.get("is_available") else "No"},
                {"Category": "Calendar", "Item": "Available nights", "Value": f"{availability.get('available_nights', 0):,}"},
                {"Category": "Calendar", "Item": "Status", "Value": str(availability.get("status", "n/a"))},
            ])
        if availability.get("coverage_text"):
            rows.append({"Category": "Calendar", "Item": "Coverage", "Value": str(availability["coverage_text"])})

    rows.extend(scenario_rows)
    table_rows = "\n".join(
        (
            "<tr>"
            f"<td>{escape(str(row.get('Category', '')))}</td>"
            f"<td>{escape(str(row.get('Item', '')))}</td>"
            f"<td>{escape(str(row.get('Value', '')))}</td>"
            "</tr>"
        )
        for row in rows
    )
    st.markdown(
        f"""
        <div class="prediction-table-wrap">
            <table class="prediction-table">
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Detail</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Prices are snapshot prices, precomputed model estimates, or live model estimates depending on data availability. "
        "They are not live Airbnb quotes and do not include fees or taxes."
    )


def infer_city_from_prompt(prompt: str, master_data: dict[str, pd.DataFrame], fallback: str | None = None) -> str | None:
    lowered = prompt.lower()
    mentioned = [city for city in master_data if city.lower() in lowered]
    if len(mentioned) == 1:
        return mentioned[0]
    if len(mentioned) > 1:
        return None
    if fallback in master_data:
        return fallback
    return None


def chat_context_fallback_city(filters: dict, master_data: dict[str, pd.DataFrame]) -> str | None:
    context = st.session_state.get("chat_context", {})
    context_city = context.get("city")
    if context_city in master_data:
        return context_city

    history_city = infer_city_from_chat_history(master_data)
    if history_city in master_data:
        context["city"] = history_city
        return history_city

    filter_city = filters.get("city_choice")
    if filter_city in master_data:
        return filter_city
    return None


def infer_city_from_chat_history(master_data: dict[str, pd.DataFrame]) -> str | None:
    for message in reversed(st.session_state.get("messages", [])):
        if message.get("role") != "user":
            continue
        city = infer_city_from_prompt(str(message.get("content", "")), master_data)
        if city in master_data:
            return city
    return None


def backfill_chat_context_from_history(
    filters: dict,
    master_data: dict[str, pd.DataFrame],
) -> dict:
    context = st.session_state.setdefault("chat_context", {})

    for message in st.session_state.get("messages", []):
        if message.get("role") != "user":
            continue

        text = str(message.get("content", ""))
        city = infer_city_from_prompt(text, master_data)
        if city in master_data:
            context["city"] = city

        nights = _explicit_nights(text)
        if nights:
            context["nights"] = nights

        guests = _explicit_guests(text)
        if guests:
            context["guests"] = guests

        total_budget = _parse_money_amount(text)
        if total_budget:
            context["total_budget"] = total_budget

        room_type = _parse_room_type(text)
        if room_type:
            context["room_type"] = room_type

        amenities = _parse_amenities(text)
        if amenities:
            existing = context.get("amenities", [])
            context["amenities"] = sorted(set(existing) | set(amenities))

        months = _parse_months(text)
        if months:
            context["months"] = months

    if "nights" not in context:
        context["nights"] = max((filters["check_out"] - filters["check_in"]).days, 1)
    if "guests" not in context:
        context["guests"] = int(filters.get("guests", 2) or 2)

    return context


MONTH_NAMES = {
    "january": "January",
    "february": "February",
    "march": "March",
    "april": "April",
    "may": "May",
    "june": "June",
    "july": "July",
    "august": "August",
    "september": "September",
    "october": "October",
    "november": "November",
    "december": "December",
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


SEASON_TO_MONTHS = {
    "Winter": "December to February",
    "Spring": "March to May",
    "Summer": "June to August",
    "Autumn": "September to November",
}


def _parse_months(prompt: str) -> list[str]:
    lowered = prompt.lower()
    return [label for key, label in MONTH_NAMES.items() if re.search(rf"\b{key}\b", lowered)]


def _parse_explicit_seasons(prompt: str) -> list[str]:
    lowered = prompt.lower()
    season_terms = {
        "winter": "Winter",
        "spring": "Spring",
        "summer": "Summer",
        "autumn": "Autumn",
        "fall": "Autumn",
    }
    matches: list[tuple[int, str]] = []

    for term, label in season_terms.items():
        match = re.search(rf"\b{term}\b", lowered)
        if match:
            matches.append((match.start(), label))

    ordered = [label for _, label in sorted(matches, key=lambda item: item[0])]
    return list(dict.fromkeys(ordered))


def _parse_seasons(prompt: str) -> list[str]:
    lowered = prompt.lower()
    matches: list[tuple[int, str]] = []

    season_terms = {
        "winter": "Winter",
        "spring": "Spring",
        "summer": "Summer",
        "autumn": "Autumn",
        "fall": "Autumn",
    }
    for term, label in season_terms.items():
        match = re.search(rf"\b{term}\b", lowered)
        if match:
            matches.append((match.start(), label))

    for month_key, month_label in MONTH_NAMES.items():
        match = re.search(rf"\b{month_key}\b", lowered)
        if match and month_label in MONTH_TO_SEASON:
            matches.append((match.start(), MONTH_TO_SEASON[month_label]))

    ordered = [label for _, label in sorted(matches, key=lambda item: item[0])]
    return list(dict.fromkeys(ordered))


def _parse_money_amount(prompt: str) -> float | None:
    money_matches = re.findall(
        r"(?:eur|euro|euros|€)\s*([0-9][0-9,.]*)|([0-9][0-9,.]*)\s*(?:eur|euro|euros|€)",
        prompt,
        flags=re.IGNORECASE,
    )
    values = [left or right for left, right in money_matches]
    if not values:
        return None
    cleaned = values[0].replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _explicit_nights(prompt: str) -> int | None:
    match = re.search(r"\b(\d{1,2})\s*(?:night|nights)\b", prompt, flags=re.IGNORECASE)
    if match:
        return max(int(match.group(1)), 1)
    return None


def _parse_nights(prompt: str, filters: dict, context: dict | None = None) -> int:
    explicit = _explicit_nights(prompt)
    if explicit:
        return explicit
    if context and context.get("nights"):
        return max(int(context["nights"]), 1)
    return max((filters["check_out"] - filters["check_in"]).days, 1)


def _explicit_guests(prompt: str) -> int | None:
    lowered = prompt.lower()
    if any(term in lowered for term in ["girlfriend", "boyfriend", "partner", "couple", "two people", "two guests"]):
        return 2

    match = re.search(r"\b(\d{1,2})\s*(?:guest|guests|people|person|traveller|travellers|travelers)\b", lowered)
    if match:
        return max(int(match.group(1)), 1)
    return None


def _parse_guests(prompt: str, filters: dict, context: dict | None = None) -> int:
    explicit = _explicit_guests(prompt)
    if explicit:
        return explicit
    if context and context.get("guests"):
        return max(int(context["guests"]), 1)
    return int(filters.get("guests", 2) or 2)


def _parse_room_type(prompt: str) -> str | None:
    lowered = prompt.lower()
    if "private room" in lowered:
        return "Private room"
    if any(term in lowered for term in ["entire flat", "entire apartment", "entire apt", "entire home", "whole place"]):
        return "Entire home/apt"
    if "flat" in lowered or "apartment" in lowered:
        return "Entire home/apt"
    if "shared room" in lowered:
        return "Shared room"
    return None


def _parse_min_rating(prompt: str, fallback: float) -> float:
    lowered = prompt.lower()
    if any(term in lowered for term in ["excellent reviews", "best reviews", "great reviews", "high rating", "top rated"]):
        return max(fallback, 4.7)
    if any(term in lowered for term in ["good reviews", "good rating", "well reviewed", "well-reviewed"]):
        return max(fallback, 4.5)
    return fallback


def _parse_max_distance(prompt: str, fallback: float) -> float:
    lowered = prompt.lower()
    if any(term in lowered for term in ["as close to the center", "as close to the centre", "closest to the center", "closest to the centre"]):
        return min(fallback, 5)
    if any(term in lowered for term in ["city center", "city centre", "central", "near the center", "near the centre"]):
        return min(fallback, 7)
    return fallback


def _budget_personality_line(total_budget: float | None, per_night_budget: float | None, city: str | None = None) -> str:
    city_text = f" for {city}" if city else ""
    if per_night_budget is None:
        return f"I will keep the planning practical{city_text} and balance price, location, reviews, and availability."
    if per_night_budget >= 180:
        return f"That is a generous nightly budget{city_text}, so we can be a little picky and look for polished, well-located stays."
    if per_night_budget >= 120:
        return f"Nice, that gives us a comfortable search range{city_text}; I can prioritise quality without ignoring value."
    if per_night_budget >= 80:
        return f"Good, we have a realistic value-focused budget{city_text}, so I will look for the best balance rather than just the cheapest option."
    return f"This is more of a smart-value search{city_text}, so I will keep a close eye on price while protecting the basics like reviews and location."


def _explicitly_wants_cheapest(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(
        term in lowered
        for term in [
            "cheapest",
            "lowest price",
            "lowest cost",
            "as cheap as possible",
            "absolute cheapest",
        ]
    )


def _chat_price_sanity_floor(
    city: str,
    recommendation_filters: dict,
    prompt: str,
) -> tuple[float | None, int]:
    if _explicitly_wants_cheapest(prompt):
        return None, 0

    candidate_filters = {key: value for key, value in recommendation_filters.items() if key != "min_price_eur"}
    return cached_recommendation_price_floor(
        city,
        _recommendation_filter_key(candidate_filters),
        quantile=0.10,
        min_candidates=30,
    )


def _parse_amenities(prompt: str) -> list[str]:
    lowered = prompt.lower()
    amenities: list[str] = []
    checks = {
        "WiFi": ["wifi", "wi-fi", "wireless internet"],
        "Air conditioning": ["air conditioning", "aircon", "air con", " a/c", " ac "],
        "Kitchen": ["kitchen"],
        "Washer": ["washer", "washing machine"],
        "Parking": ["parking"],
        "Elevator": ["elevator", "lift"],
        "Dedicated workspace": ["workspace", "desk"],
        "Self check-in": ["self check", "self-check"],
    }
    padded = f" {lowered} "
    for amenity, terms in checks.items():
        if any(term in padded for term in terms):
            amenities.append(amenity)
    return amenities


def is_recommendation_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    triggers = [
        "recommend",
        "best",
        "holiday",
        "stay",
        "property",
        "listing",
        "option",
        "where should",
        "good value",
        "best value",
    ]
    return any(trigger in lowered for trigger in triggers)


def is_timing_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    timing_terms = [
        *MONTH_NAMES.keys(),
        "winter",
        "spring",
        "summer",
        "autumn",
        "fall",
        "june",
        "july",
        "august",
        "september",
        "month",
        "season",
        "time of year",
        "another time",
        "when should",
        "when would",
        "dates",
        "date",
    ]
    budget_terms = ["cheap", "cheaper", "budget", "price", "cost", "best"]
    projection_terms = [
        "projection",
        "projections",
        "forecast",
        "forecasting",
        "future",
        "availability",
        "calendar",
        "compare",
    ]
    return any(term in lowered for term in timing_terms) and any(
        term in lowered for term in [*budget_terms, *projection_terms]
    )


def is_seasonal_recommendation_prompt(prompt: str) -> bool:
    if not is_recommendation_prompt(prompt):
        return False

    explicit_seasons = _parse_explicit_seasons(prompt)
    if len(explicit_seasons) >= 1:
        return True

    lowered = prompt.lower()
    comparison_terms = ["compare", "compared", "versus", "vs", "between", "rather than"]
    return len(_parse_months(prompt)) >= 2 and any(term in lowered for term in comparison_terms)


def is_price_factor_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    price_terms = ["price", "prices", "cost", "nightly rate", "expensive", "cheap"]
    factor_terms = ["factor", "factors", "impact", "affect", "influence", "drive", "drives", "important", "importance"]
    return any(term in lowered for term in price_terms) and any(term in lowered for term in factor_terms)


def update_chat_context_from_prompt(
    prompt: str,
    city: str | None,
    filters: dict,
    master_data: dict[str, pd.DataFrame],
) -> None:
    context = st.session_state.setdefault("chat_context", {})

    if city in master_data:
        context["city"] = city

    nights = _explicit_nights(prompt)
    if nights:
        context["nights"] = nights

    guests = _explicit_guests(prompt)
    if guests:
        context["guests"] = guests

    total_budget = _parse_money_amount(prompt)
    if total_budget:
        context["total_budget"] = total_budget

    room_type = _parse_room_type(prompt)
    if room_type:
        context["room_type"] = room_type

    amenities = _parse_amenities(prompt)
    if amenities:
        existing = context.get("amenities", [])
        context["amenities"] = sorted(set(existing) | set(amenities))

    months = _parse_months(prompt)
    if months:
        context["months"] = months

    if "nights" not in context:
        context["nights"] = max((filters["check_out"] - filters["check_in"]).days, 1)
    if "guests" not in context:
        context["guests"] = int(filters.get("guests", 2) or 2)


def infer_chat_intent_from_prompt(prompt: str, context: dict | None = None) -> str:
    """Route the chat turn with lightweight parsing before the single LLM polish call."""
    lowered = prompt.lower()
    previous_intent = (context or {}).get("last_intent")

    if is_price_factor_prompt(prompt):
        return "price_factors"

    if any(term in lowered for term in ["model", "prediction", "xgboost", "lightgbm", "random forest", "rmse", "mae", "r2"]):
        return "model_info"

    if is_seasonal_recommendation_prompt(prompt):
        return "seasonal_recommendation"

    if is_timing_prompt(prompt):
        return "timing"

    if is_recommendation_prompt(prompt):
        return "recommendation"

    if previous_intent in {"recommendation", "seasonal_recommendation"} and any(
        term in lowered
        for term in [
            "show me",
            "give me",
            "what about",
            "another",
            "more options",
            "recommendations",
            "properties",
            "listings",
        ]
    ):
        return str(previous_intent)

    return "general_question"


def build_timing_response(
    prompt: str,
    master_data: dict[str, pd.DataFrame],
    filters: dict,
    context: dict,
) -> tuple[str, str | None]:
    selected_city = infer_city_from_prompt(
        prompt,
        master_data,
        fallback=context.get("city") or (filters.get("city_choice") if filters.get("city_choice") != "Compare both" else None),
    )

    if selected_city is None:
        return (
            "I can answer that, but please specify either Madrid or Tokyo so I can use the right calendar and price context.",
            None,
        )

    nights = _parse_nights(prompt, filters, context)
    guests = _parse_guests(prompt, filters, context)
    total_budget = _parse_money_amount(prompt) or context.get("total_budget")
    per_night_budget = (
        total_budget / nights
        if total_budget and nights
        else context.get("per_night_budget_eur") or filters.get("budget_max")
    )
    room_type = _parse_room_type(prompt) or context.get("room_type")
    amenities = _parse_amenities(prompt) or context.get("amenities", [])
    min_rating = max(
        _parse_min_rating(prompt, float(filters.get("min_rating", 4.0))),
        float(context.get("min_rating", filters.get("min_rating", 4.0)) or 4.0),
    )
    max_distance_km = min(
        _parse_max_distance(prompt, float(filters.get("max_distance_km", 15))),
        float(context.get("max_distance_km", filters.get("max_distance_km", 15)) or 15),
    )

    candidate_filters = {
        "budget_max": per_night_budget,
        "guests": guests,
        "room_type": room_type,
        "amenities": amenities,
        "min_rating": min_rating,
        "max_distance_km": max_distance_km,
        "min_available_days_30": int(filters.get("min_available_days_30", 0) or 0),
    }
    candidates = cached_filtered_listings(
        selected_city,
        _recommendation_filter_key(candidate_filters),
    )

    season_cols = {
        "Winter": "calendar_unavailable_winter",
        "Spring": "calendar_unavailable_spring",
        "Summer": "calendar_unavailable_summer",
        "Autumn": "calendar_unavailable_autumn",
    }

    budget_text = format_eur(total_budget) if total_budget else "not specified"
    nightly_text = format_eur(per_night_budget) if per_night_budget else "no fixed nightly cap"
    room_text = room_type or "any room type"
    amenities_text = ", ".join(amenities) if amenities else "no specific amenities carried over"
    months = list(dict.fromkeys([*_parse_months(prompt), *context.get("months", [])]))
    month_text = ", ".join(months) if months else "no specific month"
    budget_mood = _budget_personality_line(total_budget, per_night_budget, selected_city)

    if candidates.empty:
        return (
            f"I kept the trip context for **{selected_city}**, but the current filters leave no matching listings. "
            "I would loosen the budget, room type, amenities, or rating first, then rerun the month comparison.",
            selected_city,
        )

    requested_seasons = list(
        dict.fromkeys(
            [
                *_parse_seasons(prompt),
                *context.get("seasons", []),
                *(MONTH_TO_SEASON[m] for m in months if m in MONTH_TO_SEASON),
            ]
        )
    )
    lowered = prompt.lower()
    compare_all = (
        not requested_seasons
        or len(requested_seasons) == 1
        or any(term in lowered for term in ["another time", "best time", "time of year", "when should", "when would"])
    )
    comparison_seasons = list(season_cols) if compare_all else requested_seasons

    cache_summary = prediction_window_summary(
        selected_city,
        candidates["listing_id"],
        months=months,
        seasons=comparison_seasons,
    )

    season_rows = []
    if not cache_summary.empty:
        for row in cache_summary.itertuples(index=False):
            season_rows.append(
                {
                    "window": row.window,
                    "months": row.months,
                    "count": int(row.count),
                    "mean_price": float(row.mean_price),
                    "median_price": float(row.median_price),
                    "mean_unavailable": float(row.mean_unavailable),
                    "price_source": row.price_source,
                }
            )
    else:
        for season in comparison_seasons:
            column = season_cols.get(season)
            if not column or column not in candidates.columns:
                continue

            season_pool = candidates.copy()
            season_pool["_season_unavailable"] = pd.to_numeric(season_pool[column], errors="coerce")
            season_pool = season_pool[season_pool["_season_unavailable"].notna()].copy()
            if season_pool.empty:
                continue

            likely_available = season_pool[season_pool["_season_unavailable"] < 1.0].copy()
            price_pool = likely_available if len(likely_available) >= 10 else season_pool
            prices = pd.to_numeric(price_pool["price_eur"], errors="coerce").dropna()
            if prices.empty:
                continue

            season_rows.append(
                {
                    "window": season,
                    "months": SEASON_TO_MONTHS.get(season, season),
                    "count": int(len(price_pool)),
                    "mean_price": float(prices.mean()),
                    "median_price": float(prices.median()),
                    "mean_unavailable": float(price_pool["_season_unavailable"].mean()),
                    "price_source": "cleaned listing snapshot price",
                }
            )

    if not season_rows:
        return (
            f"I can keep planning around **{selected_city}**, but I cannot calculate seasonal price estimates because the seasonal calendar fields are missing for the matching listings.",
            selected_city,
        )

    season_table = pd.DataFrame(season_rows).sort_values("mean_price")
    best_price_row = season_table.iloc[0]
    target_window = months[0] if months else (requested_seasons[0] if requested_seasons else None)
    target_row = (
        season_table[season_table["window"] == target_window].head(1)
        if target_window
        else pd.DataFrame()
    )
    price_source = str(season_table["price_source"].iloc[0])

    lines = [
        f"Good question. Let's make this practical for **{selected_city}** rather than just talking about availability.",
        budget_mood,
        "",
        "**Trip context I am using**",
        f"- Guests: {guests}",
        f"- Stay length: {nights} nights",
        f"- Total budget: {budget_text}",
        f"- Nightly cap implied: {nightly_text}",
        f"- Room type: {room_text}",
        f"- Amenities: {amenities_text}",
        f"- Month mentioned: {month_text}",
        "",
        "**Estimated nightly price by travel window**",
    ]

    for row in season_table.sort_values("window").itertuples(index=False):
        total_at_mean = row.mean_price * nights
        window_text = row.window if str(row.months) == str(row.window) else f"{row.window} ({row.months})"
        lines.append(
            f"- **{window_text}**: average {format_eur(row.mean_price)} per night "
            f"({format_eur(total_at_mean)} for {nights} nights), median {format_eur(row.median_price)}, "
            f"based on {row.count:,} matching listings."
        )

    lines.append("")
    price_spread = float(season_table["mean_price"].max() - season_table["mean_price"].min())
    availability_rank = season_table.dropna(subset=["mean_unavailable"]).sort_values("mean_unavailable")
    if len(season_table) > 1 and price_spread <= 2 and not availability_rank.empty:
        best_availability_row = availability_rank.iloc[0]
        lines.append(
            f"**My take:** the predicted prices are effectively tied across these windows "
            "(within less than EUR 2 per night). I would use availability as the tie-breaker: "
            f"**{best_availability_row['window']}** has the stronger availability signal, with an average unavailable rate "
            f"of {best_availability_row['mean_unavailable'] * 100:.1f}%."
        )
    elif not target_row.empty and best_price_row["window"] != target_row.iloc[0]["window"]:
        target = target_row.iloc[0]
        nightly_saving = float(target["mean_price"] - best_price_row["mean_price"])
        lines.append(
            f"**My take:** {target['window']} looks more expensive for this search than **{best_price_row['window']}**. "
            f"The average difference is about {format_eur(nightly_saving)} per night, or roughly "
            f"{format_eur(nightly_saving * nights)} over {nights} nights."
        )
    else:
        lines.append(
            f"**My take:** based on this snapshot, **{best_price_row['window']}** is the cheapest window among the periods compared, "
            f"at about {format_eur(best_price_row['mean_price'])} per night on average."
        )

    lines.extend(
        [
            "",
            "Availability is still useful as a second check, but I would not lead with it. "
            "For a traveller, the useful decision is price first, then whether exact dates are available.",
            "",
            f"Small caveat: these are not live Airbnb future rates. The price source is **{price_source}**, combined with the seasonal calendar signal to estimate which matching listings are plausible for each travel window.",
        ]
    )
    return "\n".join(lines), selected_city


def _feature_takeaway(feature: str) -> str:
    clean = feature.replace("_", " ").strip()
    lowered = clean.lower()
    if "host" in lowered or "calculated host listings count" in lowered:
        return "host/listing scale"
    if "bathroom type" in lowered:
        return "bathroom privacy"
    if any(term in lowered for term in ["accommodates", "bedrooms", "beds", "bathrooms", "capacity"]):
        return "size and guest capacity"
    if any(term in lowered for term in ["room type", "entire home", "private room", "shared room"]):
        return "privacy / room type"
    if "property group" in lowered or "hotel" in lowered:
        return "property type"
    if any(term in lowered for term in ["distance", "central", "latitude", "longitude", "neighbourhood"]):
        return "location"
    if lowered.startswith("has ") or "amenities" in lowered:
        return "amenities"
    if any(term in lowered for term in ["review", "rating"]):
        return "review activity / quality"
    return "other listing attributes"


def build_price_factor_response(
    prompt: str,
    master_data: dict[str, pd.DataFrame],
    ml_outputs: dict[str, pd.DataFrame],
    filters: dict,
    context: dict | None = None,
) -> tuple[str, str | None]:
    selected_city = infer_city_from_prompt(
        prompt,
        master_data,
        fallback=(context or {}).get("city") or (filters.get("city_choice") if filters.get("city_choice") != "Compare both" else None),
    )
    selected_cities = [selected_city] if selected_city in master_data else available_cities(master_data)

    best = best_models(ml_outputs)
    lines = [
        "Good question. The useful answer is not just the city median price; it is what the model learned actually moves the nightly price.",
        "",
    ]

    for city in selected_cities:
        city_best = best[best["city"].astype(str) == city] if not best.empty and "city" in best.columns else pd.DataFrame()
        model_name = city_best["model"].iloc[0] if not city_best.empty else None
        features = top_features(ml_outputs, city=city, model=model_name, limit=10)

        if features.empty:
            lines.append(f"**{city}:** feature importance is not available yet.")
            continue

        grouped = {}
        for _, row in features.iterrows():
            feature = str(row.get("feature", "unknown feature"))
            importance = pd.to_numeric(row.get("importance"), errors="coerce")
            grouped.setdefault(_feature_takeaway(feature), 0.0)
            grouped[_feature_takeaway(feature)] += float(importance) if pd.notna(importance) else 0.0

        grouped_text = ", ".join(
            f"{name} ({score:.2f})"
            for name, score in sorted(grouped.items(), key=lambda item: item[1], reverse=True)[:5]
        )

        lines.append(f"**{city}**")
        if model_name:
            lines.append(f"- Current best model used for this read: **{model_name}**.")
        lines.append(f"- Main driver groups: {grouped_text}.")
        lines.append("- Top individual features:")

        for _, row in features.head(6).iterrows():
            importance = pd.to_numeric(row.get("importance"), errors="coerce")
            importance_text = f"{float(importance):.3f}" if pd.notna(importance) else "n/a"
            feature = str(row.get("feature", "unknown feature")).replace("_", " ")
            lines.append(f"  - {feature}: importance {importance_text}")

        lines.append("")

    lines.extend(
        [
            "**Plain-English takeaway:** price is mostly shaped by how much space/privacy the guest gets, how many people the listing can host, bathroom setup, property type, and then location/amenities/review activity. "
            "Amenities matter, but usually less than room type and capacity.",
            "",
            "Small modelling caveat: feature importance explains what the trained model used to predict price; it is not strict proof of causation.",
        ]
    )
    return "\n".join(lines), selected_city


def _format_listing_id(value) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.0f}" if value.is_integer() else str(value)
    return str(value)


def build_chat_recommendation_cards(
    recs: pd.DataFrame,
    nights: int,
    fallback_city: str,
) -> list[dict]:
    cards: list[dict] = []
    for rank, (_, row) in enumerate(recs.iterrows(), start=1):
        price = row.get("price_eur")
        total = price * nights if pd.notna(price) else None
        rating = row.get("review_scores_rating")
        value_score = row.get("value_score")
        distance = row.get("distance_to_center_km")
        availability = row.get("availability_30")

        cards.append(
            {
                "rank": rank,
                "neighbourhood": _safe_value(row, "neighbourhood_cleansed", "Unknown neighbourhood"),
                "city": _safe_value(row, "city", fallback_city) or fallback_city,
                "room_type": _safe_value(row, "room_type", "n/a"),
                "property_group": _safe_value(row, "property_group", "Property"),
                "listing_id": _format_listing_id(row.get("listing_id", "n/a")),
                "price_text": format_eur(price),
                "total_text": format_eur(total),
                "nights": nights,
                "rating_text": f"{float(rating):.2f}" if pd.notna(rating) else "n/a",
                "value_text": f"{float(value_score):.1f}" if pd.notna(value_score) else "n/a",
                "distance_text": f"{float(distance):.1f} km from centre" if pd.notna(distance) else "Distance n/a",
                "availability_text": f"{int(availability)} days next 30" if pd.notna(availability) else "Availability n/a",
                "season_label": _safe_value(row, "season_label", ""),
                "season_fit_text": _safe_value(row, "season_fit_text", ""),
                "season_signal_text": _safe_value(row, "season_signal_text", ""),
                "reason": _safe_value(
                    row,
                    "recommendation_reason",
                    "Strong balance of price, rating, location, amenities, and availability.",
                ),
            }
        )
    return cards


def build_seasonal_recommendation_response(
    prompt: str,
    master_data: dict[str, pd.DataFrame],
    filters: dict,
    context: dict | None = None,
    limit_per_season: int = 3,
) -> tuple[str, str | None, list[dict], str | None]:
    selected_city = infer_city_from_prompt(
        prompt,
        master_data,
        fallback=(context or {}).get("city") or (filters["city_choice"] if filters.get("city_choice") != "Compare both" else None),
    )

    if selected_city is None:
        return (
            "I can compare seasonal recommendations, but tell me whether this trip is for Madrid or Tokyo first.",
            None,
            [],
            None,
        )

    requested_seasons = list(dict.fromkeys([*_parse_seasons(prompt), *(context or {}).get("seasons", [])]))
    if not requested_seasons:
        requested_seasons = ["Summer", "Winter"]

    season_cols = {
        "Winter": "calendar_unavailable_winter",
        "Spring": "calendar_unavailable_spring",
        "Summer": "calendar_unavailable_summer",
        "Autumn": "calendar_unavailable_autumn",
    }

    nights = _parse_nights(prompt, filters, context)
    guests = _parse_guests(prompt, filters, context)
    total_budget = _parse_money_amount(prompt) or (context or {}).get("total_budget")
    per_night_budget = (
        total_budget / nights
        if total_budget and nights
        else (context or {}).get("per_night_budget_eur") or filters.get("budget_max")
    )
    room_type = _parse_room_type(prompt) or (context or {}).get("room_type") or (filters.get("room_type") if filters.get("room_type") != "Any" else None)
    amenities = _parse_amenities(prompt) or (context or {}).get("amenities", []) or filters.get("amenities", [])
    min_rating = max(
        _parse_min_rating(prompt, float(filters.get("min_rating", 4.0))),
        float((context or {}).get("min_rating", filters.get("min_rating", 4.0)) or 4.0),
    )
    max_distance_km = min(
        _parse_max_distance(prompt, float(filters.get("max_distance_km", 15))),
        float((context or {}).get("max_distance_km", filters.get("max_distance_km", 15)) or 15),
    )

    recommendation_filters = {
        "budget_max": per_night_budget,
        "guests": guests,
        "room_type": room_type,
        "amenities": amenities,
        "min_rating": min_rating,
        "max_distance_km": max_distance_km,
        "min_available_days_30": max(int(filters.get("min_available_days_30", 0) or 0), min(nights, 30)),
    }

    price_floor, candidate_count = _chat_price_sanity_floor(
        selected_city,
        recommendation_filters,
        prompt,
    )
    if price_floor:
        recommendation_filters["min_price_eur"] = price_floor

    base_candidates = cached_filtered_listings(
        selected_city,
        _recommendation_filter_key(recommendation_filters),
    )
    relaxed_note = ""
    if base_candidates.empty and amenities:
        relaxed_filters = {**recommendation_filters, "amenities": []}
        base_candidates = cached_filtered_listings(
            selected_city,
            _recommendation_filter_key(relaxed_filters),
        )
        recommendation_filters = relaxed_filters
        relaxed_note = " I relaxed the amenity filter because no seasonal matches had every requested amenity."
    if base_candidates.empty and price_floor:
        relaxed_filters = {key: value for key, value in recommendation_filters.items() if key != "min_price_eur"}
        base_candidates = cached_filtered_listings(
            selected_city,
            _recommendation_filter_key(relaxed_filters),
        )
        recommendation_filters = relaxed_filters
        price_floor = None
        relaxed_note = " I relaxed the low-price sanity floor because it removed every seasonal match."

    if base_candidates.empty:
        return (
            f"I tried to compare seasonal recommendations for **{selected_city}**, but the current filters leave no matching listings. "
            "I would loosen the budget, amenities, or room type first, then rerun the seasonal comparison.",
            selected_city,
            [],
            None,
        )

    cards: list[dict] = []
    season_summaries: list[str] = []

    for season in requested_seasons:
        season_column = season_cols.get(season)
        if not season_column or season_column not in base_candidates.columns:
            season_summaries.append(f"- {season}: seasonal calendar field is not available.")
            continue

        scored = add_value_score(base_candidates)
        season_unavailable = pd.to_numeric(scored[season_column], errors="coerce")
        if season_unavailable.notna().sum() == 0:
            season_summaries.append(f"- {season}: no seasonal availability signal available for the matching listings.")
            continue

        filled_unavailable = season_unavailable.fillna(season_unavailable.median())
        season_fit = 1 - filled_unavailable.rank(pct=True, method="average")
        scored["season_fit_score"] = (season_fit * 100).round(1)
        scored["season_adjusted_score"] = (
            0.70 * pd.to_numeric(scored["value_score"], errors="coerce").fillna(50)
            + 0.30 * scored["season_fit_score"]
        ).round(1)
        scored["season_unavailable_rate"] = filled_unavailable
        scored["season_label"] = season
        scored["season_fit_text"] = scored["season_fit_score"].map(lambda value: f"{float(value):.1f}")
        scored["season_signal_text"] = scored["season_unavailable_rate"].map(
            lambda value: f"{float(value) * 100:.1f}% unavailable"
        )
        scored["recommendation_reason"] = scored.apply(build_recommendation_reason, axis=1)
        scored["recommendation_reason"] = scored.apply(
            lambda row: (
                f"{row['recommendation_reason']}; {season.lower()} signal: "
                f"{row['season_signal_text']}"
            ),
            axis=1,
        )

        recs = scored.sort_values(
            ["season_adjusted_score", "value_score", "review_scores_rating", "price_eur"],
            ascending=[False, False, False, True],
        ).head(limit_per_season)

        median_unavailable = float(season_unavailable.median())
        season_summaries.append(
            f"- {season}: median unavailable signal is {median_unavailable * 100:.1f}% across the matching pool."
        )
        cards.append(
            {
                "type": "section",
                "title": f"{season} shortlist",
                "note": f"Ranked by value score plus the {season.lower()} availability signal.",
            }
        )
        cards.extend(build_chat_recommendation_cards(recs, nights=nights, fallback_city=selected_city))

    if not any(card.get("type") != "section" for card in cards):
        return (
            f"I found the seasonal request for **{selected_city}**, but the needed seasonal availability fields were not available for the matching listings.",
            selected_city,
            [],
            None,
        )

    season_text = " vs ".join(requested_seasons)
    budget_text = format_eur(per_night_budget) + " per night" if per_night_budget else "No budget cap"
    total_text = format_eur(total_budget) if total_budget else "Not specified"
    selected_room_text = room_type or "Any room type"
    amenities_text = ", ".join(amenities) if amenities else "No specific amenities requested"

    intro = "\n".join(
        [
            f"Great idea, let's make the timing decision easier: I will compare **{season_text}** stays for **{selected_city}** using the trip details you have already given me.",
            "I have split the shortlist by season, keeping strong value, good reviews, location, and your must-have amenities in the mix.",
            relaxed_note,
        ]
    ).strip()

    footer_lines = [
        "**How I compared the seasons**",
        *season_summaries,
        "",
        "**Trip filters carried forward**",
        f"- Stay length: {nights} nights",
        f"- Guests: {guests}",
        f"- Total budget: {total_text}",
        f"- Nightly cap used: {budget_text}",
        f"- Room type: {selected_room_text}",
        f"- Amenities: {amenities_text}",
        f"- Minimum rating: {recommendation_filters['min_rating']:.1f}",
        f"- Max distance to centre: {recommendation_filters['max_distance_km']:.0f} km",
        (
            f"- Price sanity check: ignored unusually low matching rows below {format_eur(price_floor)} "
            f"from a pool of {candidate_count:,} listings"
            if price_floor
            else "- Price sanity check: no low-price outlier floor applied"
        ),
        "- Source: cleaned local Airbnb snapshot data. The seasonal signal is from local calendar-derived availability, not a live Airbnb future quote.",
    ]
    return intro, selected_city, cards, "\n".join(footer_lines)


def build_chat_recommendation_response(
    prompt: str,
    master_data: dict[str, pd.DataFrame],
    filters: dict,
    context: dict | None = None,
    limit: int = 5,
) -> tuple[str, str | None, list[dict], str | None]:
    selected_city = infer_city_from_prompt(
        prompt,
        master_data,
        fallback=(context or {}).get("city") or (filters["city_choice"] if filters.get("city_choice") != "Compare both" else None),
    )

    if selected_city is None:
        return (
            "Absolutely, I can help plan that. Which city are we dreaming about for this trip, Madrid or Tokyo?",
            None,
            [],
            None,
        )

    nights = _parse_nights(prompt, filters, context)
    guests = _parse_guests(prompt, filters, context)
    total_budget = _parse_money_amount(prompt) or (context or {}).get("total_budget")
    per_night_budget = (
        total_budget / nights
        if total_budget and nights
        else (context or {}).get("per_night_budget_eur") or filters.get("budget_max")
    )
    room_type = _parse_room_type(prompt) or (context or {}).get("room_type") or (filters.get("room_type") if filters.get("room_type") != "Any" else None)
    amenities = _parse_amenities(prompt) or (context or {}).get("amenities", []) or filters.get("amenities", [])
    min_rating = max(
        _parse_min_rating(prompt, float(filters.get("min_rating", 4.0))),
        float((context or {}).get("min_rating", filters.get("min_rating", 4.0)) or 4.0),
    )
    max_distance_km = min(
        _parse_max_distance(prompt, float(filters.get("max_distance_km", 15))),
        float((context or {}).get("max_distance_km", filters.get("max_distance_km", 15)) or 15),
    )

    recommendation_filters = {
        "budget_max": per_night_budget,
        "guests": guests,
        "room_type": room_type,
        "amenities": amenities,
        "min_rating": min_rating,
        "max_distance_km": max_distance_km,
        "min_available_days_30": max(int(filters.get("min_available_days_30", 0) or 0), min(nights, 30)),
    }

    price_floor, candidate_count = _chat_price_sanity_floor(
        selected_city,
        recommendation_filters,
        prompt,
    )
    if price_floor:
        recommendation_filters["min_price_eur"] = price_floor

    recs = cached_recommendations(
        selected_city,
        _recommendation_filter_key(recommendation_filters),
        limit=limit,
    )

    if recs.empty and amenities:
        relaxed_filters = {**recommendation_filters, "amenities": []}
        recs = cached_recommendations(
            selected_city,
            _recommendation_filter_key(relaxed_filters),
            limit=limit,
        )
        relaxed_note = " I could not find enough listings with every requested amenity, so the shortlist below relaxes the amenity filter."
    elif recs.empty and price_floor:
        relaxed_filters = {key: value for key, value in recommendation_filters.items() if key != "min_price_eur"}
        recs = cached_recommendations(
            selected_city,
            _recommendation_filter_key(relaxed_filters),
            limit=limit,
        )
        relaxed_note = " I relaxed the price sanity check because it removed every matching option."
        price_floor = None
    else:
        relaxed_note = ""

    budget_text = format_eur(per_night_budget) + " per night" if per_night_budget else "No budget cap"
    total_text = format_eur(total_budget) if total_budget else "Not specified"
    selected_room_text = room_type or "Any room type"
    amenities_text = ", ".join(amenities) if amenities else "No specific amenities requested"
    budget_mood = _budget_personality_line(total_budget, per_night_budget, selected_city)

    if recs.empty:
        return (
            f"I checked **{selected_city}** using these filters: {guests} guests, {nights} nights, "
            f"{budget_text}, {selected_room_text}, amenities: {amenities_text}. "
            "I could not find a strong match yet, so I would relax either budget, room type, or amenities first. "
            "We can still make the trip work; the search just needs a little more breathing room.",
            selected_city,
            [],
            None,
        )

    cards = build_chat_recommendation_cards(recs, nights=nights, fallback_city=selected_city)
    lines = [
        f"Lovely, **{selected_city}** it is. {budget_mood}",
        f"Here is a shortlist I would start with.{relaxed_note}",
    ]

    footer_lines = [
        "**My pick:** I would start with option 1. It has the strongest balance of price, review quality, location, and requested amenities.",
        "",
        "**Filters I used**",
        f"- Stay length: {nights} nights",
        f"- Guests: {guests}",
        f"- Total budget: {total_text}",
        f"- Nightly cap used: {budget_text}",
        f"- Room type: {selected_room_text}",
        f"- Amenities: {amenities_text}",
        f"- Minimum rating: {recommendation_filters['min_rating']:.1f}",
        f"- Max distance to centre: {recommendation_filters['max_distance_km']:.0f} km",
        f"- Availability: at least {recommendation_filters['min_available_days_30']} available days in the next 30",
        (
            f"- Price sanity check: ignored the lowest 10% of {candidate_count:,} matching rows "
            f"below {format_eur(price_floor)} because they can be data outliers or unusually discounted listings"
            if price_floor
            else "- Price sanity check: no low-price outlier floor applied"
        ),
        "- Source: cleaned local Airbnb snapshot data, not a live Airbnb quote.",
    ]
    return "\n".join(lines), selected_city, cards, "\n".join(footer_lines)


def price_prediction_view(master_data: dict[str, pd.DataFrame], filters: dict) -> None:
    st.subheader("Price Check")

    model_cities = available_price_model_cities()
    if not model_cities:
        st.warning("No saved price models found. Run `python train_price_models.py` from the chatbot folder.")
        return

    default_city = filters["city_choice"] if filters["city_choice"] in model_cities else model_cities[0]
    min_prediction_date, max_prediction_date = _price_prediction_date_bounds()
    header_city_col, check_in_col, check_out_col, header_mode_col = st.columns([1.1, 0.85, 0.85, 1.9])
    with header_city_col:
        city = st.selectbox("Prediction city", model_cities, index=model_cities.index(default_city))
    with check_in_col:
        check_in = st.date_input(
            "Check-in",
            value=filters["check_in"],
            min_value=min_prediction_date,
            max_value=max_prediction_date,
            key="prediction_check_in",
        )
    with check_out_col:
        check_out = st.date_input(
            "Check-out",
            value=filters["check_out"],
            min_value=min_prediction_date,
            max_value=max_prediction_date,
            key="prediction_check_out",
        )
    with header_mode_col:
        prediction_mode = st.radio(
            "Prediction mode",
            ["Custom stay", "Existing listing ID"],
            horizontal=True,
        )

    options = model_feature_options(city)
    defaults = model_feature_defaults(city)
    city_df = master_data.get(city, pd.DataFrame())

    dates_valid = check_out > check_in
    nights = max((check_out - check_in).days, 0)
    if not dates_valid:
        st.warning("Choose a check-out date after the check-in date before running a prediction.")

    if prediction_mode == "Existing listing ID":
        default_listing_id = int(city_df["listing_id"].iloc[0]) if not city_df.empty and "listing_id" in city_df.columns else 0
        with st.form("listing_price_prediction_form"):
            listing_id = st.number_input("Listing ID", min_value=0, value=default_listing_id, step=1, key="prediction_listing_id")
            run_listing_prediction = st.form_submit_button("Run listing prediction", type="primary", use_container_width=True)

        if not run_listing_prediction:
            st.info("Enter a listing ID, then run the prediction.")
            return

        if not dates_valid:
            return

        listing = city_df[city_df["listing_id"] == int(listing_id)].head(1) if not city_df.empty else pd.DataFrame()

        if listing.empty:
            st.warning("Listing ID not found in the master dataset for this city.")
            return

        snapshot_price = pd.to_numeric(listing["price_eur"], errors="coerce").iloc[0] if "price_eur" in listing.columns else None
        availability = _calendar_result_for_listing(city, int(listing_id), check_in, check_out)
        snapshot_prediction = (
            _snapshot_price_estimate(
                city,
                snapshot_price,
                label="Snapshot-backed listing price",
                model_name="Cleaned listing snapshot",
                listing_count=1,
            )
            if availability.get("calendar_checked") and availability.get("is_available")
            else None
        )
        precomputed_prediction = None
        if snapshot_prediction:
            prediction = snapshot_prediction
        else:
            precomputed_prediction = _precomputed_prediction_estimate(city, [int(listing_id)], check_in, check_out)
            if precomputed_prediction:
                prediction = precomputed_prediction
            else:
                prediction = predict_listing_price(city, listing.iloc[0])
                prediction["price_source"] = "live_model_estimate"
                prediction["price_source_label"] = "Live model estimate"

        if snapshot_prediction:
            price_source_detail = "Using the listing's cleaned snapshot price because the local calendar confirms this stay is available."
            price_source_status = "good"
        elif precomputed_prediction:
            price_source_detail = f"Cached monthly prediction for {prediction.get('cache_month_window')}."
            price_source_status = "good"
        else:
            price_source_detail = "No matching snapshot-backed or cached future price was available, so the model ran on demand."
            price_source_status = "warning"

        availability_status = "good" if availability.get("is_available") else "warning"
        if availability.get("calendar_checked") is False:
            availability_value = "Not in calendar window"
            availability_detail = availability.get("coverage_text", "The requested dates are outside local calendar coverage.")
        elif availability.get("is_available"):
            availability_value = "Verified availability"
            availability_detail = f"{availability.get('available_nights', 0)} of {availability.get('nights', 0)} requested nights available."
        else:
            availability_value = "Availability rule failed"
            availability_detail = str(availability.get("status", "Requested stay is not available."))

        price_source_cards = [
            {
                "title": "Availability",
                "value": availability_value,
                "detail": availability_detail,
                "status": availability_status,
            },
            {
                "title": "Snapshot price",
                "value": format_eur(snapshot_price),
                "detail": "Actual listed nightly price from the cleaned Airbnb dataset.",
                "status": "neutral",
            },
            {
                "title": "Displayed price source",
                "value": str(prediction.get("price_source_label", "Live model estimate")),
                "detail": price_source_detail,
                "status": price_source_status,
            },
        ]

        scenario_rows = [
            {"Category": "Listing", "Item": "Listing ID", "Value": f"{int(listing_id):,}"},
            {"Category": "Listing", "Item": "Feature source", "Value": prediction.get("feature_source", "listing attributes")},
        ]
        if "neighbourhood_cleansed" in listing.columns:
            scenario_rows.append({
                "Category": "Listing",
                "Item": "Neighbourhood",
                "Value": str(listing["neighbourhood_cleansed"].iloc[0]),
            })
        if "room_type" in listing.columns:
            scenario_rows.append({"Category": "Listing", "Item": "Room type", "Value": str(listing["room_type"].iloc[0])})

        render_price_prediction_result(
            prediction,
            nights=availability.get("nights", 0),
            scenario_rows=scenario_rows,
            availability=availability,
            source_cards=price_source_cards,
        )

        with st.expander("Listing attributes used", expanded=False):
            st.dataframe(listing, use_container_width=True, hide_index=True)
        return

    room_type_values = options.get("room_type", ["Entire home/apt", "Private room"])
    property_values = options.get("property_group", ["Apartment/condo"])
    neighbourhood_values = options.get("neighbourhood_cleansed", [])

    amenity_labels = {
        "has_wifi": "WiFi",
        "has_kitchen": "Kitchen",
        "has_air_conditioning": "Air conditioning",
        "has_washer": "Washer",
        "has_elevator": "Elevator",
        "has_parking": "Parking",
        "has_dedicated_workspace": "Dedicated workspace",
        "has_self_checkin": "Self check-in",
    }

    with st.form("custom_price_prediction_form"):
        stay_left, stay_middle, stay_right = st.columns(3)
        with stay_left:
            accommodates = st.number_input("Guests", min_value=1, max_value=16, value=int(filters["guests"]), step=1)
        with stay_middle:
            bedrooms = st.number_input(
                "Bedrooms",
                min_value=0,
                max_value=10,
                value=_safe_int(defaults.get("bedrooms"), fallback=1, minimum=0, maximum=10),
                step=1,
            )
        with stay_right:
            bathrooms = st.number_input(
                "Bathrooms",
                min_value=0,
                max_value=10,
                value=_safe_int(defaults.get("bathrooms"), fallback=1, minimum=0, maximum=10),
                step=1,
            )

        property_left, property_middle, property_right = st.columns(3)
        with property_left:
            room_type = st.selectbox(
                "Room type",
                room_type_values,
                index=_option_index(room_type_values, defaults.get("room_type")),
            )
        with property_middle:
            property_group = st.selectbox(
                "Property group",
                property_values,
                index=_option_index(property_values, defaults.get("property_group")),
            )
        with property_right:
            neighbourhood = (
                st.selectbox(
                    "Neighbourhood",
                    neighbourhood_values,
                    index=_option_index(neighbourhood_values, defaults.get("neighbourhood_cleansed")),
                )
                if neighbourhood_values
                else None
            )

        default_distance = max(0.0, min(30.0, _safe_float(defaults.get("distance_to_center_km"), 5.0)))
        distance_to_center_km, neighbourhood_distance_count = neighbourhood_average_distance(
            city_df,
            neighbourhood,
            default_distance,
        )

        slider_left, slider_middle, slider_right = st.columns(3)
        with slider_left:
            render_neighbourhood_distance_card(
                distance_to_center_km,
                neighbourhood_distance_count,
                neighbourhood,
            )
        with slider_middle:
            min_rating = st.slider(
                "Minimum rating",
                0.0,
                5.0,
                float(filters.get("min_rating", 4.0)),
                0.1,
            )
        with slider_right:
            min_available_days_30 = st.slider(
                "Minimum available days next 30",
                0,
                30,
                int(filters.get("min_available_days_30", 0)),
                1,
            )

        st.markdown("Key amenities")
        amenity_columns = st.columns(4)
        amenity_cols = {}
        for index, (column, label) in enumerate(amenity_labels.items()):
            with amenity_columns[index % 4]:
                amenity_cols[column] = st.checkbox(
                    label,
                    value=_default_bool(defaults, column, fallback=column in {"has_wifi", "has_kitchen"}),
                )

        run_prediction = st.form_submit_button("Run prediction", type="primary", use_container_width=True)

    if not run_prediction:
        st.info("Adjust the filters, then run the prediction to calculate the estimate.")
        return

    if not dates_valid:
        return

    estimated_beds = max(int(round(accommodates / 2)), 1)
    review_scores_rating = min_rating
    review_scores_value = _safe_float(defaults.get("review_scores_value"), review_scores_rating)
    amenities_count = _safe_float(defaults.get("amenities_count"), float(sum(amenity_cols.values())))
    if min_available_days_30 > 0:
        availability_30 = int(min_available_days_30)
        availability_60 = min(60, availability_30 * 2)
        availability_90 = min(90, availability_30 * 3)
        availability_365 = min(365, max(availability_90 * 4, availability_30))
    else:
        availability_30 = _safe_int(defaults.get("availability_30"), fallback=15, minimum=0, maximum=30)
        availability_60 = _safe_int(defaults.get("availability_60"), fallback=30, minimum=0, maximum=60)
        availability_90 = _safe_int(defaults.get("availability_90"), fallback=45, minimum=0, maximum=90)
        availability_365 = _safe_int(defaults.get("availability_365"), fallback=180, minimum=0, maximum=365)

    overrides = {
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "beds": estimated_beds,
        "bathrooms": bathrooms,
        "bathrooms_per_guest": bathrooms / max(accommodates, 1),
        "beds_per_guest": estimated_beds / max(accommodates, 1),
        "room_type": room_type,
        "property_group": property_group,
        "amenities_count": amenities_count,
        "distance_to_center_km": distance_to_center_km,
        "is_central_5km": int(distance_to_center_km <= 5),
        "review_scores_rating": review_scores_rating,
        "review_scores_value": review_scores_value,
        "availability_30": availability_30,
        "availability_60": availability_60,
        "availability_90": availability_90,
        "availability_365": availability_365,
        "availability_30_ratio": availability_30 / 30,
        "availability_365_ratio": availability_365 / 365,
    }
    if neighbourhood:
        overrides["neighbourhood_cleansed"] = neighbourhood
    overrides.update({column: int(value) for column, value in amenity_cols.items()})

    selected_amenities = [label for column, label in amenity_labels.items() if amenity_cols.get(column)]
    candidates = _custom_price_candidates(
        city_df,
        accommodates=accommodates,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        room_type=room_type,
        property_group=property_group,
        neighbourhood=neighbourhood,
        selected_amenities=selected_amenities,
        min_rating=min_rating,
        min_available_days_30=min_available_days_30,
    )

    if not candidates.empty and "listing_id" in candidates.columns:
        calendar_summary = _calendar_result_for_candidates(city, candidates["listing_id"], check_in, check_out)
        prediction_listing_ids = (
            calendar_summary["available_ids"]
            if calendar_summary.get("available_ids")
            else candidates["listing_id"]
        )
    else:
        calendar_summary = {
            "calendar_checked": False,
            "candidate_count": 0,
            "available_count": 0,
            "available_ids": [],
            "nights": nights,
            "is_available": False,
            "status": "No matching snapshot listings were found for the selected filters.",
        }
        prediction_listing_ids = []

    candidate_count = int(len(candidates))
    verified_count = int(calendar_summary.get("available_count", 0))
    if verified_count and calendar_summary.get("available_ids") and "listing_id" in candidates.columns:
        verified_ids = set(int(value) for value in calendar_summary.get("available_ids", []))
        snapshot_pool = candidates[candidates["listing_id"].astype("int64").isin(verified_ids)].copy()
    else:
        snapshot_pool = candidates.copy()
    snapshot_median = (
        pd.to_numeric(snapshot_pool["price_eur"], errors="coerce").median()
        if not snapshot_pool.empty and "price_eur" in snapshot_pool.columns
        else None
    )
    snapshot_source_count = int(len(snapshot_pool)) if not snapshot_pool.empty else candidate_count
    snapshot_prediction = (
        _snapshot_price_estimate(
            city,
            snapshot_median,
            label="Snapshot-backed estimate",
            model_name="Cleaned snapshot median",
            listing_count=snapshot_source_count,
        )
        if calendar_summary.get("calendar_checked") and verified_count and snapshot_median is not None
        else None
    )
    precomputed_prediction = None
    if snapshot_prediction:
        prediction = snapshot_prediction
    else:
        precomputed_prediction = _precomputed_prediction_estimate(city, prediction_listing_ids, check_in, check_out)
        if precomputed_prediction:
            prediction = precomputed_prediction
        else:
            prediction = predict_price(city, overrides)
            prediction["price_source"] = "live_model_estimate"
            prediction["price_source_label"] = "Live model estimate"

    if calendar_summary.get("calendar_checked") and verified_count:
        availability_value = f"{verified_count:,} verified matches"
        availability_detail = f"From {candidate_count:,} snapshot listings matching your filters."
        availability_status = "good"
    elif calendar_summary.get("calendar_checked"):
        availability_value = "No verified matches"
        availability_detail = str(calendar_summary.get("status", "No matching listings pass the calendar rules."))
        availability_status = "warning"
    else:
        availability_value = "Not in calendar window"
        availability_detail = str(calendar_summary.get("coverage_text", calendar_summary.get("status", "Calendar was not checked.")))
        availability_status = "warning"

    if snapshot_prediction:
        price_source_detail = (
            f"Median listed nightly price across {snapshot_source_count:,} calendar-verified snapshot listings."
        )
        price_source_status = "good"
    elif precomputed_prediction:
        price_source_detail = f"Cached monthly prediction for {prediction.get('cache_month_window')}."
        price_source_status = "good"
    else:
        price_source_detail = "No matching snapshot-backed or cached future price was available, so the model ran on demand."
        price_source_status = "warning"

    price_source_cards = [
        {
            "title": "Availability",
            "value": availability_value,
            "detail": availability_detail,
            "status": availability_status,
        },
        {
            "title": "Snapshot price",
            "value": format_eur(snapshot_median),
            "detail": f"Median listed nightly price across {snapshot_source_count:,} matching snapshot listings.",
            "status": "neutral",
        },
        {
            "title": "Displayed price source",
            "value": str(prediction.get("price_source_label", "Live model estimate")),
            "detail": price_source_detail,
            "status": price_source_status,
        },
    ]

    scenario_rows = [
        {"Category": "Stay", "Item": "City", "Value": city},
        {"Category": "Stay", "Item": "Check-in", "Value": check_in.isoformat()},
        {"Category": "Stay", "Item": "Check-out", "Value": check_out.isoformat()},
        {"Category": "Stay", "Item": "Nights", "Value": f"{nights}"},
        {"Category": "Data match", "Item": "Snapshot listings matched", "Value": f"{candidate_count:,}"},
        {"Category": "Data match", "Item": "Calendar-verified matches", "Value": f"{verified_count:,}"},
        {"Category": "Stay", "Item": "Guests", "Value": f"{accommodates}"},
        {"Category": "Stay", "Item": "Bedrooms", "Value": f"{bedrooms}"},
        {"Category": "Stay", "Item": "Bathrooms", "Value": f"{bathrooms}"},
        {"Category": "Property", "Item": "Room type", "Value": str(room_type)},
        {"Category": "Property", "Item": "Property group", "Value": str(property_group)},
        {"Category": "Location", "Item": "Neighbourhood", "Value": str(neighbourhood or "Model default")},
        {"Category": "Location", "Item": "Neighbourhood avg. distance", "Value": f"{distance_to_center_km:.1f} km"},
        {"Category": "Quality", "Item": "Minimum rating", "Value": f"{min_rating:.1f}"},
        {
            "Category": "Availability",
            "Item": "Minimum days next 30",
            "Value": f"{min_available_days_30}" if min_available_days_30 > 0 else "No minimum selected",
        },
        {
            "Category": "Amenities",
            "Item": "Selected amenities",
            "Value": ", ".join(selected_amenities) if selected_amenities else "None selected",
        },
        {
            "Category": "Model assumption",
            "Item": "Additional amenities",
            "Value": "City/model default",
        },
        {
            "Category": "Model assumption",
            "Item": "Review score",
            "Value": f"Selected minimum: {_metric_text(review_scores_rating, decimals=2)}",
        },
    ]
    render_price_prediction_result(
        prediction,
        nights=nights,
        scenario_rows=scenario_rows,
        availability=calendar_summary,
        source_cards=price_source_cards,
    )


def render_chat_recommendation_card(card: dict) -> None:
    if card.get("type") == "section":
        title = escape(str(card.get("title", "Recommendation shortlist")))
        note = escape(str(card.get("note", "")))
        html = (
            '<div style="margin: 18px 0 8px 0;">'
            f'<div class="recommendation-title">{title}</div>'
            f'<div class="recommendation-meta">{note}</div>'
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
        return

    rank = int(card.get("rank", 0) or 0)
    neighbourhood = escape(str(card.get("neighbourhood", "Unknown neighbourhood")))
    city = escape(str(card.get("city", "")))
    room_type = escape(str(card.get("room_type", "n/a")))
    property_group = escape(str(card.get("property_group", "Property")))
    listing_id = escape(str(card.get("listing_id", "n/a")))
    price_text = escape(str(card.get("price_text", "n/a")))
    total_text = escape(str(card.get("total_text", "n/a")))
    nights = escape(str(card.get("nights", "n/a")))
    rating_text = escape(str(card.get("rating_text", "n/a")))
    value_text = escape(str(card.get("value_text", "n/a")))
    distance_text = escape(str(card.get("distance_text", "Distance n/a")))
    availability_text = escape(str(card.get("availability_text", "Availability n/a")))
    season_label = escape(str(card.get("season_label", "")))
    season_fit_text = escape(str(card.get("season_fit_text", "")))
    season_signal_text = escape(str(card.get("season_signal_text", "")))
    reason = escape(str(card.get("reason", "Strong balance of price, rating, location, and availability.")))
    badges = []
    if season_label and season_fit_text and season_signal_text:
        badges.extend(
            [
                f'<span class="badge badge-strong">{season_label} fit {season_fit_text}</span>',
                f'<span class="badge">{season_signal_text}</span>',
            ]
        )
    badges.extend(
        [
            f'<span class="badge badge-strong">Value score {value_text}</span>',
            f'<span class="badge">{price_text} / night</span>',
            f'<span class="badge">{total_text} total for {nights} nights</span>',
            f'<span class="badge">Rating {rating_text}</span>',
            f'<span class="badge">{distance_text}</span>',
            f'<span class="badge">{availability_text}</span>',
        ]
    )

    html = (
        '<div class="recommendation-card">'
        f'<div class="recommendation-title">#{rank} - {neighbourhood}</div>'
        f'<div class="recommendation-meta">{city} &middot; {room_type} &middot; {property_group}</div>'
        f'<div>{"".join(badges)}</div>'
        f'<div class="small-muted"><strong>Listing ID:</strong> {listing_id}</div>'
        f'<div class="small-muted" style="margin-top: 6px;"><strong>Why it fits:</strong> {reason}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_chat_message(message: dict) -> None:
    role = message.get("role", "assistant")
    avatar = FAVICON_PATH if role == "assistant" and FAVICON_PATH.exists() else None
    with st.chat_message(role, avatar=avatar):
        if message.get("content"):
            st.markdown(message["content"])
        for card in message.get("recommendation_cards", []):
            render_chat_recommendation_card(card)
        if message.get("footer"):
            st.markdown(message["footer"])


def process_chat_prompt(
    prompt: str,
    master_data: dict[str, pd.DataFrame],
    filters: dict,
    ml_outputs: dict[str, pd.DataFrame],
    rag_index: dict,
) -> None:
    prompt = prompt.strip()
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})

    context = backfill_chat_context_from_history(filters, master_data)
    fallback_city = chat_context_fallback_city(filters, master_data)
    prompt_city = infer_city_from_prompt(prompt, master_data, fallback=fallback_city)
    update_chat_context_from_prompt(prompt, prompt_city, filters, master_data)
    context = st.session_state.get("chat_context", {})
    intent_name = infer_chat_intent_from_prompt(prompt, context)
    context["last_intent"] = intent_name
    recommendation_cards = []
    recommendation_footer = None

    if intent_name == "price_factors":
        base_response, city = build_price_factor_response(
            prompt,
            master_data,
            ml_outputs,
            filters,
            context=context,
        )
    elif intent_name == "model_info":
        base_response = model_summary_text(ml_outputs)
        city = prompt_city
    elif intent_name == "seasonal_recommendation":
        base_response, response_city, recommendation_cards, recommendation_footer = build_seasonal_recommendation_response(
            prompt,
            master_data,
            filters,
            context=context,
            limit_per_season=3,
        )
        city = response_city
    elif intent_name == "timing":
        base_response, city = build_timing_response(prompt, master_data, filters, context)
    elif intent_name == "recommendation":
        base_response, response_city, recommendation_cards, recommendation_footer = build_chat_recommendation_response(
            prompt,
            master_data,
            filters,
            context=context,
            limit=5,
        )
        city = response_city
    else:
        city = prompt_city
        base_response = answer_from_data(prompt, master_data, city=city)

    if city in master_data:
        st.session_state.chat_context["city"] = city

    retrieved = cached_retrieved_context(prompt, city=city, top_k=5)
    sources = format_sources(retrieved)
    render_chat_message(st.session_state.messages[-1])

    response_chunks: list[str] = []
    avatar = FAVICON_PATH if FAVICON_PATH.exists() else None
    with st.chat_message("assistant", avatar=avatar):
        response_placeholder = st.empty()
        for chunk in stream_polished_answer_with_llm(
            question=prompt,
            base_answer=base_response,
            retrieved_context=context_text(retrieved),
            trip_context=context,
            has_recommendation_cards=bool(recommendation_cards),
        ):
            response_chunks.append(chunk)
            response_placeholder.markdown("".join(response_chunks) + "▌")

        response = "".join(response_chunks).strip() or base_response
        response_placeholder.markdown(response)

        for card in recommendation_cards:
            render_chat_recommendation_card(card)
        if recommendation_footer:
            st.markdown(recommendation_footer)

    message = {"role": "assistant", "content": response, "sources": sources}
    if recommendation_cards:
        message["recommendation_cards"] = recommendation_cards
    if recommendation_footer:
        message["footer"] = recommendation_footer
    st.session_state.messages.append(message)


def chat_view(
    master_data: dict[str, pd.DataFrame],
    filters: dict,
    ml_outputs: dict[str, pd.DataFrame],
    rag_index: dict,
) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi traveller, I'm Lilly, your AI travel intelligence advisor, I can help you find the perfect "
                    "accommodation for your holiday. Let me know what your budget, dates and must have amenities are."
                ),
            }
        ]

    if "quick_question_nonce" not in st.session_state:
        st.session_state.quick_question_nonce = 0

    st.markdown("### Here are some examples to get us started")

    quick_prompt = st.pills(
        "Example questions",
        EXAMPLE_QUESTIONS,
        selection_mode="single",
        key=f"quick_question_{st.session_state.quick_question_nonce}",
        label_visibility="collapsed",
        width="stretch",
    )

    for message in st.session_state.messages:
        render_chat_message(message)

    typed_prompt = st.chat_input("Where are we going, what is the budget, and what would make the stay feel right?")
    submitted_prompt = quick_prompt or typed_prompt
    if submitted_prompt:
        if quick_prompt:
            st.session_state.quick_question_nonce += 1
        process_chat_prompt(submitted_prompt, master_data, filters, ml_outputs, rag_index)


def market_view(master_data: dict[str, pd.DataFrame], filters: dict) -> None:
    st.subheader("Market Intelligence")

    if filters["city_choice"] == "Compare both":
        comparison = cached_city_comparison()
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        if not comparison.empty:
            chart_data = comparison.set_index("city")[["median_price_eur", "mean_price_eur"]]
            st.bar_chart(chart_data)
        return

    city = filters["city_choice"]
    df = master_data.get(city, pd.DataFrame())
    metric_row(cached_market_summary(city))

    areas = cached_neighbourhood_summary(city)
    st.markdown("Top neighbourhoods by value")
    st.dataframe(areas.head(20), use_container_width=True, hide_index=True)
    if not areas.empty:
        st.bar_chart(areas.head(12).set_index("neighbourhood_cleansed")["neighbourhood_value_score"])






def _find_reviews_file(city: str) -> Path | None:
    """Locate the reviews file across common project folders."""
    city_key = city.lower()
    bundle = load_all_data()
    paths = bundle.get("paths", {})

    candidates = []
    for key in ["raw", "data", "master"]:
        folder = paths.get(key)
        if folder:
            dated_review_files = sorted(
                [
                    *folder.glob(f"{city_key}_reviews_*.csv"),
                    *folder.glob(f"{city_key}_reviews_*.csv.gz"),
                ],
                key=lambda path: path.name,
                reverse=True,
            )
            candidates.extend(dated_review_files)
            candidates.extend([
                folder / f"{city_key}_reviews.csv",
                folder / f"{city_key}_reviews.csv.gz",
                folder / f"{city_key}_reviews_clean.csv",
            ])

    # Also support common local project layout used during exploration
    candidates.extend([
        Path("Data") / f"{city_key}_reviews.csv",
        Path("Data") / f"{city_key}_reviews.csv.gz",
        Path("1. Data") / "raw" / f"{city_key}_reviews.csv",
        Path("1. Data") / "raw" / f"{city_key}_reviews.csv.gz",
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


@st.cache_data(show_spinner=False)
def load_reviews_for_city(city: str) -> pd.DataFrame:
    review_file = _find_reviews_file(city)
    if review_file is None:
        return pd.DataFrame()

    try:
        reviews = pd.read_csv(review_file)
    except Exception:
        return pd.DataFrame()

    reviews.columns = reviews.columns.str.lower().str.strip()

    if "comments" not in reviews.columns:
        return pd.DataFrame()

    reviews["comments"] = reviews["comments"].astype(str)
    reviews = reviews[reviews["comments"].str.strip().ne("")].copy()

    if "date" in reviews.columns:
        reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")

    return reviews


def _keyword_sentiment_score(text: str) -> float:
    """Simple transparent sentiment proxy for capstone demo purposes."""
    positive_words = {
        "great", "excellent", "amazing", "perfect", "clean", "comfortable", "nice",
        "good", "friendly", "recommend", "wonderful", "beautiful", "convenient",
        "location", "quiet", "spacious", "helpful", "easy", "best", "love", "loved",
    }
    negative_words = {
        "bad", "dirty", "noisy", "small", "problem", "issue", "broken", "poor",
        "uncomfortable", "difficult", "late", "worst", "cold", "hot", "smell",
        "expensive", "disappointing", "not", "never", "complaint",
    }

    words = re.findall(r"[a-zA-Z]+", str(text).lower())
    if not words:
        return 0.0

    pos = sum(word in positive_words for word in words)
    neg = sum(word in negative_words for word in words)
    return (pos - neg) / max(len(words), 1)


def build_review_intelligence(reviews: pd.DataFrame, limit: int = 5000) -> dict:
    if reviews.empty or "comments" not in reviews.columns:
        return {
            "reviews": pd.DataFrame(),
            "summary": {},
            "top_terms": pd.DataFrame(),
            "samples_positive": pd.DataFrame(),
            "samples_negative": pd.DataFrame(),
        }

    sample = reviews.copy()
    if len(sample) > limit:
        sample = sample.sample(limit, random_state=42)

    sample["sentiment_score"] = sample["comments"].apply(_keyword_sentiment_score)

    def label(score: float) -> str:
        if score > 0.015:
            return "Positive"
        if score < -0.015:
            return "Negative"
        return "Neutral"

    sample["sentiment_label"] = sample["sentiment_score"].apply(label)

    summary = {
        "reviews_analyzed": int(len(sample)),
        "positive_share": round(float((sample["sentiment_label"] == "Positive").mean() * 100), 1),
        "neutral_share": round(float((sample["sentiment_label"] == "Neutral").mean() * 100), 1),
        "negative_share": round(float((sample["sentiment_label"] == "Negative").mean() * 100), 1),
        "avg_sentiment_score": round(float(sample["sentiment_score"].mean()), 4),
    }

    comments = sample["comments"].fillna("").astype(str)
    top_terms = pd.DataFrame()
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=30,
            ngram_range=(1, 2),
            min_df=5,
        )
        matrix = vectorizer.fit_transform(comments)
        scores = matrix.mean(axis=0).A1
        top_terms = (
            pd.DataFrame({"term": vectorizer.get_feature_names_out(), "importance": scores})
            .sort_values("importance", ascending=False)
            .head(15)
        )
        top_terms["importance"] = top_terms["importance"].round(4)
    except Exception:
        top_terms = pd.DataFrame(columns=["term", "importance"])

    samples_positive = (
        sample.sort_values("sentiment_score", ascending=False)
        .head(5)[["comments", "sentiment_score", "sentiment_label"]]
        .copy()
    )
    samples_negative = (
        sample.sort_values("sentiment_score", ascending=True)
        .head(5)[["comments", "sentiment_score", "sentiment_label"]]
        .copy()
    )

    for df in [samples_positive, samples_negative]:
        if not df.empty:
            df["comments"] = df["comments"].astype(str).str.slice(0, 280)

    return {
        "reviews": sample,
        "summary": summary,
        "top_terms": top_terms,
        "samples_positive": samples_positive,
        "samples_negative": samples_negative,
    }


def review_intelligence_view(master_data: dict[str, pd.DataFrame], filters: dict) -> None:
    st.subheader("Review Intelligence")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        This section analyzes guest review comments to extract customer experience signals.
        It helps identify what guests value most and where potential pain points appear.
        """
    )

    cities = available_cities(master_data)
    if not cities:
        st.warning("No cities available.")
        return

    default_city = filters["city_choice"] if filters["city_choice"] in cities else cities[0]
    city = st.selectbox("Review city", cities, index=cities.index(default_city))

    reviews = load_reviews_for_city(city)

    if reviews.empty:
        st.warning(
            "Review comments were not found in the expected project folders. "
            "Place `madrid_reviews.csv(.gz)` and `tokyo_reviews.csv(.gz)` in the raw data folder or Data folder."
        )
        return

    if "date" in reviews.columns and reviews["date"].notna().any():
        min_date = reviews["date"].min().date()
        max_date = reviews["date"].max().date()
        st.caption(f"Review date coverage: {min_date} to {max_date}")

    max_reviews = st.slider("Max reviews to analyze", 500, 10000, 5000, 500)
    intelligence = build_review_intelligence(reviews, limit=max_reviews)

    summary = intelligence["summary"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reviews analyzed", f"{summary.get('reviews_analyzed', 0):,}")
    col2.metric("Positive share", f"{summary.get('positive_share', 0)}%")
    col3.metric("Neutral share", f"{summary.get('neutral_share', 0)}%")
    col4.metric("Negative share", f"{summary.get('negative_share', 0)}%")

    st.markdown("### Sentiment Distribution")
    sentiment_counts = (
        intelligence["reviews"]["sentiment_label"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="reviews")
    )
    st.bar_chart(sentiment_counts.set_index("sentiment"))

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Top Review Topics")
        top_terms = intelligence["top_terms"]
        if top_terms.empty:
            st.info("Not enough review text to extract topics.")
        else:
            st.dataframe(top_terms, use_container_width=True, hide_index=True)
            st.bar_chart(top_terms.set_index("term")["importance"])

    with col_right:
        st.markdown("### Experience Insight")
        positive = summary.get("positive_share", 0)
        negative = summary.get("negative_share", 0)

        if positive >= 50:
            insight = (
                f"{city} shows a strong positive guest experience profile, "
                f"with {positive}% of analyzed reviews classified as positive by the transparent keyword-based proxy."
            )
        elif negative > 25:
            insight = (
                f"{city} shows a meaningful share of negative experience signals, "
                f"which should be explored further through topic-level analysis."
            )
        else:
            insight = (
                f"{city} shows a balanced review profile, with most comments falling into neutral or positive sentiment."
            )

        st.markdown(
            f"""
            <div class="blue-insight-card">
                <strong>Review intelligence insight:</strong><br>
                {insight}<br><br>
                These results should be interpreted as exploratory signals, not as a fully trained sentiment model.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Representative Positive Reviews")
    st.dataframe(intelligence["samples_positive"], use_container_width=True, hide_index=True)

    st.markdown("### Representative Negative / Risk Reviews")
    st.dataframe(intelligence["samples_negative"], use_container_width=True, hide_index=True)

    st.markdown("### How this supports the AI Agent")
    st.info(
        "Review intelligence can improve the agent by adding customer-experience context to recommendations, "
        "such as cleanliness, location convenience, noise, comfort, and host quality."
    )

def map_view(master_data: dict[str, pd.DataFrame], filters: dict) -> None:
    st.subheader("Interactive Map")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        Explore Airbnb listings geographically using the cleaned project data.
        The map highlights listing location, price, rating, and value score for the selected city.
        """
    )

    city_options = available_cities(master_data)
    if not city_options:
        st.warning("No city data available for mapping.")
        return

    default_city = filters["city_choice"] if filters["city_choice"] in city_options else city_options[0]
    map_city_col, map_room_col = st.columns(2)
    with map_city_col:
        city = st.selectbox("Map city", city_options, index=city_options.index(default_city))
    df = master_data.get(city, pd.DataFrame()).copy()

    required_cols = {"latitude", "longitude", "price_eur"}
    if df.empty or not required_cols.issubset(df.columns):
        st.warning("Map data is not available. Latitude, longitude, and price fields are required.")
        return

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["price_eur"] = pd.to_numeric(df["price_eur"], errors="coerce")

    if "review_scores_rating" in df.columns:
        df["review_scores_rating"] = pd.to_numeric(df["review_scores_rating"], errors="coerce")
    else:
        df["review_scores_rating"] = None

    for optional_numeric in ["availability_30", "number_of_reviews", "minimum_nights", "value_score"]:
        if optional_numeric in df.columns:
            df[optional_numeric] = pd.to_numeric(df[optional_numeric], errors="coerce")

    if "value_score" not in df.columns:
        try:
            from analytics import add_value_score
            df = add_value_score(df)
        except Exception:
            df["value_score"] = None

    map_data = df.dropna(subset=["latitude", "longitude", "price_eur"]).copy()

    if map_data.empty:
        st.warning("No valid geo-located listings found for the selected city.")
        return

    room_filter = "Any"
    if "room_type" in map_data.columns:
        room_options = ["Any", *sorted(map_data["room_type"].dropna().astype(str).unique().tolist())]
        with map_room_col:
            room_filter = st.selectbox("Map room type", room_options)

    max_price_default = int(min(500, max(50, map_data["price_eur"].quantile(0.95))))
    price_col, rating_col, availability_col, reviews_col, limit_col = st.columns([1.35, 0.95, 0.95, 0.85, 0.85])
    with price_col:
        min_price, max_price = st.slider(
            "Price range (snapshot EUR)",
            min_value=int(max(0, map_data["price_eur"].min())),
            max_value=int(max(50, map_data["price_eur"].quantile(0.98))),
            value=(0, max_price_default),
            step=10,
        )
    with rating_col:
        min_rating = st.slider("Minimum map rating", 0.0, 5.0, 4.0, 0.1, key="map_min_rating")
        include_unrated = st.checkbox("Include unrated", value=False, key="map_include_unrated")
    with availability_col:
        min_available_days = st.slider(
            "Min days next 30",
            0,
            30,
            max(1, int(filters.get("min_available_days_30", 0) or 0)),
            1,
            key="map_min_available_days",
        )
    with reviews_col:
        min_reviews = st.slider("Min reviews", 0, 50, 1, 1, key="map_min_reviews")
    with limit_col:
        max_points = st.slider("Max listings on map", 100, 3000, 800, 100)

    map_data = map_data[
        (map_data["price_eur"] >= min_price)
        & (map_data["price_eur"] <= max_price)
    ].copy()

    if "review_scores_rating" in map_data.columns:
        if include_unrated:
            map_data = map_data[
                (map_data["review_scores_rating"] >= min_rating) | (map_data["review_scores_rating"].isna())
            ].copy()
        else:
            map_data = map_data[map_data["review_scores_rating"] >= min_rating].copy()

    if "availability_30" in map_data.columns:
        map_data = map_data[map_data["availability_30"].fillna(0) >= min_available_days].copy()

    if "number_of_reviews" in map_data.columns:
        map_data = map_data[map_data["number_of_reviews"].fillna(0) >= min_reviews].copy()

    if room_filter != "Any" and "room_type" in map_data.columns:
        map_data = map_data[map_data["room_type"].astype(str) == room_filter].copy()

    filtered_matches = len(map_data)
    if filtered_matches > max_points:
        map_data = map_data.sample(n=max_points, random_state=42).copy()

    if map_data.empty:
        st.warning("No listings match the current map filters.")
        return

    rating_display = map_data["review_scores_rating"].round(2).astype("string").fillna("Unrated")
    reviews_display = (
        map_data["number_of_reviews"].fillna(0).round(0).astype("int64").astype(str)
        if "number_of_reviews" in map_data.columns
        else pd.Series("n/a", index=map_data.index)
    )
    availability_display = (
        map_data["availability_30"].fillna(0).round(0).astype("int64").astype(str)
        if "availability_30" in map_data.columns
        else pd.Series("n/a", index=map_data.index)
    )
    min_nights_display = (
        map_data["minimum_nights"].fillna(0).round(0).astype("int64").astype(str)
        if "minimum_nights" in map_data.columns
        else pd.Series("n/a", index=map_data.index)
    )
    listing_id_display = (
        map_data["listing_id"].fillna(0).round(0).astype("int64").astype(str)
        if "listing_id" in map_data.columns
        else pd.Series("n/a", index=map_data.index)
    )

    map_data["tooltip"] = (
        "Neighbourhood: " + map_data.get("neighbourhood_cleansed", pd.Series("n/a", index=map_data.index)).astype(str)
        + "<br>Price: EUR " + map_data["price_eur"].round(0).astype(str)
        + "<br>Rating: " + rating_display.astype(str)
        + "<br>Reviews: " + reviews_display
        + "<br>Availability next 30: " + availability_display
        + "<br>Minimum nights: " + min_nights_display
        + "<br>Room type: " + map_data.get("room_type", pd.Series("n/a", index=map_data.index)).astype(str)
        + "<br>Listing ID: " + listing_id_display
    )

    center_lat = float(map_data["latitude"].median())
    center_lon = float(map_data["longitude"].median())

    col1, col2, col3 = st.columns(3)
    col1.metric("Mapped listings", f"{len(map_data):,}")
    col2.metric("Median price", format_eur(map_data["price_eur"].median()))
    col3.metric("Avg. rating", round(float(map_data["review_scores_rating"].mean()), 2) if map_data["review_scores_rating"].notna().any() else "n/a")
    if filtered_matches > max_points:
        st.caption(f"Showing a representative sample of {max_points:,} listings from {filtered_matches:,} matches after map filters.")

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position="[longitude, latitude]",
        get_radius=35,
        get_fill_color="[0, 51, 141, 130]",
        radius_min_pixels=1.5,
        radius_max_pixels=8,
        pickable=True,
        opacity=0.68,
    )

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=11,
        pitch=0,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"html": "{tooltip}", "style": {"backgroundColor": "#00338D", "color": "white"}},
        map_style="light",
    )

    st.pydeck_chart(deck, use_container_width=True, height=720)

    st.markdown("### Top neighbourhoods in the mapped selection")
    if "neighbourhood_cleansed" in map_data.columns:
        area_summary = (
            map_data.groupby("neighbourhood_cleansed")
            .agg(
                listings=("price_eur", "size"),
                median_price_eur=("price_eur", "median"),
                mean_rating=("review_scores_rating", "mean"),
            )
            .reset_index()
            .sort_values(["listings", "mean_rating"], ascending=[False, False])
            .head(10)
        )
        area_summary["median_price_eur"] = area_summary["median_price_eur"].round(2)
        area_summary["mean_rating"] = area_summary["mean_rating"].round(2)
        render_overview_table(
            area_summary,
            {
                "neighbourhood_cleansed": "Neighbourhood",
                "listings": "Listings",
                "median_price_eur": "Median price",
                "mean_rating": "Mean rating",
            },
        )
    else:
        st.info("Neighbourhood field is not available for this map selection.")

def model_view(ml_outputs: dict[str, pd.DataFrame], filters: dict) -> None:
    st.subheader("Model Performance")
    st.markdown(model_summary_text(ml_outputs))

    leaderboard = model_leaderboard(ml_outputs)
    if not leaderboard.empty:
        st.markdown("Top 3 models by city")
        st.caption("Each city is ranked separately by RMSE, so the highlighted #1 row is the best model for that city's market.")
        render_city_model_leaderboard(leaderboard, limit_per_city=3)

    st.info(prediction_status())


def data_sources_view(bundle: dict) -> None:
    st.subheader("Data Lineage")
    st.markdown("Loaded dataset sizes")
    rows = []
    for layer in ["master", "model_ready"]:
        for city, df in bundle[layer].items():
            rows.append({"layer": layer, "city": city, "rows": len(df), "columns": len(df.columns)})
    render_loaded_dataset_size_table(pd.DataFrame(rows))

    inventory = data_inventory()
    if not inventory.empty:
        st.markdown("Data lineage")
        display_inventory = inventory.copy()
        if "path" in display_inventory.columns:
            project_root = Path(__file__).resolve().parents[2]
            display_inventory["path"] = display_inventory["path"].apply(
                lambda value: str(Path(value).resolve().relative_to(project_root))
                if Path(value).exists() and str(Path(value).resolve()).startswith(str(project_root))
                else str(value)
            )
        render_data_lineage_inventory_table(display_inventory)
    else:
        st.info("No dataset inventory is currently available.")


DATA_LINEAGE_LAYER_HELP = {
    "raw": "Original source files kept as downloaded or lightly staged. These are the starting point before cleaning, merging, feature engineering, or modelling.",
    "master": "Cleaned city-level datasets with rich attributes for EDA, chatbot recommendations, map views, RAG context, and user-facing exploration.",
    "model_ready": "Machine-learning-ready datasets after cleaning, feature engineering, encoding, and target preparation. These feed the supervised price models.",
    "ml_outputs": "Generated model outputs such as performance metrics, feature importance, tuning results, clustering outputs, and comparison tables.",
    "chatbot_outputs": "Files generated for the Streamlit chatbot, including reusable price-model artifacts and the local calendar snapshot database.",
}


def _data_lineage_file_description(layer: str, file_name: str) -> str:
    layer_text = DATA_LINEAGE_LAYER_HELP.get(
        str(layer),
        "Project data artifact used somewhere in the analysis, modelling, or chatbot workflow.",
    )
    lowered = str(file_name).lower()

    if "master_model_dataset" in lowered:
        detail = "This is the main cleaned city dataset used for recommendations, maps, market summaries, and chatbot context."
    elif "model_ready" in lowered:
        detail = "This is the transformed modelling table used for training and evaluating machine-learning models."
    elif "calendar_snapshot" in lowered:
        detail = "This local SQLite cache supports date availability checks without repeatedly reading large raw calendar files."
    elif "calendar" in lowered:
        detail = "Calendar data supports availability checks and seasonal availability signals."
    elif "review" in lowered:
        detail = "Review data supports review counts, recency, and customer-experience context where available."
    elif "listing" in lowered:
        detail = "Listing data is the core Airbnb property information used to build master and model-ready datasets."
    elif "feature_importance" in lowered:
        detail = "Feature-importance output explains which attributes the price models relied on most."
    elif "model_results" in lowered or "leaderboard" in lowered:
        detail = "Model-results output stores RMSE, MAE, R2, and ranking information for comparing supervised models."
    elif "comparison" in lowered:
        detail = "Comparison output is used to compare city-specific models, baseline models, or earlier combined-data approaches."
    elif "error" in lowered:
        detail = "Error-analysis output identifies where the model performs less well and where predictions need more caution."
    elif "tuning" in lowered:
        detail = "Tuning output records hyperparameter-search results for models such as XGBoost or Random Forest."
    elif "cluster" in lowered or "kmeans" in lowered or "gmm" in lowered or "agglomerative" in lowered:
        detail = "Clustering output supports unsupervised segmentation and comparison of listing groups."
    elif "dictionary" in lowered:
        detail = "Dictionary output documents columns, units, and descriptions so the datasets are easier to interpret."
    else:
        detail = "This file is part of the project data inventory and is shown for traceability."

    return f"{detail} {layer_text}"


def _loaded_dataset_description(layer: str, city: str, rows: int, columns: int) -> str:
    if layer == "master":
        return (
            f"{city} master dataset: {rows:,} rows and {columns:,} columns. "
            "Used for city EDA, recommendations, maps, calendar-aware filtering, and RAG knowledge summaries."
        )
    if layer == "model_ready":
        return (
            f"{city} model-ready dataset: {rows:,} rows and {columns:,} columns. "
            "Used to train, evaluate, and run the supervised price prediction models."
        )
    return f"{city} {layer} dataset with {rows:,} rows and {columns:,} columns."


def _format_info_table_value(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if abs(number - round(number)) < 0.005:
            return f"{number:,.0f}"
        return f"{number:,.2f}"
    return str(value)


def render_table_with_info_tooltips(
    df: pd.DataFrame,
    columns: list[str],
    info_builder,
) -> None:
    if df.empty:
        return

    selected_columns = [column for column in columns if column in df.columns]
    header_cells = "".join(f"<th>{escape(_pretty_column_label(column))}</th>" for column in selected_columns)
    header_cells += "<th>Info</th>"

    rows = []
    for _, row in df.iterrows():
        help_text = info_builder(row)
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{escape(_format_info_table_value(row[column]))}</td>"
                for column in selected_columns
            )
            + '<td class="knowledge-info-cell">'
            + '<span class="kb-help-wrap">'
            + '<span class="kb-help" tabindex="0" aria-label="Row information">?</span>'
            + f'<span class="kb-help-tooltip">{escape(help_text)}</span>'
            + "</span>"
            + "</td>"
            + "</tr>"
        )

    st.markdown(
        f"""
        <div class="overview-table-wrap knowledge-table-wrap">
            <table class="overview-table knowledge-table">
                <thead><tr>{header_cells}</tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_lineage_inventory_table(inventory: pd.DataFrame) -> None:
    render_table_with_info_tooltips(
        inventory,
        ["layer", "file", "size_mb", "path"],
        lambda row: _data_lineage_file_description(str(row.get("layer", "")), str(row.get("file", ""))),
    )


def render_loaded_dataset_size_table(sizes: pd.DataFrame) -> None:
    render_table_with_info_tooltips(
        sizes,
        ["layer", "city", "rows", "columns"],
        lambda row: _loaded_dataset_description(
            str(row.get("layer", "")),
            str(row.get("city", "")),
            int(row.get("rows", 0)),
            int(row.get("columns", 0)),
        ),
    )


KNOWLEDGE_CATEGORY_HELP = {
    "data_dictionary": (
        "Column definitions, units, and feature decisions. Used when the chatbot explains what fields mean "
        "or why an attribute was kept for modelling."
    ),
    "neighbourhood": (
        "Neighbourhood-level summaries: price, rating, value score, distance, and availability. Used for "
        "area recommendations and questions like where to stay."
    ),
    "model_result": (
        "Performance records for each supervised price model, including RMSE, MAE, and R2. Used when users ask "
        "which model performs best."
    ),
    "room_type": (
        "Market summaries by room type, such as entire home, private room, shared room, or hotel room. Used for "
        "privacy and price trade-off questions."
    ),
    "project_context": (
        "High-level project notes, data-layer explanations, and snapshot-data limitations. Used to explain what "
        "the app can and cannot claim."
    ),
    "market_summary": (
        "City-level market snapshots covering listing counts, median price, mean price, ratings, availability, "
        "and distance. Used for Madrid versus Tokyo comparisons."
    ),
    "model_summary": (
        "Short summaries of the current best model per city. Used for quick model-performance explanations."
    ),
    "feature_importance": (
        "Important model features from the trained price models. Used to explain what tends to influence the "
        "predicted nightly price."
    ),
    "model_error": (
        "Segments where the model makes larger errors. Used to explain model limitations and where predictions "
        "should be treated more carefully."
    ),
    "model_comparison": (
        "Comparison between city-specific models and earlier combined-data models. Used to explain whether the "
        "city-specific approach improved performance."
    ),
}


def _knowledge_category_description(category: str, city: str) -> str:
    base = KNOWLEDGE_CATEGORY_HELP.get(
        str(category),
        "Knowledge-base documents used to support chatbot answers with project data.",
    )
    if city and city != "All":
        return f"{base} This row is specific to {city}."
    return f"{base} This row applies across the project rather than one city only."


def render_knowledge_category_table(category_summary: pd.DataFrame) -> None:
    if category_summary.empty:
        return

    header_cells = "".join(
        f"<th>{label}</th>"
        for label in ["Category", "City", "Documents", "Info"]
    )
    rows = []
    for row in category_summary.itertuples(index=False):
        category = str(getattr(row, "category", ""))
        city = str(getattr(row, "city", ""))
        documents = getattr(row, "documents", 0)
        help_text = _knowledge_category_description(category, city)
        rows.append(
            "<tr>"
            f"<td>{escape(category)}</td>"
            f"<td>{escape(city)}</td>"
            f"<td>{int(documents):,}</td>"
            '<td class="knowledge-info-cell">'
            '<span class="kb-help-wrap">'
            '<span class="kb-help" tabindex="0" aria-label="Category information">?</span>'
            f'<span class="kb-help-tooltip">{escape(help_text)}</span>'
            "</span>"
            "</td>"
            "</tr>"
        )

    st.markdown(
        f"""
        <div class="overview-table-wrap knowledge-table-wrap">
            <table class="overview-table knowledge-table">
                <thead><tr>{header_cells}</tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def rag_sources_view(rag_index: dict) -> None:
    st.subheader("Knowledge Base")
    docs = rag_index.get("documents", pd.DataFrame())
    if docs.empty:
        st.warning("No RAG documents are currently available.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Retrieved documents", f"{len(docs):,}")
    col2.metric("Categories", docs["category"].nunique())
    col3.metric("Sources", docs["source"].nunique())

    st.markdown("Sample knowledge-base entries")
    sample_docs = docs[["title", "category", "city", "source", "content"]].head(5).copy()
    sample_docs["content"] = sample_docs["content"].astype(str).str.slice(0, 260)
    sample_docs.loc[sample_docs["content"].str.len() == 260, "content"] += "..."
    render_theme_table(
        sample_docs,
        ["title", "category", "city", "source", "content"],
    )

    st.markdown("Document categories")
    category_summary = (
        docs.groupby(["category", "city"])
        .size()
        .reset_index(name="documents")
        .sort_values("documents", ascending=False)
    )
    render_knowledge_category_table(category_summary)



def executive_overview(master_data: dict[str, pd.DataFrame]) -> None:
    st.markdown("### Market Snapshot")

    comparison = cached_city_comparison()

    if comparison.empty:
        st.warning("No market data available.")
        return

    total_listings = int(comparison["listings"].sum())
    avg_median_price = comparison["median_price_eur"].mean()
    avg_rating = comparison["mean_rating"].mean()
    cities = comparison["city"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Cities analyzed", cities, "Madrid and Tokyo", "🌍")
    with col2:
        render_kpi_card("Total listings", f"{total_listings:,.0f}", "Cleaned master data", "🏠")
    with col3:
        render_kpi_card("Avg. median price", format_eur(avg_median_price), "Across selected cities", "💰")
    with col4:
        render_kpi_card("Avg. rating", round(avg_rating, 2), "Guest satisfaction", "⭐")

    st.markdown("### What this advisor does")
    st.markdown(
        """
        The **Airbnb Intelligent Advisor** helps users explore Madrid and Tokyo using
        cleaned Airbnb data, explainable recommendation logic, local calendar checks,
        price prediction models, and a lightweight RAG knowledge layer.
        """
    )

    st.markdown("### Market Comparison")
    market_columns = {
        "city": "City",
        "listings": "Listings",
        "median_price_eur": "Median price",
        "mean_price_eur": "Mean price",
        "mean_rating": "Mean rating",
        "median_availability_30": "Median availability next 30 days",
    }
    available_market_columns = {
        column: label for column, label in market_columns.items() if column in comparison.columns
    }
    render_overview_table(comparison, available_market_columns)

    st.markdown("### Top Value Neighbourhood Preview")
    col_madrid, col_tokyo = st.columns(2)

    with col_madrid:
        st.markdown("#### Madrid")
        madrid_areas = cached_neighbourhood_summary("Madrid").head(6)
        if madrid_areas.empty:
            st.info("No Madrid neighbourhood summary available.")
        else:
            render_overview_table(
                madrid_areas,
                {
                    "neighbourhood_cleansed": "Neighbourhood",
                    "neighbourhood_value_score": "Value score",
                    "median_price_eur": "Median price",
                    "mean_rating": "Mean rating",
                },
            )

    with col_tokyo:
        st.markdown("#### Tokyo")
        tokyo_areas = cached_neighbourhood_summary("Tokyo").head(6)
        if tokyo_areas.empty:
            st.info("No Tokyo neighbourhood summary available.")
        else:
            render_overview_table(
                tokyo_areas,
                {
                    "neighbourhood_cleansed": "Neighbourhood",
                    "neighbourhood_value_score": "Value score",
                    "median_price_eur": "Median price",
                    "mean_rating": "Mean rating",
                },
            )


def analytics_view(
    master_data: dict[str, pd.DataFrame],
    ml_outputs: dict[str, pd.DataFrame],
    rag_index: dict,
    bundle: dict,
    filters: dict,
) -> None:
    executive_overview(master_data)
    st.divider()
    model_view(ml_outputs, filters)
    st.divider()
    rag_sources_view(rag_index)
    st.divider()
    data_sources_view(bundle)


NAV_OPTIONS = ["AI Advisor", "Recommendations", "Price Check", "Interactive Map", "Developer View"]


def set_active_view(view: str, analytics_section: str | None = None) -> None:
    st.session_state["active_view"] = view
    if analytics_section:
        st.session_state["analytics_section"] = analytics_section


def render_navigation() -> str:
    active_view = st.session_state.get("active_view", "AI Advisor")
    if active_view == "Price Prediction":
        active_view = "Price Check"
    elif active_view == "Analytics":
        active_view = "Developer View"
    if active_view not in NAV_OPTIONS:
        active_view = "AI Advisor"

    st.session_state["active_view"] = active_view

    _, nav_holder, _ = st.columns([0.08, 0.84, 0.08])
    with nav_holder:
        nav_cols = st.columns(len(NAV_OPTIONS), gap="small", vertical_alignment="top")
        for column, option in zip(nav_cols, NAV_OPTIONS):
            with column:
                button_type = "primary" if active_view == option else "secondary"
                key = f"nav_button_{option.lower().replace(' ', '_')}"
                if st.button(option, key=key, type=button_type, use_container_width=True):
                    set_active_view(option)
                    active_view = option

    return active_view


def main() -> None:
    start_llm_warmup()
    bundle = get_bundle()
    master_data = bundle["master"]
    ml_outputs = bundle["ml_outputs"]
    rag_index = get_rag_index()
    filters = default_filters(master_data)
    precompute_common_recommendation_cache()

    banner_uri = image_data_uri(BANNER_PATH)
    if banner_uri:
        st.markdown(
            f"""
            <div class="advisor-banner">
                <img src="{banner_uri}" alt="{escape(APP_NAME)} banner">
                <div class="disclaimer-help-wrap" tabindex="0" aria-label="Data limitation">
                    <div class="disclaimer-help">?</div>
                    <div class="disclaimer-tooltip">{escape(SNAPSHOT_DISCLAIMER)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="advisor-hero">
                <div class="advisor-pill">GEN AI · DATA SCIENCE · AGENTIC ANALYTICS</div>
                <h1>{APP_NAME}</h1>
                <p><strong>Data-backed AI advisor for Airbnb market intelligence in Madrid and Tokyo.</strong></p>
                <p>This solution combines market analytics, neighbourhood value scoring, price prediction,
                availability checks, and RAG-based conversational intelligence.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    active_view = render_navigation()

    if active_view == "AI Advisor":
        chat_view(master_data, filters, ml_outputs, rag_index)
    elif active_view == "Recommendations":
        recommendation_view(master_data, filters)
    elif active_view == "Price Check":
        price_prediction_view(master_data, filters)
    elif active_view == "Interactive Map":
        map_view(master_data, filters)
    else:
        analytics_view(master_data, ml_outputs, rag_index, bundle, filters)


if __name__ == "__main__":
    main()
