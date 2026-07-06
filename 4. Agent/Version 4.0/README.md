# Lilly Airbnb Intelligent Advisor

Lilly is a Streamlit travel-intelligence prototype for exploring Airbnb data in Madrid and Tokyo. It combines cleaned Inside Airbnb datasets, value-based recommendations, map exploration, calendar-aware checks, supervised price models, precomputed future price estimates, and a lightweight RAG knowledge layer.

The app is designed as a local demo: it runs from the CAPSTONE folder, reads the cleaned project outputs, and can optionally use a local Ollama/Qwen model to make answers warmer and more conversational.

## Current App Tabs

- **AI Advisor**: conversational holiday-planning assistant. It extracts intent and travel preferences from natural language, uses pandas/RAG/model outputs as the source of truth, then optionally asks Qwen to polish the final response.
- **Recommendations**: side-by-side destination comparison for Madrid and Tokyo, with Budget, In Style, and Luxury recommendation modes.
- **Price Check**: scenario-based price and availability check. It first uses local snapshot/calendar-backed evidence where possible, then uses precomputed or on-demand model estimates when the requested period is outside verified data coverage.
- **Interactive Map**: geographic listing exploration with city, price, rating, review, availability, and room-type filters.
- **Developer View**: combined project transparency view with market overview, model performance, knowledge-base samples, and data lineage.

## Architecture

```text
User message or app filters
  -> Streamlit interface
  -> optional Ollama/Qwen parsing and response polishing
  -> deterministic Python services
       - pandas filters cleaned city master datasets
       - recommendation engine ranks listings by value, price, distance, reviews, amenities, and availability
       - calendar service checks local future availability from the SQLite cache
       - prediction cache serves precomputed model estimates for common future periods
       - saved ML artifacts provide fallback/on-demand price estimates
       - RAG retrieves project context with TF-IDF similarity search
  -> grounded answer, cards, table, map, or price result shown in the app
```

The LLM does not invent listings or prices. It helps interpret user wording and make the response friendly. The factual content comes from the cleaned project data, cached calendar data, model artifacts, and generated knowledge base.

## Expected Project Structure

The app expects to sit inside the main CAPSTONE folder:

```text
CAPSTONE/
  1. Data/
    master/
    model_ready/
    Outputs/
      ml_models/
      chatbot/
        calendar_snapshot.sqlite
        models/
        prediction_cache/
  2. Code/
  4. Agent/
    Version 3.0/
      app.py
      data_loader.py
      run_chatbot.ps1
      requirements.txt
      assets/
```

The raw Inside Airbnb files are useful for rebuilding the full pipeline, but the app itself runs from the cleaned master/model-ready datasets plus generated chatbot/model outputs.

## Key App Data Dependencies

- `1. Data/master/madrid_master_model_dataset.csv`
- `1. Data/master/tokyo_master_model_dataset.csv`
- `1. Data/model_ready/madrid_model_ready.csv`
- `1. Data/model_ready/tokyo_model_ready.csv`
- `1. Data/Outputs/ml_models/`
- `1. Data/Outputs/chatbot/calendar_snapshot.sqlite`
- `1. Data/Outputs/chatbot/models/price_model_manifest.csv`
- `1. Data/Outputs/chatbot/models/property_groups/property_group_price_model_manifest.csv`
- `1. Data/Outputs/chatbot/prediction_cache/listing_price_prediction_cache.csv`
- `1. Data/Outputs/chatbot/prediction_cache/monthly_price_prediction_cache.csv`

## Machine Learning

The latest city-level leaderboard identified **LightGBM Tuned** as the best overall model for both Madrid and Tokyo. The app also keeps reusable city-level and property-group artifacts for fallback predictions and for future refinement.

The prediction cache is generated from saved artifacts:

- property-group model first, where available
- city-level fallback model where a property-group artifact is unavailable
- monthly cache for common future planning windows

Important limitation: the precomputed future estimates are model estimates, not live Airbnb quotes.

## Running The App

Double-click:

```text
Launch Airbnb Chatbot.bat
```

or run from this folder:

```powershell
.\run_chatbot.ps1
```

Manual command:

```powershell
python -m streamlit run app.py --server.port 8501
```

If port `8501` is already in use, the launcher opens the existing local app instead of starting a second server.

## Dependencies

Install the app dependencies from `4. Agent/Version 3.0`:

```powershell
pip install -r requirements.txt
```

The app is tested with the local Anaconda Python environment used during development.

## Optional Ollama/Qwen Layer

If Ollama is running locally with the configured model, Lilly uses it to parse intent and polish grounded answers:

```powershell
ollama run qwen2.5:7b
```

If Ollama is unavailable, the app still works using deterministic fallback responses.

## Rebuild Commands

Run these from `4. Agent/Version 3.0` only when the underlying data or models need to be rebuilt:

```powershell
python build_calendar_cache.py
python train_price_models.py
python train_property_group_price_models.py
python build_prediction_cache.py
```

## Data Limitation

This app uses cleaned snapshot data from the project. It should not be presented as live Airbnb availability or live Airbnb pricing unless a verified external live API is connected.
