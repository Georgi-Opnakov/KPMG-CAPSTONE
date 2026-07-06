# Lilly Airbnb Intelligent Advisor - GitHub Sample Package

This folder is a GitHub-friendly version of the project. It keeps the same folder structure as the full submission, but large datasets have been replaced with small sampled files using the same filenames where the app/notebooks expect them.

## What Is Sampled

- `1. Data/raw/`: sampled raw listings, reviews, and calendar files.
- `1. Data/master/`: sampled Madrid and Tokyo master datasets.
- `1. Data/model_ready/`: sampled model-ready datasets aligned to the sampled listing IDs.
- `1. Data/Outputs/ml_models/`: large prediction outputs are sampled; small metrics/summary files are copied as-is.
- `1. Data/Outputs/chatbot/calendar_snapshot.sqlite`: compact SQLite sample with the same calendar tables.
- `1. Data/Outputs/chatbot/prediction_cache/`: sampled listing/monthly prediction caches.
- `1. Data/Outputs/chatbot/models/`: saved model artifacts are retained where small; large feature lookup CSVs are sampled.

## How To Run

From `4. Agent/Version 3.0` run:

```powershell
pip install -r requirements.txt
python -m streamlit run app.py --server.port 8501
```

Or double-click:

```text
4. Agent/Version 3.0/Launch Airbnb Chatbot.bat
```

## Important Caveat

The GitHub package is for code review, demo structure, and lightweight reproducibility. Results will differ from the full project because the data is sampled. For final metrics, model training, and production-quality recommendations, use the full project data package.
