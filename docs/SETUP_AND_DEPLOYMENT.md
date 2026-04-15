# Setup and Deployment Guide

This guide provides a clean sequence for local setup, model training, and deployment.

## 1) Prerequisites

- Python 3.11+
- Git
- (Optional) Docker Desktop for containerized deployment

## 2) Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3) Dataset

Place the SPARCS CSV file at:

`data/Hospital_Inpatient_Discharges__SPARCS_De-Identified___2012_20240601.csv`

If not present, synthetic demo data is generated automatically so the project still runs.

## 4) Train Models

```bash
python train.py
```

Faster run (no Optuna):

```bash
python train.py --no-optuna
```

Training outputs are written to `artifacts/`.

## 5) Run API

```bash
uvicorn src.api.main:app --reload --port 8000
```

- Swagger docs: `http://localhost:8000/docs`

## 6) Run Dashboard

```bash
streamlit run app/streamlit_app.py
```

- UI: `http://localhost:8501`

## 7) Run MLflow Tracking UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

- MLflow: `http://localhost:5000`

## 8) Docker Workflow

```bash
docker compose build
docker compose --profile train run --rm trainer
docker compose up -d
```

## 9) Environment Configuration

Copy `.env.example` to `.env` and adjust values as needed for your environment.

## 10) Common Operational Notes

- Re-run `python train.py --no-optuna` whenever you need a quick model refresh.
- Keep `artifacts/` in sync with your latest model training before deployment.
- Validate API health endpoint (`/health`) after deployment to ensure models loaded correctly.
