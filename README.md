# Hospital Inpatient Cost Predictor v2

Predicts total hospital inpatient costs from patient demographics and clinical data.
Rebuilt with a modern ML stack, REST API, interactive dashboard, and full Docker deployment.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| ML Models | XGBoost 2.x, LightGBM 4.x, Random Forest, Ridge |
| Deep Learning | PyTorch 2.x (MLP with BatchNorm + Dropout) |
| Hyperparameter Tuning | Optuna 4.x |
| Experiment Tracking | MLflow 2.x |
| Preprocessing | scikit-learn Pipeline / ColumnTransformer |
| API | FastAPI 0.115 + Uvicorn |
| Dashboard | Streamlit 1.40 + Plotly |
| Config | Pydantic-Settings v2 |
| Deployment | Docker + Docker Compose |

---

## Project Structure

```
.
├── src/
│   ├── config.py                # Centralised settings (env-driven)
│   ├── data/
│   │   ├── loader.py            # SPARCS CSV loader + synthetic fallback
│   │   └── preprocessor.py      # sklearn ColumnTransformer pipeline
│   ├── models/
│   │   ├── traditional.py       # Ridge / RF / XGBoost / LightGBM
│   │   ├── neural.py            # PyTorch MLP regressor
│   │   └── trainer.py           # Optuna tuning + MLflow logging
│   └── api/
│       ├── main.py              # FastAPI app (lifespan, CORS, routes)
│       └── schemas.py           # Pydantic v2 request/response models
├── app/
│   └── streamlit_app.py         # Interactive prediction dashboard
├── train.py                     # Training entry-point
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── pyproject.toml
```

---

## Documentation

- `docs/PROJECT_STRUCTURE.md` - Detailed folder and file purpose reference
- `docs/SETUP_AND_DEPLOYMENT.md` - End-to-end setup, run, and deployment guide

---

## Quick Start (local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Add real data
Download the SPARCS 2012 dataset from NY State DOH and place it at:
```
data/Hospital_Inpatient_Discharges__SPARCS_De-Identified___2012_20240601.csv
```
If the file is absent the project auto-generates **synthetic demo data** with
realistic distributions so everything still runs end-to-end.

### 3. Train models
```bash
python train.py              # full training with Optuna (recommended)
python train.py --no-optuna  # faster, no hyperparameter search
```

### 4. Start the API
```bash
uvicorn src.api.main:app --reload --port 8000
# Swagger UI → http://localhost:8000/docs
```

### 5. Start the dashboard
```bash
streamlit run app/streamlit_app.py
# → http://localhost:8501
```

### 6. Start MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://localhost:5000
```

---

## Docker Deployment

```bash
# Build images
docker compose build

# Train models inside Docker (writes to named volume)
docker compose --profile train run --rm trainer

# Start API + Dashboard + MLflow
docker compose up -d

# Check status / logs
docker compose ps
docker compose logs -f
```

| Service | URL |
|---------|-----|
| FastAPI (Swagger) | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |
| MLflow Tracking | http://localhost:5000 |

---

## API Reference

### `POST /predict`
```json
{
  "age_group": "50 to 69",
  "gender": "M",
  "length_of_stay": 7,
  "type_of_admission": "Emergency",
  "apr_severity_code": 3,
  "apr_severity_desc": "Major",
  "apr_risk_of_mortality": "Moderate",
  "apr_medical_surgical": "Surgical",
  "payment_typology": "Medicare",
  "health_service_area": "New York City",
  "hospital_county": "New York",
  "birth_weight": 0,
  "ccs_diagnosis_code": 108,
  "apr_drg_code": 300,
  "apr_mdc_code": 5
}
```

Response:
```json
{
  "predicted_cost": 42318.50,
  "model_used": "xgboost",
  "confidence_interval_lower": 35970.73,
  "confidence_interval_upper": 48666.28,
  "features_used": [...]
}
```

### Other endpoints
- `GET /health` — liveness + model-loaded status
- `GET /metrics` — per-model test metrics (R², RMSE, MAE)
- `GET /models` — list all trained models
- `POST /predict/batch` — batch predictions (up to 1 000 records)

---

## Model Pipeline

```
Raw CSV  →  loader.py  →  ColumnTransformer (impute + scale/encode)
                       →  XGBoost / LightGBM / RF / Neural Net
                       →  Optuna (30 trials each)
                       →  MLflow (params + metrics + artifacts)
                       →  Best model saved to artifacts/
                       →  FastAPI  ←→  Streamlit
```

---

## Improvements over v1

| Area | v1 | v2 |
|------|----|----|
| Models | Linear, GradBoost, RF, PyTorch NN | Ridge, RF, **XGBoost**, **LightGBM**, PyTorch MLP |
| Hyperparameter Search | None | **Optuna** (30 trials) |
| Experiment Tracking | None | **MLflow** |
| Preprocessing | Manual pandas ops | **scikit-learn Pipeline** (no leakage) |
| Neural Net | 2 hidden layers, Adam, 20 epochs | 3 hidden layers, **BatchNorm**, **Dropout**, **AdamW**, **CosineAnnealingLR**, early stopping |
| Code Structure | Single notebook | Modular `src/` package |
| API | None | **FastAPI** with Swagger UI |
| Dashboard | None | **Streamlit** + Plotly |
| Deployment | None | **Docker Compose** (API + UI + MLflow) |
| Config | Hardcoded | **Pydantic-Settings** + `.env` |
| Data fallback | None | Synthetic data generator |

---

## Dataset

**NY SPARCS Hospital Inpatient Discharges 2012** — 2.5 M records, 34 features.
Source: [health.data.ny.gov](https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De%20Identified/u4ud-w55t/about_data)

Target variable: **Total Costs** (actual resource cost, distinct from billed charges).

---

## Author

[**Karthik Mulugu**](https://www.linkedin.com/in/karthikmulugu/)

## License

MIT License — © 2025 Karthik Mulugu
