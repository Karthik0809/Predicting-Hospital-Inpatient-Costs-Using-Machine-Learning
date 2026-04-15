.PHONY: setup install train train-fast api ui mlflow all docker-build docker-up docker-down clean help

# ── Python / local ────────────────────────────────────────────────────────────
setup:
	cp -n .env.example .env || true
	mkdir -p artifacts data

install:
	pip install --upgrade pip
	pip install -r requirements.txt

train: setup
	python train.py

train-fast: setup
	python train.py --no-optuna

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run app/streamlit_app.py

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

all: train
	@echo "Training done. Run 'make api' and 'make ui' in separate terminals."

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker compose build

docker-train:
	docker compose --profile train run --rm trainer

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf artifacts/ mlflow.db __pycache__ src/__pycache__ \
	       src/data/__pycache__ src/models/__pycache__ src/api/__pycache__

help:
	@echo ""
	@echo "  Hospital Inpatient Cost Predictor — Makefile"
	@echo "  ─────────────────────────────────────────────"
	@echo "  Local dev:"
	@echo "    make install       Install Python dependencies"
	@echo "    make train         Train all models (with Optuna)"
	@echo "    make train-fast    Train all models (no Optuna, faster)"
	@echo "    make api           Start FastAPI on :8000"
	@echo "    make ui            Start Streamlit on :8501"
	@echo "    make mlflow-ui     Start MLflow UI on :5000"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build  Build images"
	@echo "    make docker-train  Run training inside Docker"
	@echo "    make docker-up     Start all services (API + UI + MLflow)"
	@echo "    make docker-down   Stop all services"
	@echo "    make docker-logs   Follow container logs"
	@echo ""
	@echo "  Other:"
	@echo "    make clean         Delete artifacts and caches"
	@echo ""
