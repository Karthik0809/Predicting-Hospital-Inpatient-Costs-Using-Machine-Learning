from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Paths
    data_path: str = "data/Hospital_Inpatient_Discharges__SPARCS_De-Identified___2012_20240601.csv"
    artifacts_dir: str = "artifacts"

    # Data
    sample_size: int = 50000
    random_state: int = 42
    test_size: float = 0.15
    val_size: float = 0.15

    # Training
    n_trials: int = 30               # Optuna trials per model

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "hospital-cost-prediction"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Streamlit
    api_url: str = "http://localhost:8000"

    @property
    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)

    @property
    def data_file_path(self) -> Path:
        return Path(self.data_path)


settings = Settings()
