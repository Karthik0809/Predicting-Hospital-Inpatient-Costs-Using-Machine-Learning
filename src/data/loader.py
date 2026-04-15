"""Data loading utilities for the SPARCS hospital inpatient dataset."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)

# ── Column definitions ──────────────────────────────────────────────────────
TARGET = "Total Costs"

NUMERICAL_FEATURES = [
    "Length of Stay",
    "Birth Weight",
    "APR Severity of Illness Code",
    "CCS Diagnosis Code",
    "APR DRG Code",
    "APR MDC Code",
]

CATEGORICAL_FEATURES = [
    "Age Group",
    "Gender",
    "Race",
    "Ethnicity",
    "Type of Admission",
    "APR Severity of Illness Description",
    "APR Risk of Mortality",
    "APR Medical Surgical Description",
    "Payment Typology 1",
    "Health Service Area",
    "Hospital County",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

AGE_GROUP_ORDER = ["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"]
SEVERITY_ORDER = ["Minor", "Moderate", "Major", "Extreme"]
MORTALITY_ORDER = ["Minor", "Moderate", "Major", "Extreme"]


def load_data(path: str | Path | None = None, sample_size: int | None = None) -> pd.DataFrame:
    """Load and minimally clean the SPARCS dataset.

    Falls back to synthetic demo data when the CSV is absent.
    """
    path = Path(path or settings.data_path)
    if path.exists():
        logger.info("Loading data from %s", path)
        df = pd.read_csv(path, low_memory=False)
        n = sample_size or settings.sample_size
        if len(df) > n:
            df = df.sample(n, random_state=settings.random_state)
        logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])
    else:
        logger.warning("Data file not found at %s — generating synthetic demo data.", path)
        df = _generate_synthetic_data(sample_size or settings.sample_size)

    df = _basic_clean(df)
    return df


# ── Private helpers ─────────────────────────────────────────────────────────

def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing spaces from string columns and coerce numeric targets."""
    # Strip column name whitespace
    df.columns = df.columns.str.strip()

    # Strip whitespace in object columns
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    # Coerce Length of Stay: "120 +" → 120
    if "Length of Stay" in df.columns:
        df["Length of Stay"] = (
            df["Length of Stay"].astype(str).str.replace(r"\D", "", regex=True)
        )
        df["Length of Stay"] = pd.to_numeric(df["Length of Stay"], errors="coerce")

    # Remove currency symbols from cost columns
    for col in ["Total Costs", "Total Charges"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(r"[$,]", "", regex=True)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where the target is missing or non-positive
    if TARGET in df.columns:
        df = df.dropna(subset=[TARGET])
        df = df[df[TARGET] > 0]

    return df.reset_index(drop=True)


def _generate_synthetic_data(n: int = 50_000) -> pd.DataFrame:
    """Generate a synthetic dataset with realistic distributions for demo purposes."""
    rng = np.random.default_rng(42)

    age_groups = rng.choice(AGE_GROUP_ORDER, n, p=[0.05, 0.10, 0.20, 0.35, 0.30])
    genders = rng.choice(["M", "F", "U"], n, p=[0.45, 0.54, 0.01])
    races = rng.choice(
        ["White", "Black/African American", "Multi", "Other Race", "Unknown"],
        n, p=[0.55, 0.18, 0.03, 0.12, 0.12],
    )
    ethnicities = rng.choice(
        ["Not Span/Hispanic", "Spanish/Hispanic", "Unknown", "Multi-ethnic"],
        n, p=[0.70, 0.15, 0.12, 0.03],
    )
    admission_types = rng.choice(
        ["Emergency", "Urgent", "Elective", "Newborn", "Trauma", "Not Available"],
        n, p=[0.40, 0.20, 0.25, 0.08, 0.05, 0.02],
    )
    severity = rng.choice(SEVERITY_ORDER, n, p=[0.25, 0.30, 0.28, 0.17])
    mortality = rng.choice(MORTALITY_ORDER, n, p=[0.45, 0.28, 0.18, 0.09])
    med_surg = rng.choice(
        ["Medical", "Surgical", "Not Applicable"], n, p=[0.60, 0.35, 0.05]
    )
    payment = rng.choice(
        ["Medicare", "Medicaid", "Blue Cross/Blue Shield", "Private Health Insurance",
         "Self-Pay", "Miscellaneous/Other", "Federal/State/Local/VA"],
        n, p=[0.35, 0.25, 0.15, 0.12, 0.05, 0.05, 0.03],
    )
    health_areas = rng.choice(
        ["New York City", "Long Island", "Hudson Valley", "Capital/Adiron",
         "Western NY", "Central NY", "Finger Lakes", "Southern Tier"],
        n, p=[0.40, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05, 0.05],
    )
    counties = rng.choice(
        ["New York", "Kings", "Queens", "Bronx", "Nassau", "Suffolk",
         "Westchester", "Erie", "Monroe", "Albany"],
        n,
    )

    severity_map = {"Minor": 1, "Moderate": 2, "Major": 3, "Extreme": 4}
    age_base_map = {
        "0 to 17": 3_000, "18 to 29": 5_000, "30 to 49": 8_000,
        "50 to 69": 12_000, "70 or Older": 15_000,
    }

    los = np.clip(rng.exponential(4, n) + 1, 1, 120).astype(int)
    sev_codes = np.array([severity_map[s] for s in severity])
    age_base = np.array([age_base_map[a] for a in age_groups])
    surg_mult = np.where(med_surg == "Surgical", 1.5, 1.0)

    total_costs = (
        age_base
        + los * 800 * sev_codes * surg_mult
        + rng.normal(0, 2000, n)
    ).clip(500, 500_000)

    df = pd.DataFrame(
        {
            "Age Group": age_groups,
            "Gender": genders,
            "Race": races,
            "Ethnicity": ethnicities,
            "Length of Stay": los,
            "Type of Admission": admission_types,
            "APR Severity of Illness Code": sev_codes,
            "APR Severity of Illness Description": severity,
            "APR Risk of Mortality": mortality,
            "APR Medical Surgical Description": med_surg,
            "Payment Typology 1": payment,
            "Health Service Area": health_areas,
            "Hospital County": counties,
            "Birth Weight": np.where(
                rng.random(n) < 0.05, rng.integers(500, 4500, n), 0
            ),
            "CCS Diagnosis Code": rng.integers(1, 260, n),
            "APR DRG Code": rng.integers(1, 950, n),
            "APR MDC Code": rng.integers(0, 25, n),
            "Total Costs": total_costs,
        }
    )
    return df
