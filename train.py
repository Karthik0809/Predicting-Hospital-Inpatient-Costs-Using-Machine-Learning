#!/usr/bin/env python
"""Entry-point for training the hospital cost prediction models."""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

from src.models.trainer import train_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hospital cost prediction models.")
    parser.add_argument(
        "--no-optuna",
        action="store_true",
        help="Skip Optuna hyperparameter search and use default params.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Hospital Cost Prediction — Model Training")
    print("=" * 60 + "\n")

    metrics = train_all(use_optuna=not args.no_optuna)

    print("\n" + "=" * 60)
    print("  Final Test Metrics")
    print("=" * 60)
    print(f"{'Model':<20} {'R²':>8} {'RMSE':>12} {'MAE':>12}")
    print("-" * 60)
    for name, m in sorted(metrics.items(), key=lambda kv: -kv[1]["r2"]):
        print(f"{name:<20} {m['r2']:>8.4f} {m['rmse']:>12,.0f} {m['mae']:>12,.0f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
