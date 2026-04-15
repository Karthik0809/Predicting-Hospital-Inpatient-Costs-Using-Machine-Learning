"""PyTorch MLP for hospital cost regression (tabular data)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Feedforward network with BatchNorm + Dropout for tabular regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = (256, 128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class NeuralNetRegressor:
    """sklearn-compatible wrapper around the PyTorch MLP."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        epochs: int = 50,
        patience: int = 10,
        device: str | None = None,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_: MLP | None = None
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "NeuralNetRegressor":
        dev = torch.device(self.device)
        input_dim = X_train.shape[1]
        self.model_ = MLP(input_dim, list(self.hidden_dims), self.dropout).to(dev)

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val) if X_val is not None else None

        optimizer = AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.HuberLoss()

        best_val, patience_ctr = float("inf"), 0
        best_state: dict[str, Any] | None = None

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer, criterion, dev)
            scheduler.step()
            self.history_["train_loss"].append(train_loss)

            if val_loader:
                val_loss = self._eval_epoch(val_loader, criterion, dev)
                self.history_["val_loss"].append(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.patience:
                        logger.info("Early stopping at epoch %d", epoch)
                        break

            if epoch % 10 == 0:
                logger.info(
                    "Epoch %3d | train_loss=%.2f | val_loss=%.2f",
                    epoch,
                    train_loss,
                    self.history_["val_loss"][-1] if self.history_["val_loss"] else float("nan"),
                )

        if best_state:
            self.model_.load_state_dict(best_state)

        return self

    # ── predict ──────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model_ is not None, "Call fit() first."
        dev = torch.device(self.device)
        self.model_.eval()
        loader = self._make_loader(X, shuffle=False)
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for (xb,) in loader:
                preds.append(self.model_(xb.to(dev)).cpu().numpy())
        return np.concatenate(preds)

    # ── internals ────────────────────────────────────────────────────────────

    def _make_loader(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        shuffle: bool = False,
    ) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            y_t = torch.tensor(y, dtype=torch.float32)
            ds = TensorDataset(X_t, y_t)
        else:
            ds = TensorDataset(X_t)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        dev: torch.device,
    ) -> float:
        self.model_.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            loss = criterion(self.model_(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(xb)
        return total / len(loader.dataset)

    def _eval_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        dev: torch.device,
    ) -> float:
        self.model_.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(dev), yb.to(dev)
                total += criterion(self.model_(xb), yb).item() * len(xb)
        return total / len(loader.dataset)

    # ── optional Optuna space ────────────────────────────────────────────────

    @staticmethod
    def optuna_params(trial: Any) -> dict[str, Any]:
        depth = trial.suggest_int("depth", 2, 4)
        width = trial.suggest_categorical("width", [64, 128, 256])
        return {
            "hidden_dims": tuple(width // (2**i) for i in range(depth)),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "epochs": 60,
            "patience": 10,
        }
