"""Class-imbalance helpers for tabular and multimodal experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    arr = np.asarray(y, dtype=float)
    pos = float((arr == 1).sum())
    neg = float((arr == 0).sum())
    if pos == 0:
        return 1.0
    return neg / pos


def resample_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "smote",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Resample only training data. method: none/smote/smote_enn/adasyn/oversample/undersample."""
    method = (method or "none").lower()
    if method == "none":
        return X_train, y_train

    if method == "smote":
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(random_state=random_state)
    elif method == "smote_enn":
        from imblearn.combine import SMOTEENN

        sampler = SMOTEENN(random_state=random_state)
    elif method == "smote_tomek":
        from imblearn.combine import SMOTETomek

        sampler = SMOTETomek(random_state=random_state)
    elif method == "adasyn":
        from imblearn.over_sampling import ADASYN

        sampler = ADASYN(random_state=random_state)
    elif method == "oversample":
        from imblearn.over_sampling import RandomOverSampler

        sampler = RandomOverSampler(random_state=random_state)
    elif method == "undersample":
        from imblearn.under_sampling import RandomUnderSampler

        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError(f"Unknown resampling method: {method}")

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    return X_res, y_res


def best_threshold(y_true: pd.Series | np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> float:
    """Select threshold on validation data by Youden J or F1."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = float(((pred == 1) & (y_true == 1)).sum())
        fp = float(((pred == 1) & (y_true == 0)).sum())
        fn = float(((pred == 0) & (y_true == 1)).sum())
        tn = float(((pred == 0) & (y_true == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        score = f1 if metric == "f1" else (recall + specificity - 1)
        if score > best_score:
            best_score = score
            best_t = t
    return best_t


def focal_loss_torch(logits: Any, targets: Any, alpha: float = 0.25, gamma: float = 2.0) -> Any:
    """Focal loss for binary classification, used in MPDD deep models."""
    import torch
    import torch.nn.functional as F

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - pt) ** gamma * bce).mean()
