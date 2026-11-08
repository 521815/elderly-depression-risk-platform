"""Summarize TSFEL-style gait feature tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_gait_table(path: Path | str) -> dict[str, float]:
    df = pd.read_csv(path)
    num = df.select_dtypes(include="number")
    if num.empty:
        return {"gait_rows": float(len(df)), "gait_features": 0.0}
    out = {"gait_rows": float(len(df)), "gait_features": float(num.shape[1])}
    for c in num.columns:
        out[f"{c}_mean"] = float(num[c].mean(skipna=True))
        out[f"{c}_std"] = float(num[c].std(skipna=True))
    return out
