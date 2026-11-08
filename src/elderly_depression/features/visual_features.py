"""Summarize OpenFace-style facial feature tables."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _pick_cols(df: pd.DataFrame, patterns: list[str]) -> list[str]:
    out = []
    for c in df.columns:
        low = c.lower()
        if any(re.search(p, low) for p in patterns):
            out.append(c)
    return out


def summarize_openface(path: Path | str) -> dict[str, float]:
    df = pd.read_csv(path)
    au_cols = _pick_cols(df, [r"^au\d+", r"au_r\d+", r"au_c\d+"])
    emotion_cols = _pick_cols(df, ["happy", "sad", "angry", "surprise", "fear", "neutral"])
    gaze_cols = _pick_cols(df, ["gaze", "pose_tx", "pose_ty", "pose_tz", "pose_rx", "pose_ry", "pose_rz"])
    selected = list(dict.fromkeys(au_cols + emotion_cols + gaze_cols))
    if not selected:
        return {"face_frames": float(len(df)), "face_features": 0.0}
    num = df[selected].apply(pd.to_numeric, errors="coerce")
    out = {"face_frames": float(len(df)), "face_features": float(len(selected))}
    for c in selected:
        out[f"{c}_mean"] = float(num[c].mean(skipna=True))
        out[f"{c}_std"] = float(num[c].std(skipna=True))
    return out


def summarize_face_table(df: pd.DataFrame) -> dict[str, float]:
    return {"face_frames": float(len(df)), "face_columns": float(len(df.columns))}
