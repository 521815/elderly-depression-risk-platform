"""Compact feature storage and incremental manifest for MPDD tables.

These helpers only change how features are stored/loaded (Parquet + float32 +
zstd) and never alter cross-validation or PCA boundaries, so experiment
results stay comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_DIR, MPDD_ROOT
from .mpdd import _modality_from_path, _subject_id_from_path

PARQUET_DIR = DATA_DIR / "mpdd_parquet"
MANIFEST_JSON = DATA_DIR / "mpdd_manifest.json"
MODALITY_PREFIXES = ["face_", "audio_", "gait_", "personality_"]


def _snapshot(root: Path) -> dict:
    total = 0
    count = 0
    latest = 0.0
    for p in root.rglob("*.npy"):
        if "descriptions_embeddings_with_ids" in p.name:
            continue
        try:
            stat = p.stat()
        except OSError:
            continue
        total += stat.st_size
        count += 1
        latest = max(latest, stat.st_mtime)
    return {"count": count, "total_size": total, "latest_mtime": latest}


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def write_manifest(root: Path | str | None = None) -> dict:
    root = Path(root) if root is not None else MPDD_ROOT
    entries = []
    for p in root.rglob("*.npy"):
        if "descriptions_embeddings_with_ids" in p.name:
            continue
        try:
            stat = p.stat()
        except OSError:
            continue
        subject = _subject_id_from_path(p)
        modality = _modality_from_path(p)
        entries.append(
            {
                "path": str(p.relative_to(root)),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "subject": subject,
                "modality": modality[0] if modality else None,
            }
        )
    manifest = {
        "version": 1,
        "entries": entries,
        "snapshot": _snapshot(root),
    }
    MANIFEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_JSON.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def manifest_fresh(root: Path | str | None = None) -> bool:
    if not MANIFEST_JSON.exists():
        return False
    root = Path(root) if root is not None else MPDD_ROOT
    try:
        manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("snapshot") == _snapshot(root)


def save_parquet_features(df: pd.DataFrame, name: str) -> dict:
    """Split a wide feature table into per-modality Parquet files (float32+zstd)."""
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    meta_cols = [
        c
        for c in df.columns
        if not any(c.startswith(prefix) for prefix in MODALITY_PREFIXES)
    ]
    sizes = {}
    for prefix in MODALITY_PREFIXES:
        feat_cols = [c for c in df.columns if c.startswith(prefix)]
        if not feat_cols:
            continue
        part = df[meta_cols + feat_cols].copy()
        part[feat_cols] = part[feat_cols].astype("float32")
        path = PARQUET_DIR / f"{name}_{prefix.rstrip('_')}.parquet"
        part.to_parquet(path, index=False, compression="zstd", engine="pyarrow")
        sizes[prefix.rstrip("_")] = path.stat().st_size
    meta = df[meta_cols].copy()
    meta_path = PARQUET_DIR / f"{name}_meta.parquet"
    meta.to_parquet(meta_path, index=False, compression="zstd", engine="pyarrow")
    sizes["meta"] = meta_path.stat().st_size
    return sizes


def load_parquet_features(name: str) -> pd.DataFrame | None:
    meta_path = PARQUET_DIR / f"{name}_meta.parquet"
    if not meta_path.exists():
        return None
    try:
        meta = pd.read_parquet(meta_path)
        for prefix in MODALITY_PREFIXES:
            path = PARQUET_DIR / f"{name}_{prefix.rstrip('_')}.parquet"
            if not path.exists():
                continue
            part = pd.read_parquet(path)
            feat_cols = [c for c in part.columns if c.startswith(prefix)]
            meta = pd.concat(
                [meta.reset_index(drop=True), part[feat_cols].reset_index(drop=True)],
                axis=1,
            )
        return meta
    except (ImportError, OSError, ValueError):
        # CSV remains the canonical fallback when the optional Parquet engine
        # is unavailable or a cache is stale/corrupt.
        return None


def build_feature_cache(force: bool = False) -> dict:
    from .mpdd import build_feature_table, build_utterance_feature_table

    subject_csv = DATA_DIR / "mpdd_features.csv"
    utterance_csv = DATA_DIR / "mpdd_utterance_features.csv"
    if not subject_csv.exists() or force:
        build_feature_table(output_path=subject_csv)
    if not utterance_csv.exists() or force:
        build_utterance_feature_table(output_path=utterance_csv)

    before = {
        "mpdd_raw": _snapshot(MPDD_ROOT)["total_size"],
        "subject_csv": subject_csv.stat().st_size if subject_csv.exists() else 0,
        "utterance_csv": utterance_csv.stat().st_size if utterance_csv.exists() else 0,
        "parquet": _dir_size(PARQUET_DIR),
    }
    subject_df = pd.read_csv(subject_csv)
    utterance_df = pd.read_csv(utterance_csv)
    subject_sizes = save_parquet_features(subject_df, "mpdd_features")
    utterance_sizes = save_parquet_features(utterance_df, "mpdd_utterance_features")
    write_manifest(MPDD_ROOT)
    after = {
        "mpdd_raw": _snapshot(MPDD_ROOT)["total_size"],
        "subject_csv": subject_csv.stat().st_size,
        "utterance_csv": utterance_csv.stat().st_size,
        "parquet": _dir_size(PARQUET_DIR),
    }
    return {
        "before": before,
        "after": after,
        "subject_modalities": subject_sizes,
        "utterance_modalities": utterance_sizes,
        "manifest_fresh": manifest_fresh(MPDD_ROOT),
    }


def cache_status() -> dict:
    raw_size = None
    if MANIFEST_JSON.exists():
        try:
            raw_size = json.loads(MANIFEST_JSON.read_text(encoding="utf-8")).get(
                "snapshot", {}
            ).get("total_size")
        except (OSError, json.JSONDecodeError):
            pass
    if raw_size is None:
        raw_size = _snapshot(MPDD_ROOT)["total_size"]
    subject_csv = DATA_DIR / "mpdd_features.csv"
    utterance_csv = DATA_DIR / "mpdd_utterance_features.csv"
    return {
        "raw_npy_bytes": raw_size,
        "subject_csv_bytes": subject_csv.stat().st_size if subject_csv.exists() else 0,
        "utterance_csv_bytes": utterance_csv.stat().st_size if utterance_csv.exists() else 0,
        "parquet_bytes": _dir_size(PARQUET_DIR),
        "manifest_fresh": manifest_fresh(MPDD_ROOT),
        "parquet_ready": (PARQUET_DIR / "mpdd_utterance_features_meta.parquet").exists(),
    }
