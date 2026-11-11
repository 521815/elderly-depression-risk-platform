"""Load and standardize MPDD extracted feature files."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SEQUENCE_STATS = ["mean", "std", "min", "max"]
MPDD_BINARY_COLUMN = "mpdd_binary"

MODALITY_NAMES = {
    "mfcc": ("audio", "mfcc"),
    "opensmile": ("audio", "opensmile"),
    "wav2vec2": ("audio", "wav2vec2"),
    "wav2vec": ("audio", "wav2vec2"),
    "densenet": ("face", "densenet"),
    "resnet": ("face", "resnet"),
    "openface": ("face", "openface"),
    "imu": ("gait", "imu"),
}


def _find(root: Path, keywords: list[str], required: list[str] | None = None) -> list[Path]:
    files = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".csv", ".parquet", ".npz"}:
            continue
        low = p.name.lower()
        if all(k in low for k in keywords) and (required is None or any(r in low for r in required)):
            files.append(p)
    return sorted(files)


def _infer_subject_col(df: pd.DataFrame) -> str:
    for c in ["subject_id", "subjectid", "subject", "id", "participant", "sample_id", "ID"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "id" in c.lower() or "subject" in c.lower():
            return c
    raise ValueError(f"Cannot find subject id column in {list(df.columns)[:20]}")


def _infer_label_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        low = c.lower()
        if any(k in low for k in ["hamd", "label", "depress", "severity", "score"]):
            return c
    return None


def add_mpdd_binary_target(
    df: pd.DataFrame,
    threshold: int = 8,
) -> pd.DataFrame:
    """Add a canonical MPDD binary target with official labels taking precedence."""
    out = df.copy()
    target = pd.Series(pd.NA, index=out.index, dtype="Float64")
    source = pd.Series("missing", index=out.index, dtype="object")

    if "label2" in out.columns:
        official = pd.to_numeric(out["label2"], errors="coerce")
        official = official.where(official.isin([0, 1]))
        target = official.astype("Float64")
        source.loc[official.notna()] = "label2"

    score_col = next(
        (
            column
            for column in ("phq9_score", "hamd_score")
            if column in out.columns
        ),
        None,
    )
    if score_col is not None:
        score = pd.to_numeric(out[score_col], errors="coerce")
        fallback = (score >= float(threshold)).astype("Float64")
        fallback = fallback.where(score.notna())
        use_fallback = target.isna() & fallback.notna()
        target.loc[use_fallback] = fallback.loc[use_fallback]
        source.loc[use_fallback] = f"{score_col}>={threshold}"

    out[MPDD_BINARY_COLUMN] = target.astype("Int64")
    out["mpdd_label_source"] = source
    return out


def mpdd_label_audit(
    df: pd.DataFrame,
    threshold: int = 8,
) -> dict:
    """Summarize official-label coverage and PHQ-9 threshold disagreement."""
    if df.empty:
        return {
            "labeled_subjects": 0,
            "unlabeled_subjects": 0,
            "label2_positive": 0,
            "label2_negative": 0,
            "threshold_positive": 0,
            "threshold_disagreement": 0,
            "phq9_threshold": int(threshold),
            "label_semantics": "official label2 when available; PHQ-9 threshold only as fallback",
        }

    subject_col = "subject_id" if "subject_id" in df.columns else None
    subjects = df.drop_duplicates(subject_col) if subject_col else df
    official = (
        pd.to_numeric(subjects["label2"], errors="coerce")
        if "label2" in subjects.columns
        else pd.Series(np.nan, index=subjects.index, dtype=float)
    )
    official = official.where(official.isin([0, 1]))
    score_col = next(
        (
            column
            for column in ("phq9_score", "hamd_score")
            if column in subjects.columns
        ),
        None,
    )
    threshold_label = (
        pd.to_numeric(subjects[score_col], errors="coerce") >= float(threshold)
        if score_col is not None
        else pd.Series(np.nan, index=subjects.index, dtype=float)
    )
    if score_col is None:
        threshold_label[:] = np.nan
    threshold_label = threshold_label.where(
        pd.to_numeric(subjects[score_col], errors="coerce").notna()
        if score_col is not None
        else pd.Series(False, index=subjects.index)
    )

    comparable = official.notna() & threshold_label.notna()
    return {
        "labeled_subjects": int(official.notna().sum()),
        "unlabeled_subjects": int(official.isna().sum()),
        "label2_positive": int((official == 1).sum()),
        "label2_negative": int((official == 0).sum()),
        "threshold_positive": int(threshold_label.eq(True).sum()),
        "threshold_disagreement": int(
            (
                comparable
                & (official.astype(float) != threshold_label.astype(float))
            ).sum()
        ),
        "phq9_threshold": int(threshold),
        "label_semantics": "official label2 when available; PHQ-9 threshold only as fallback",
    }


def _load_csv_or_parquet(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".npz":
        raise ValueError(f"npz features not supported yet: {path}")
    return pd.read_csv(path)


def discover_files(mpdd_root: Path | str) -> dict[str, list[Path]]:
    root = Path(mpdd_root)
    return {
        "labels": _find(root, ["label"], required=["train", "test"]) + _find(root, ["label"]),
        "face": _find(root, ["face"]) + _find(root, ["openface"]) + _find(root, ["resnet", "visual"]),
        "audio": _find(root, ["audio"]) + _find(root, ["opensmile", "wav2vec", "mel"]),
        "gait": _find(root, ["gait"]) + _find(root, ["tsfel"]),
        "personality": _find(root, ["bigfive", "personality"]),
    }


def _official_elder_dirs(mpdd_root: Path) -> list[Path]:
    """Return Elder dirs from the official MPDD extracted-feature layout."""
    dirs = []
    for elder in mpdd_root.rglob("Elder"):
        if not elder.is_dir():
            continue
        if any((elder / name).is_dir() for name in ("Audio", "Video", "IMU")):
            dirs.append(elder)
    return sorted(dirs)


def _subject_id_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        if part.isdigit():
            return str(int(part))
    return None


def _modality_from_path(path: Path) -> tuple[str, str] | None:
    parts = [p.lower() for p in path.parts]
    for key, (prefix, modality) in MODALITY_NAMES.items():
        if key in parts:
            return prefix, modality
    return None


def _load_npy_sequence(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def _aggregate_sequence_files(paths: list[Path]) -> pd.Series:
    chunks = [_load_npy_sequence(p) for p in paths]
    seq = np.concatenate(chunks, axis=0).astype(np.float64)
    values = []
    for i in range(seq.shape[1]):
        col = seq[:, i]
        values.extend([col.mean(), col.std(), float(col.min()), float(col.max())])
    index = [f"{stat}_{i}" for i in range(seq.shape[1]) for stat in SEQUENCE_STATS]
    return pd.Series(values, index=index)


def _aggregate_single_file(path: Path) -> pd.Series:
    seq = _load_npy_sequence(path).astype(np.float64)
    values = []
    for i in range(seq.shape[1]):
        col = seq[:, i]
        values.extend([col.mean(), col.std(), float(col.min()), float(col.max())])
    index = [f"{stat}_{i}" for i in range(seq.shape[1]) for stat in SEQUENCE_STATS]
    return pd.Series(values, index=index)


def _load_personality_embeddings(elder_dir: Path) -> dict[str, np.ndarray]:
    embedding_path = elder_dir / "descriptions_embeddings_with_ids.npy"
    if not embedding_path.exists():
        matches = list(elder_dir.rglob("descriptions_embeddings_with_ids.npy"))
        embedding_path = matches[0] if matches else None
    if embedding_path is None:
        return {}
    data = np.load(embedding_path, allow_pickle=True)
    return {
        str(entry["id"]): np.asarray(entry["embedding"]).reshape(-1)
        for entry in data
    }


def _official_features(elder_dir: Path) -> pd.DataFrame:
    grouped: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in elder_dir.rglob("*.npy"):
        if "descriptions_embeddings_with_ids" in path.name:
            continue
        subject = _subject_id_from_path(path)
        modality = _modality_from_path(path)
        if subject is None or modality is None:
            continue
        prefix, name = modality
        grouped[subject][f"{prefix}_{name}"].append(path)

    embeddings = _load_personality_embeddings(elder_dir)

    rows = []
    for subject in sorted(grouped, key=lambda x: int(x)):
        row = {"subject_id": subject}
        for feature_name, paths in sorted(grouped[subject].items()):
            series = _aggregate_sequence_files(paths)
            series.index = [f"{feature_name}_{i}" for i in series.index]
            row.update(series.to_dict())
        if subject in embeddings:
            series = pd.Series(
                embeddings[subject],
                index=[f"personality_{i}" for i in range(len(embeddings[subject]))],
            )
            row.update(series.to_dict())
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _official_utterance_features(elder_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(elder_dir.rglob("*.npy")):
        if "descriptions_embeddings_with_ids" in path.name:
            continue
        subject = _subject_id_from_path(path)
        modality = _modality_from_path(path)
        if subject is None or modality is None:
            continue
        prefix, name = modality
        feature_name = f"{prefix}_{name}"
        row = {"subject_id": subject, "segment": path.stem, "modality": feature_name}
        series = _aggregate_single_file(path)
        series.index = [f"{feature_name}_{i}" for i in series.index]
        row.update(series.to_dict())
        rows.append(row)

    embeddings = _load_personality_embeddings(elder_dir)
    for row in rows:
        subject = row["subject_id"]
        if subject in embeddings:
            series = pd.Series(
                embeddings[subject],
                index=[f"personality_{i}" for i in range(len(embeddings[subject]))],
            )
            row.update(series.to_dict())
    return pd.DataFrame.from_records(rows)


def _official_labels(elder_dir: Path) -> pd.DataFrame | None:
    for path in sorted(elder_dir.rglob("*.csv")):
        if "description" in path.name.lower():
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            continue
        if any(
            c in df.columns
            for c in ["label2", "label3", "PHQ-9", "PHQ", "HAMD", "hamd", "label", "depress", "score"]
        ):
            return df
    return None


def _official_label_table(df: pd.DataFrame) -> pd.DataFrame:
    subject_col = _infer_subject_col(df)
    out = {"subject_id": df[subject_col].astype(str)}
    if "split" in df.columns:
        out["fold"] = df["split"]
    if "PHQ-9" in df.columns:
        out["phq9_score"] = pd.to_numeric(df["PHQ-9"], errors="coerce")
    hamd_col = next((c for c in df.columns if "hamd" in c.lower()), None)
    if hamd_col is not None:
        out["hamd_score"] = pd.to_numeric(df[hamd_col], errors="coerce")
    elif "phq9_score" in out:
        out["hamd_score"] = out["phq9_score"]
    if "label2" in df.columns:
        out["label2"] = pd.to_numeric(df["label2"], errors="coerce")
    if "label3" in df.columns:
        out["label3"] = pd.to_numeric(df["label3"], errors="coerce")
    return pd.DataFrame(out)


def _build_official_table(mpdd_root: Path) -> pd.DataFrame | None:
    elder_dirs = _official_elder_dirs(mpdd_root)
    if not elder_dirs:
        return None

    frames = []
    for elder_dir in elder_dirs:
        features = _official_features(elder_dir)
        if features.empty:
            continue
        labels = _official_labels(elder_dir)
        if labels is not None:
            table = features.merge(_official_label_table(labels), on="subject_id", how="left")
        else:
            table = features.copy()
            table["fold"] = "test" if "test" in str(elder_dir).lower() else "train"
        if "fold" not in table.columns:
            table["fold"] = "train" if "train" in str(elder_dir).lower() else "test"
        for col in ["phq9_score", "hamd_score", "label2", "label3"]:
            if col not in table.columns:
                table[col] = pd.NA
        table = add_mpdd_binary_target(table)
        frames.append(table)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_feature_table(
    mpdd_root: Path | str | None = None,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    if mpdd_root is None:
        from ..config import MPDD_ROOT

        mpdd_root = MPDD_ROOT
    root = Path(mpdd_root)

    official = _build_official_table(root)
    if official is not None:
        if output_path is None:
            from ..config import DATA_DIR

            output_path = DATA_DIR / "mpdd_features.csv"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        official.to_csv(output_path, index=False, encoding="utf-8-sig")
        return official

    files = discover_files(root)
    if not files["labels"]:
        raise FileNotFoundError(
            f"No MPDD label file found under {root}. Apply for data first, see data/README.md"
        )

    label_df = _load_csv_or_parquet(files["labels"][0])
    subj_col = _infer_subject_col(label_df)
    label_col = _infer_label_col(label_df)
    if label_col is None:
        raise ValueError("MPDD label file should contain a HAMD/label/depression column")
    table = label_df[[subj_col, label_col]].rename(
        columns={subj_col: "subject_id", label_col: "hamd_score"}
    )
    table["fold"] = "unknown"
    for p in files["labels"][1:]:
        df = _load_csv_or_parquet(p)
        c = _infer_subject_col(df)
        lab = _infer_label_col(df)
        if lab is None:
            continue
        part = df[[c, lab]].rename(columns={c: "subject_id", lab: "hamd_score"})
        table = pd.concat([table, part], ignore_index=True).drop_duplicates(subset=["subject_id"])

    for modality, paths in files.items():
        if modality == "labels":
            continue
        for p in paths:
            df = _load_csv_or_parquet(p)
            c = _infer_subject_col(df)
            prefix = modality
            part = df.rename(columns={col: f"{prefix}_{col}" for col in df.columns if col != c})
            part = part.rename(columns={c: "subject_id"})
            table = table.merge(part, on="subject_id", how="left")

    if output_path is None:
        from ..config import DATA_DIR

        output_path = DATA_DIR / "mpdd_features.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False, encoding="utf-8-sig")
    return table


def build_utterance_feature_table(
    mpdd_root: Path | str | None = None,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Build one row per MPDD utterance, keeping subject labels and personality."""
    if mpdd_root is None:
        from ..config import MPDD_ROOT

        mpdd_root = MPDD_ROOT
    root = Path(mpdd_root)
    elder_dirs = _official_elder_dirs(root)
    if not elder_dirs:
        raise FileNotFoundError(
            f"No official MPDD Elder feature directory found under {root}."
        )

    frames = []
    for elder_dir in elder_dirs:
        features = _official_utterance_features(elder_dir)
        if features.empty:
            continue
        labels = _official_labels(elder_dir)
        if labels is not None:
            table = features.merge(_official_label_table(labels), on="subject_id", how="left")
        else:
            table = features.copy()
            table["fold"] = "test" if "test" in str(elder_dir).lower() else "train"
        if "fold" not in table.columns:
            table["fold"] = "train" if "train" in str(elder_dir).lower() else "test"
        for col in ["phq9_score", "hamd_score", "label2", "label3"]:
            if col not in table.columns:
                table[col] = pd.NA
        table = add_mpdd_binary_target(table)
        frames.append(table)

    table = pd.concat(frames, ignore_index=True)
    if output_path is None:
        from ..config import DATA_DIR

        output_path = DATA_DIR / "mpdd_utterance_features.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False, encoding="utf-8-sig")
    return table
