"""Build a de-identified subject-level table from the CMDC AES archives."""

from __future__ import annotations

import io
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


CMDC_PAPER_DOI = "10.1109/TAFFC.2022.3181210"
CMDC_DATASET_DOI = "10.21227/jng1-rn06"
CMDC_MODALITIES = ("text", "visual", "audio_deep", "acoustic")
CMDC_FEATURE_COUNTS = {
    "text": 768,
    "visual": 768,
    "audio_deep": 128,
    "acoustic": 88,
}

_FEATURE_PATTERNS = {
    "text": re.compile(r"^(HC|MDD)\d+/vector_Q\d+\.csv$"),
    "visual": re.compile(r"^(HC|MDD)\d+/Q\d+\.npy$"),
    "audio_deep": re.compile(r"^(HC|MDD)\d+/Q\d+\.pkl$"),
    "acoustic": re.compile(r"^(HC|MDD)\d+/Q\d+\.wav\.csv$"),
}
_TRANSCRIPT_PATTERN = re.compile(r"^(HC|MDD)\d+/Q\d+\.txt$")
_AUDIO_PATTERN = re.compile(r"^(HC|MDD)\d+/Q\d+\.wav$")


class _SafeNumpyUnpickler(pickle.Unpickler):
    """Allow only the NumPy classes used by CMDC feature pickles."""

    _ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked pickle class: {module}.{name}")


def _load_text_vector(data: bytes) -> np.ndarray:
    text = data.decode("utf-8", errors="replace")
    values = np.fromstring(text.strip(), sep=",", dtype=np.float64)
    return values.reshape(-1)


def _load_numpy_vector(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False).reshape(-1)


def _load_pickle_vector(data: bytes) -> np.ndarray:
    value = _SafeNumpyUnpickler(io.BytesIO(data)).load()
    return np.asarray(value).reshape(-1)


def _load_arff_vector(data: bytes) -> np.ndarray | None:
    text = data.decode("utf-8", errors="replace")
    data_line = next(
        (
            line
            for line in text.splitlines()
            if line.strip() and not line.startswith(("@", "%"))
        ),
        "",
    )
    if not data_line:
        return None
    fields = data_line.split(",")
    if len(fields) < 3:
        return None
    try:
        return np.asarray(
            [float(value) for value in fields[1:-1]],
            dtype=np.float64,
        )
    except ValueError:
        return None


def _aggregate_vectors(vectors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.stack([np.asarray(item, dtype=np.float64) for item in vectors])
    matrix[~np.isfinite(matrix)] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
    return mean, std


def _modality_columns(df: pd.DataFrame, modality: str) -> list[str]:
    return [
        column
        for column in df.columns
        if column.startswith(f"{modality}_")
        and not column.endswith("_questions")
    ]


def build_cmdc_feature_table(
    cmdc_root: Path | str,
    output_path: Path | str,
    audit_path: Path | str | None = None,
    password: str | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Read encrypted CMDC question-level features and aggregate by subject.

    Only numerical features and de-identified subject IDs are retained. Raw
    transcripts, audio, video, archive passwords and questionnaire source files
    are never copied into the generated table or audit.
    """

    try:
        import libarchive
    except ImportError as exc:  # pragma: no cover - environment error
        raise RuntimeError(
            "缺少 libarchive-c，无法读取 CMDC 加密压缩包"
        ) from exc

    root = Path(cmdc_root)
    archives = sorted(root.glob("part*.zip"))
    if not archives:
        raise FileNotFoundError(f"CMDC 压缩包目录中未找到 part*.zip：{root}")

    archive_password = password or os.getenv("CMDC_ARCHIVE_PASSWORD")
    subject_vectors: defaultdict[
        str, defaultdict[str, list[np.ndarray]]
    ] = defaultdict(lambda: defaultdict(list))
    question_counts: defaultdict[str, Counter] = defaultdict(Counter)
    labels: dict[str, int] = {}
    file_counts: Counter = Counter()
    malformed_counts: Counter = Counter()
    archive_members = 0
    audio_bytes = 0

    loaders = {
        "text": _load_text_vector,
        "visual": _load_numpy_vector,
        "audio_deep": _load_pickle_vector,
        "acoustic": _load_arff_vector,
    }

    for archive_index, archive in enumerate(archives):
        try:
            with libarchive.file_reader(
                str(archive),
                passphrase=archive_password,
            ) as entries:
                for entry in entries:
                    if not entry.isfile:
                        continue
                    archive_members += 1
                    pathname = str(entry.pathname)
                    subject_id = pathname.split("/", 1)[0]
                    if subject_id.startswith("MDD"):
                        labels[subject_id] = 1
                    elif subject_id.startswith("HC"):
                        labels[subject_id] = 0

                    if _TRANSCRIPT_PATTERN.match(pathname):
                        file_counts["transcripts"] += 1
                        continue
                    if _AUDIO_PATTERN.match(pathname):
                        file_counts["audio_files"] += 1
                        audio_bytes += int(getattr(entry, "size", 0) or 0)
                        continue

                    modality = next(
                        (
                            name
                            for name, pattern in _FEATURE_PATTERNS.items()
                            if pattern.match(pathname)
                        ),
                        None,
                    )
                    if modality is None:
                        continue
                    file_counts[f"{modality}_files"] += 1
                    data = b"".join(entry.get_blocks())
                    try:
                        vector = loaders[modality](data)
                    except Exception:
                        malformed_counts[modality] += 1
                        continue
                    expected = CMDC_FEATURE_COUNTS[modality]
                    if vector is None or vector.shape != (expected,):
                        malformed_counts[modality] += 1
                        continue
                    subject_vectors[subject_id][modality].append(vector)
                    question_counts[subject_id][modality] += 1
        except Exception as exc:
            raise ValueError(
                f"无法读取 CMDC 压缩包 {archive.name}；请检查归档密码和解压环境"
            ) from exc
        if progress:
            progress(
                (archive_index + 1) / len(archives),
                f"已读取 {archive_index + 1}/{len(archives)} 个压缩包",
            )

    rows = []
    for subject_id in sorted(subject_vectors):
        vectors_by_modality = subject_vectors[subject_id]
        row = {
            "source": "CMDC",
            "subject_id": subject_id,
            "label": labels.get(subject_id),
            "group": "MDD" if labels.get(subject_id) == 1 else "HC",
            "age_scope": "mixed_adults_not_older_specific",
        }
        for modality in CMDC_MODALITIES:
            row[f"{modality}_questions"] = int(
                question_counts[subject_id][modality]
            )
            vectors = vectors_by_modality.get(modality, [])
            if not vectors:
                continue
            mean, std = _aggregate_vectors(vectors)
            for index, value in enumerate(mean):
                row[f"{modality}_mean_{index:03d}"] = float(value)
            for index, value in enumerate(std):
                row[f"{modality}_std_{index:03d}"] = float(value)
        row["complete_multimodal"] = int(
            all(
                question_counts[subject_id][modality] > 0
                for modality in CMDC_MODALITIES
            )
        )
        rows.append(row)

    if not rows:
        raise ValueError("CMDC 压缩包中未找到可用的数值特征")

    df = pd.DataFrame.from_records(rows)
    metadata_columns = [
        "source",
        "subject_id",
        "label",
        "group",
        "age_scope",
        *[f"{modality}_questions" for modality in CMDC_MODALITIES],
        "complete_multimodal",
    ]
    feature_columns = [column for column in df.columns if column not in metadata_columns]
    df = df[metadata_columns + feature_columns]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    output.chmod(0o600)

    modality_audit = {}
    for modality in CMDC_MODALITIES:
        covered = int((df[f"{modality}_questions"] > 0).sum())
        modality_audit[modality] = {
            "question_files_read": int(file_counts[f"{modality}_files"]),
            "subjects_covered": covered,
            "feature_columns": len(_modality_columns(df, modality)),
            "malformed_or_unreadable": int(malformed_counts[modality]),
        }
    label_counts = df["label"].value_counts().to_dict()
    audit = {
        "ready": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "CMDC",
        "paper_doi": CMDC_PAPER_DOI,
        "dataset_doi": CMDC_DATASET_DOI,
        "archive_count": len(archives),
        "archive_members": archive_members,
        "subjects": int(len(df)),
        "hc_subjects": int(label_counts.get(0, 0)),
        "mdd_subjects": int(label_counts.get(1, 0)),
        "questions_per_subject": 12,
        "audio_hours": round(audio_bytes / 192000 / 3600, 2),
        "transcripts_observed": int(file_counts["transcripts"]),
        "modalities": modality_audit,
        "complete_multimodal_subjects": int(df["complete_multimodal"].sum()),
        "derived_feature_columns": len(feature_columns),
        "raw_data_copied": False,
        "password_persisted": False,
        "age_scope": "混合成人样本，非老年专属",
        "usage_boundary": (
            "仅用于非商业学术研究、外部验证和多模态预训练；"
            "不得替代 CHARLS 老年主队列或临床诊断。"
        ),
        "license_boundary": (
            "遵守 CMDC EULA；不得商用、再分发、公开原始文件或转交归档密码。"
        ),
    }
    if audit_path is not None:
        audit_output = Path(audit_path)
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        audit_output.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        audit_output.chmod(0o600)
    return df, audit
