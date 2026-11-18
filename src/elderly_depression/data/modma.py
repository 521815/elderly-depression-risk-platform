"""Audit the authorized MODMA auxiliary audio cohort."""

from __future__ import annotations

import json
import shutil
import tempfile
import wave
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd


MODMA_SOURCE = "MODMA"
MODMA_AGE_SCOPE = "adults_18_55_not_older_specific"
MODMA_AUDIO_FILES_PER_SUBJECT = 29
MODMA_AUDIO_SUBJECTS = 52
MODMA_VERIFIED_ARCHIVE = "audio_lanzhou_2015_verified.zip"
MODMA_METADATA_COLUMNS = {
    "subject id": "subject_id",
    "type": "group",
    "age": "age",
    "gender": "gender",
    "education（years）": "education_years",
    "PHQ-9": "phq9_score",
    "CTQ-SF": "ctq_sf_score",
    "LES": "les_score",
    "SSRS": "ssrs_score",
    "GAD-7": "gad7_score",
    "PSQI": "psqi_score",
}


def load_modma_metadata(path: Path | str) -> pd.DataFrame:
    """Load and normalize the official MODMA audio metadata workbook."""
    metadata = pd.read_excel(path, sheet_name="Sheet1")
    missing = [column for column in MODMA_METADATA_COLUMNS if column not in metadata]
    if missing:
        raise ValueError(f"MODMA metadata missing columns: {missing}")
    data = metadata[list(MODMA_METADATA_COLUMNS)].rename(
        columns=MODMA_METADATA_COLUMNS
    )
    data["subject_id"] = data["subject_id"].astype(str).str.strip()
    data["group"] = data["group"].astype(str).str.strip().str.upper()
    data["gender"] = data["gender"].astype(str).str.strip().str.upper()
    if not data["group"].isin({"MDD", "HC"}).all():
        raise ValueError("MODMA group must contain only MDD and HC")
    data["label"] = data["group"].map({"HC": 0, "MDD": 1}).astype(int)
    data = data.drop_duplicates(subset=["subject_id"]).reset_index(drop=True)
    return data


def _wav_properties(path: Path) -> dict:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = int(handle.getnchannels())
            sample_rate = int(handle.getframerate())
            frames = int(handle.getnframes())
    except (OSError, EOFError, wave.Error) as exc:
        try:
            header_hex = path.read_bytes()[:16].hex()
        except OSError:
            header_hex = ""
        return {
            "readable": False,
            "error": str(exc),
            "header_hex": header_hex,
            "duration_seconds": np.nan,
            "sample_rate": np.nan,
            "channels": np.nan,
        }
    duration = frames / sample_rate if sample_rate else np.nan
    return {
        "readable": True,
        "error": "",
        "header_hex": "",
        "duration_seconds": float(duration),
        "sample_rate": sample_rate,
        "channels": channels,
    }


def _subject_directories(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit()
    )


@contextmanager
def materialize_modma_audio_root(root: Path | str):
    """Yield an extracted audio root, cleaning temporary extraction on exit."""
    root = Path(root)
    directories = _subject_directories(root) if root.exists() else []
    if directories:
        yield root, False
        return
    archive = root / MODMA_VERIFIED_ARCHIVE
    if not archive.exists():
        raise FileNotFoundError(
            "MODMA subject directories and verified archive are both missing"
        )
    temporary = Path(tempfile.mkdtemp(prefix="modma_audio_"))
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = (temporary / member.filename).resolve()
                if temporary.resolve() not in target.parents and target != temporary.resolve():
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            bundle.extractall(temporary)
        yield temporary, True
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def canonical_subject_id(raw_subject_id: str) -> str:
    value = str(raw_subject_id).strip()
    return str(int(value)) if value.isdigit() else value


def _segment_number(path: Path) -> int | None:
    return int(path.stem) if path.stem.isdigit() else None


def build_modma_subject_index(
    root: Path | str,
    metadata_path: Path | str,
    *,
    inspect_wav_headers: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Match MODMA metadata to subject directories and audit audio files."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"MODMA root does not exist: {root}")
    metadata = load_modma_metadata(metadata_path)
    metadata_ids = set(metadata["subject_id"])
    directories = _subject_directories(root)
    directory_ids = {canonical_subject_id(path.name) for path in directories}
    metadata_by_id = metadata.set_index("subject_id")

    rows = []
    warnings = []
    total_wavs = 0
    complete_subjects = 0
    unreadable_wavs = 0
    unreadable_files = []
    sample_rates: Counter = Counter()
    durations = []

    for directory in directories:
        source_subject_id = directory.name
        subject_id = canonical_subject_id(source_subject_id)
        wav_paths = sorted(
            directory.glob("*.wav"),
            key=lambda path: (
                _segment_number(path) or 10**9,
                path.name,
            ),
        )
        expected_segments = set(
            range(1, MODMA_AUDIO_FILES_PER_SUBJECT + 1)
        )
        actual_segments = {
            number
            for path in wav_paths
            if (number := _segment_number(path)) is not None
        }
        missing_segments = sorted(expected_segments - actual_segments)
        extra_segments = sorted(actual_segments - expected_segments)
        non_numeric_files = sorted(
            path.name for path in wav_paths if _segment_number(path) is None
        )
        properties = []
        if inspect_wav_headers:
            properties = [_wav_properties(path) for path in wav_paths]
        readable = sum(bool(item["readable"]) for item in properties)
        if inspect_wav_headers:
            unreadable_wavs += len(properties) - readable
        valid_durations = [
            float(item["duration_seconds"])
            for item in properties
            if item["readable"] and np.isfinite(item["duration_seconds"])
        ]
        durations.extend(valid_durations)
        for item in properties:
            if item["readable"]:
                sample_rates[int(item["sample_rate"])] += 1
        metadata_row = (
            metadata_by_id.loc[subject_id].to_dict()
            if subject_id in metadata_by_id.index
            else {}
        )
        is_complete = (
            len(wav_paths) == MODMA_AUDIO_FILES_PER_SUBJECT
            and not missing_segments
            and not extra_segments
            and not non_numeric_files
            and (not inspect_wav_headers or readable == len(wav_paths))
        )
        complete_subjects += int(is_complete)
        total_wavs += len(wav_paths)
        if missing_segments:
            warnings.append(
                f"subject {subject_id} missing WAV segments: {missing_segments}"
            )
        if extra_segments:
            warnings.append(
                f"subject {subject_id} unexpected WAV files: {extra_segments}"
            )
        if non_numeric_files:
            warnings.append(
                f"subject {subject_id} non-numeric WAV filenames: {non_numeric_files}"
            )
        if inspect_wav_headers and readable != len(wav_paths):
            failed_names = [
                path.name
                for path, item in zip(wav_paths, properties)
                if not item["readable"]
            ]
            for path, item in zip(wav_paths, properties):
                if item["readable"]:
                    continue
                unreadable_files.append(
                    {
                        "subject_id": subject_id,
                        "filename": path.name,
                        "size_bytes": int(path.stat().st_size),
                        "header_hex": item["header_hex"],
                        "error": item["error"],
                    }
                )
            warnings.append(
                f"subject {subject_id} unreadable WAV files: "
                f"{failed_names}"
            )
        rows.append(
            {
                "source": MODMA_SOURCE,
                "subject_id": subject_id,
                "source_subject_id": source_subject_id,
                "subject_path": str(directory),
                "group": metadata_row.get("group"),
                "label": metadata_row.get("label"),
                "age": metadata_row.get("age"),
                "gender": metadata_row.get("gender"),
                "education_years": metadata_row.get("education_years"),
                "phq9_score": metadata_row.get("phq9_score"),
                "ctq_sf_score": metadata_row.get("ctq_sf_score"),
                "les_score": metadata_row.get("les_score"),
                "ssrs_score": metadata_row.get("ssrs_score"),
                "gad7_score": metadata_row.get("gad7_score"),
                "psqi_score": metadata_row.get("psqi_score"),
                "wav_count": len(wav_paths),
                "readable_wav_count": readable if inspect_wav_headers else np.nan,
                "missing_segments": ";".join(map(str, missing_segments)),
                "extra_segments": ";".join(map(str, extra_segments)),
                "total_duration_seconds": float(np.sum(valid_durations))
                if valid_durations
                else np.nan,
                "mean_duration_seconds": float(np.mean(valid_durations))
                if valid_durations
                else np.nan,
                "complete_29_segments": int(is_complete),
                "age_scope": MODMA_AGE_SCOPE,
            }
        )

    index = pd.DataFrame(rows)
    age_values = pd.to_numeric(metadata["age"], errors="coerce").dropna()
    group_counts = metadata["group"].value_counts().to_dict()
    metadata_ids_missing_audio = sorted(metadata_ids - directory_ids)
    audio_ids_missing_metadata = sorted(directory_ids - metadata_ids)
    if metadata_ids_missing_audio:
        warnings.append(
            f"metadata subjects without audio directory: {metadata_ids_missing_audio}"
        )
    if audio_ids_missing_metadata:
        warnings.append(
            f"audio directories without metadata: {audio_ids_missing_metadata}"
        )
    expected_total = len(metadata) * MODMA_AUDIO_FILES_PER_SUBJECT
    audit = {
        "source": MODMA_SOURCE,
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "root": str(root),
        "metadata_path": str(metadata_path),
        "authorization": {
            "status": "mentor_confirmed_authorized",
            "note": (
                "导师已完成官方申请并确认可用于本项目；"
                "EULA原件和签署日期由负责人另行归档。"
            ),
        },
        "scope": {
            "population": "clinical MDD and healthy controls",
            "age_scope": MODMA_AGE_SCOPE,
            "primary_use": "auxiliary pretraining and cross-domain robustness",
            "is_older_adult_primary_dataset": False,
            "has_face_video": False,
            "has_gait": False,
            "has_eeg_package": True,
        },
        "metadata": {
            "subjects": int(len(metadata)),
            "group_counts": group_counts,
            "gender_counts": metadata["gender"].value_counts().to_dict(),
            "age_min": float(age_values.min()),
            "age_max": float(age_values.max()),
            "age_mean": float(age_values.mean()),
            "age_median": float(age_values.median()),
        },
        "audio": {
            "subject_directories": len(directories),
            "expected_wav_files": int(expected_total),
            "actual_wav_files": int(total_wavs),
            "complete_subjects": int(complete_subjects),
            "unreadable_wav_files": int(unreadable_wavs),
            "unreadable_files": unreadable_files,
            "sample_rate_counts": {
                str(key): int(value) for key, value in sample_rates.items()
            },
            "duration_seconds_total": float(np.sum(durations))
            if durations
            else np.nan,
            "duration_seconds_mean": float(np.mean(durations))
            if durations
            else np.nan,
            "complete_dataset": bool(
                len(directories) == MODMA_AUDIO_SUBJECTS
                and total_wavs == expected_total
                and complete_subjects == MODMA_AUDIO_SUBJECTS
                and not unreadable_wavs
                and not metadata_ids_missing_audio
                and not audio_ids_missing_metadata
            ),
        },
        "pairing": {
            "metadata_subjects_missing_audio": metadata_ids_missing_audio,
            "audio_subjects_missing_metadata": audio_ids_missing_metadata,
            "allowed_with_mpdd_or_charls": False,
            "reason": "不同研究、不同年龄段与不同标签定义，不能构造个体级配对。",
        },
        "warnings": warnings,
    }
    return index, audit


def write_modma_audit(audit: dict, output_path: Path | str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output
