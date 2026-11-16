"""Acoustic feature extraction and source-internal baselines for MODMA audio."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_STATE
from ..evaluation import imbalanced_metrics_frame
from ..models.fusion import expected_calibration_error


MODMA_TARGET_SAMPLE_RATE = 16000
MODMA_FRAME_LENGTH = 400
MODMA_HOP_LENGTH = 160
MODMA_N_MELS = 26
MODMA_N_MFCC = 13


def modma_task_type(segment: int) -> str:
    if 1 <= segment <= 18:
        return "interview"
    if segment == 19:
        return "passage_reading"
    if 20 <= segment <= 25:
        return "word_reading"
    if 26 <= segment <= 29:
        return "picture_description"
    return "unknown"


def _read_mono(path: Path) -> tuple[int, np.ndarray, int]:
    sample_rate, data = wavfile.read(path)
    original_channels = int(data.shape[1]) if data.ndim > 1 else 1
    values = np.asarray(data)
    if values.ndim > 1:
        values = values.mean(axis=1)
    if np.issubdtype(values.dtype, np.integer):
        peak_abs = float(np.max(np.abs(values.astype(np.float64))))
        scale = peak_abs if peak_abs > 0 else 1.0
        values = values.astype(np.float64) / scale
    else:
        values = values.astype(np.float64)
        peak = float(np.max(np.abs(values))) if values.size else 0.0
        if peak > 0:
            values = values / peak
    values = np.nan_to_num(values, copy=False)
    if sample_rate > MODMA_TARGET_SAMPLE_RATE:
        values = resample_poly(
            values,
            MODMA_TARGET_SAMPLE_RATE,
            sample_rate,
        )
        sample_rate = MODMA_TARGET_SAMPLE_RATE
    return sample_rate, values, original_channels


def _frame_signal(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if signal.size < frame_length:
        signal = np.pad(signal, (0, frame_length - signal.size))
    frames = np.lib.stride_tricks.sliding_window_view(
        signal,
        frame_length,
    )[::hop_length]
    return frames.astype(np.float64, copy=False)


def _hz_to_mel(value: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def _mel_to_hz(value: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(value) / 2595.0) - 1.0)


def _mel_filterbank(
    sample_rate: int,
    fft_size: int,
    n_mels: int = MODMA_N_MELS,
) -> np.ndarray:
    mel_points = np.linspace(
        _hz_to_mel(20.0),
        _hz_to_mel(sample_rate / 2.0),
        n_mels + 2,
    )
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((n_mels, fft_size // 2 + 1), dtype=np.float64)
    for index in range(1, n_mels + 1):
        left, center, right = bins[index - 1 : index + 2]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        left_slice = np.arange(left, min(center, filters.shape[1]))
        right_slice = np.arange(center, min(right, filters.shape[1]))
        if left_slice.size:
            filters[index - 1, left_slice] = (
                left_slice - left
            ) / max(center - left, 1)
        if right_slice.size:
            filters[index - 1, right_slice] = (
                right - right_slice
            ) / max(right - center, 1)
    return filters


def extract_modma_wav_features(path: Path | str) -> dict:
    """Extract compact task-level acoustic features from one MODMA WAV."""
    path = Path(path)
    sample_rate, signal, original_channels = _read_mono(path)
    frames = _frame_signal(
        signal,
        MODMA_FRAME_LENGTH,
        MODMA_HOP_LENGTH,
    )
    window = np.hamming(MODMA_FRAME_LENGTH)
    emphasized = np.concatenate(
        (
            frames[:, :1],
            frames[:, 1:] - 0.97 * frames[:, :-1],
        ),
        axis=1,
    )
    spectrum = np.abs(np.fft.rfft(emphasized * window, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(MODMA_FRAME_LENGTH, 1.0 / sample_rate)
    power_sum = np.maximum(spectrum.sum(axis=1), 1e-12)
    centroid = (spectrum * frequencies).sum(axis=1) / power_sum
    bandwidth = np.sqrt(
        (spectrum * (frequencies[None, :] - centroid[:, None]) ** 2).sum(axis=1)
        / power_sum
    )
    cumulative = np.cumsum(spectrum, axis=1)
    rolloff_bins = np.argmax(cumulative >= 0.85 * cumulative[:, -1:], axis=1)
    rolloff = frequencies[rolloff_bins]
    flatness = np.exp(np.mean(np.log(spectrum + 1e-12), axis=1)) / np.mean(
        spectrum + 1e-12,
        axis=1,
    )
    rms = np.sqrt(np.mean(emphasized**2, axis=1))
    zcr = np.mean(np.abs(np.diff(np.signbit(emphasized), axis=1)), axis=1)
    mel_energy = spectrum @ _mel_filterbank(sample_rate, MODMA_FRAME_LENGTH).T
    mfcc = dct(np.log(mel_energy + 1e-10), type=2, axis=1, norm="ortho")[
        :, :MODMA_N_MFCC
    ]
    silence_threshold = max(1e-4, float(np.percentile(rms, 95)) * 0.05)

    features = {
        "duration_seconds": float(len(signal) / sample_rate),
        "sample_rate": int(sample_rate),
        "original_channels": int(original_channels),
        "frame_count": int(frames.shape[0]),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
        "bandwidth_mean": float(np.mean(bandwidth)),
        "bandwidth_std": float(np.std(bandwidth)),
        "rolloff_mean": float(np.mean(rolloff)),
        "rolloff_std": float(np.std(rolloff)),
        "flatness_mean": float(np.mean(flatness)),
        "flatness_std": float(np.std(flatness)),
        "silence_ratio": float(np.mean(rms < silence_threshold)),
    }
    for index in range(MODMA_N_MFCC):
        features[f"mfcc_{index:02d}_mean"] = float(np.mean(mfcc[:, index]))
        features[f"mfcc_{index:02d}_std"] = float(np.std(mfcc[:, index]))
    return features


def build_modma_audio_features(
    root: Path | str,
    subject_index: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """Extract one feature row per readable MODMA WAV segment."""
    root = Path(root)
    rows = []
    errors = []
    metadata = subject_index.set_index("subject_id")
    for subject_id, subject_row in metadata.iterrows():
        subject_dir = Path(subject_row["subject_path"])
        for path in sorted(
            subject_dir.glob("*.wav"),
            key=lambda item: int(item.stem) if item.stem.isdigit() else 10**9,
        ):
            if not path.stem.isdigit():
                continue
            segment = int(path.stem)
            try:
                features = extract_modma_wav_features(path)
            except Exception as exc:
                errors.append(
                    {
                        "subject_id": str(subject_id),
                        "segment": segment,
                        "filename": path.name,
                        "error": str(exc),
                    }
                )
                continue
            rows.append(
                {
                    "source": "MODMA",
                    "subject_id": str(subject_id),
                    "group": subject_row.get("group"),
                    "label": int(subject_row.get("label")),
                    "age": subject_row.get("age"),
                    "gender": subject_row.get("gender"),
                    "segment": segment,
                    "task_type": modma_task_type(segment),
                    "wav_path": str(path),
                    **features,
                }
            )
    if not rows:
        raise ValueError("No readable MODMA WAV files were found")
    return pd.DataFrame(rows), errors


def aggregate_modma_subject_features(utterance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate segment features by subject with mean and standard deviation."""
    metadata_columns = [
        "source",
        "subject_id",
        "group",
        "label",
        "age",
        "gender",
    ]
    feature_columns = [
        column
        for column in utterance.select_dtypes(include="number").columns
        if column not in {"label", "age", "segment"}
    ]
    grouped = utterance.groupby("subject_id", sort=True)
    rows = []
    for subject_id, group in grouped:
        first = group.iloc[0]
        row = {
            "source": "MODMA",
            "subject_id": subject_id,
            "group": first["group"],
            "label": int(first["label"]),
            "age": first["age"],
            "gender": first["gender"],
            "readable_segments": int(len(group)),
            "task_types": int(group["task_type"].nunique()),
        }
        for column in feature_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_task_mean"] = float(values.mean())
            row[f"{column}_task_std"] = float(values.std())
        rows.append(row)
    return pd.DataFrame(rows)


def train_modma_subject_baseline(
    subject_features: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run subject-grouped repeated cross-validation for MODMA audio features."""
    data = subject_features.dropna(subset=["label", "subject_id"]).copy()
    feature_columns = [
        column
        for column in data.select_dtypes(include="number").columns
        if column
        not in {
            "label",
            "age",
            "readable_segments",
            "task_types",
        }
    ]
    y = data["label"].astype(int)
    groups = data["subject_id"].astype(str)
    fold_rows = []
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE + repeat,
        )
        probabilities = np.full(len(data), np.nan, dtype=float)
        for fold, (train_idx, validation_idx) in enumerate(
            splitter.split(data, y, groups)
        ):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            )
            model.fit(data.iloc[train_idx][feature_columns], y.iloc[train_idx])
            probability = model.predict_proba(
                data.iloc[validation_idx][feature_columns]
            )[:, 1]
            probabilities[validation_idx] = probability
            prediction = (probability >= 0.5).astype(int)
            metrics = imbalanced_metrics_frame(
                y.iloc[validation_idx],
                prediction,
                probability,
            ).iloc[0]
            fold_rows.append(
                {
                    "source": "MODMA",
                    "repeat": repeat,
                    "fold": fold,
                    **{key: float(value) for key, value in metrics.to_dict().items()},
                    "ece": float(
                        expected_calibration_error(
                            y.iloc[validation_idx].to_numpy(),
                            probability,
                        )
                    ),
                    "n_train_subjects": int(groups.iloc[train_idx].nunique()),
                    "n_validation_subjects": int(
                        groups.iloc[validation_idx].nunique()
                    ),
                }
            )
        fold_rows[-n_splits]["oof_repeat_auc"] = float(
            imbalanced_metrics_frame(
                y,
                (probabilities >= 0.5).astype(int),
                probabilities,
            ).iloc[0]["auc"]
        )
    fold_report = pd.DataFrame(fold_rows)
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "auc",
        "pr_auc",
        "brier",
        "ece",
    ]
    summary = (
        fold_report.groupby("source")[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return fold_report, summary
