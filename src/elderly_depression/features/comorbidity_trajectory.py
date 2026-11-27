"""Leakage-safe chronic-disease trajectory features.

The current CHARLS table contains disease states by survey wave but not the
reported age of first diagnosis. These features therefore use the first wave in
which a diagnosis is observed. They must be interpreted as wave-observed
trajectories, not exact clinical onset trajectories.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


CHRONIC_DISEASE_LABELS = {
    "hibpe": "高血压",
    "diabe": "糖尿病",
    "cancre": "癌症",
    "lunge": "慢性肺病",
    "hearte": "心脏病",
    "stroke": "卒中",
    "psyche": "情绪与精神问题",
    "arthre": "关节炎",
    "dyslipe": "血脂异常",
    "livere": "肝病",
    "kidneye": "肾病",
    "digeste": "消化系统疾病",
    "asthmae": "哮喘",
    "memrye": "记忆相关疾病",
}

# Chen et al. (2024) placed these conditions at the centre or terminal end of
# the CHARLS disease-trajectory network.
CENTRAL_DISEASES = ("hibpe", "hearte", "dyslipe", "digeste")
TERMINAL_DISEASES = ("stroke", "diabe", "memrye")
ENTRY_DISEASE = "arthre"

WAVE_YEAR = {
    1: 2011,
    2: 2013,
    3: 2015,
    4: 2018,
}

COMORBIDITY_TRAJECTORY_FEATURES = [
    "trajectory_active_burden",
    "trajectory_new_diagnoses_this_wave",
    "trajectory_recent_diagnosis",
    "trajectory_observation_years",
    "trajectory_first_order_unambiguous",
    "trajectory_entry_burden",
    "trajectory_post_entry_burden",
    "trajectory_second_delay_years",
    "trajectory_progression_rate",
    "trajectory_central_burden",
    "trajectory_terminal_burden",
    "trajectory_first_is_arthritis",
    "trajectory_arthritis_to_hypertension",
    "trajectory_arthritis_hypertension_to_dyslipidemia",
    "trajectory_arthritis_hypertension_to_heart",
    "trajectory_arthritis_hypertension_to_diabetes",
]


def _chronic_columns(df: pd.DataFrame) -> list[str]:
    return [
        f"chronic_{disease}"
        for disease in CHRONIC_DISEASE_LABELS
        if f"chronic_{disease}" in df.columns
    ]


def _normalise_binary(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values.isin([0, 1]))


def _second_delay(first_waves: dict[str, float]) -> float:
    ordered = sorted(set(first_waves.values()))
    if len(ordered) < 2:
        return np.nan
    first_wave = ordered[0]
    second_wave = ordered[1]
    first_year = WAVE_YEAR.get(int(first_wave), first_wave)
    second_year = WAVE_YEAR.get(int(second_wave), second_wave)
    return float(second_year - first_year)


def _strictly_precedes(
    first_waves: dict[str, float],
    sequence: Iterable[str],
) -> int:
    sequence = list(sequence)
    waves = [first_waves.get(disease) for disease in sequence]
    if any(wave is None or pd.isna(wave) for wave in waves):
        return 0
    return int(all(left < right for left, right in zip(waves, waves[1:])))


def append_comorbidity_trajectory_features(
    df: pd.DataFrame,
    *,
    id_col: str = "id",
    wave_col: str = "wave",
) -> pd.DataFrame:
    """Append wave-observed comorbidity trajectory features.

    Features for a row use only that row and earlier waves from the same
    subject. This prevents diagnoses observed at later waves from leaking into
    earlier depression-risk predictions.
    """
    if id_col not in df.columns:
        raise ValueError(f"Comorbidity trajectory features require {id_col!r}")
    if wave_col not in df.columns:
        raise ValueError(f"Comorbidity trajectory features require {wave_col!r}")

    result = df.copy().reset_index(drop=True)
    chronic_cols = _chronic_columns(result)
    if not chronic_cols:
        raise ValueError("No chronic disease columns are available")

    for column in chronic_cols:
        result[column] = _normalise_binary(result[column])

    result["__trajectory_row__"] = np.arange(len(result))
    result["__trajectory_id__"] = result[id_col].astype(str)
    result["__trajectory_wave__"] = pd.to_numeric(
        result[wave_col],
        errors="coerce",
    )
    ordered = result.sort_values(
        ["__trajectory_id__", "__trajectory_wave__", "__trajectory_row__"],
        kind="mergesort",
    )

    for feature in COMORBIDITY_TRAJECTORY_FEATURES:
        result[feature] = np.nan

    for _, subject_rows in ordered.groupby(
        "__trajectory_id__",
        sort=False,
    ):
        first_observed: dict[str, float] = {}
        for position, row in subject_rows.iterrows():
            wave = row["__trajectory_wave__"]
            if pd.isna(wave):
                continue

            known_values = {
                column: row[column]
                for column in chronic_cols
                if pd.notna(row[column])
            }
            if not known_values:
                continue

            active = {
                column: int(value)
                for column, value in known_values.items()
                if int(value) == 1
            }
            active_fields = {
                column.removeprefix("chronic_") for column in active
            }
            new_fields = active_fields - set(first_observed)
            for disease in new_fields:
                first_observed[disease] = float(wave)

            first_wave = min(first_observed.values()) if first_observed else np.nan
            entry_burden = (
                sum(
                    wave_value == first_wave
                    for wave_value in first_observed.values()
                )
                if first_observed
                else np.nan
            )
            observed_years = (
                float(
                    WAVE_YEAR.get(int(wave), wave)
                    - WAVE_YEAR.get(int(first_wave), first_wave)
                )
                if first_observed
                else np.nan
            )
            post_entry_burden = (
                len(first_observed) - entry_burden
                if first_observed
                else np.nan
            )
            second_delay = _second_delay(first_observed)

            result.at[position, "trajectory_active_burden"] = float(
                sum(active.values())
            )
            result.at[position, "trajectory_new_diagnoses_this_wave"] = float(
                len(new_fields)
            )
            result.at[position, "trajectory_recent_diagnosis"] = float(
                bool(new_fields)
            )
            result.at[position, "trajectory_observation_years"] = (
                observed_years
            )
            result.at[position, "trajectory_first_order_unambiguous"] = (
                float(entry_burden == 1) if first_observed else np.nan
            )
            result.at[position, "trajectory_entry_burden"] = entry_burden
            result.at[position, "trajectory_post_entry_burden"] = (
                post_entry_burden
            )
            result.at[position, "trajectory_second_delay_years"] = (
                second_delay
            )
            result.at[position, "trajectory_progression_rate"] = (
                float(post_entry_burden / max(observed_years, 1.0))
                if first_observed
                else np.nan
            )
            result.at[position, "trajectory_central_burden"] = float(
                len(active_fields.intersection(CENTRAL_DISEASES))
            )
            result.at[position, "trajectory_terminal_burden"] = float(
                len(active_fields.intersection(TERMINAL_DISEASES))
            )
            result.at[position, "trajectory_first_is_arthritis"] = float(
                ENTRY_DISEASE in first_observed
                and first_observed[ENTRY_DISEASE] == first_wave
            )
            result.at[
                position,
                "trajectory_arthritis_to_hypertension",
            ] = float(
                _strictly_precedes(first_observed, ("arthre", "hibpe"))
            )
            result.at[
                position,
                "trajectory_arthritis_hypertension_to_dyslipidemia",
            ] = float(
                _strictly_precedes(
                    first_observed,
                    ("arthre", "hibpe", "dyslipe"),
                )
            )
            result.at[
                position,
                "trajectory_arthritis_hypertension_to_heart",
            ] = float(
                _strictly_precedes(
                    first_observed,
                    ("arthre", "hibpe", "hearte"),
                )
            )
            result.at[
                position,
                "trajectory_arthritis_hypertension_to_diabetes",
            ] = float(
                _strictly_precedes(
                    first_observed,
                    ("arthre", "hibpe", "diabe"),
                )
            )

    return result.drop(
        columns=[
            "__trajectory_row__",
            "__trajectory_id__",
            "__trajectory_wave__",
        ]
    )
