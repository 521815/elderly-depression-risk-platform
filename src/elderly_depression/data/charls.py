"""Build a standardized elderly risk table from Harmonized CHARLS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CESD_THRESHOLD, DATA_DIR, HARMONIZED_GLOB, MIN_AGE

CHRONIC_FIELDS = [
    "hibpe",
    "diabe",
    "cancre",
    "lunge",
    "hearte",
    "stroke",
    "psyche",
    "arthre",
    "dyslipe",
    "livere",
    "kidneye",
    "digeste",
    "asthmae",
    "memrye",
]

WAVE_FIELDS = {
    "age": "agey",
    "marital_status": "mstat",
    "self_rated_health": "shlt",
    "adl_score": "adla_c",
    "mobility_score": "mobilsev",
    "vigorous_activity": "vgact_c",
    "moderate_activity": "mdact_c",
    "light_activity": "ltact_c",
    "sleep_quality": "sleeprl",
    "smoker": "smokev",
    "drinker": "drinkev",
    "self_memory": "slfmem",
    "immediate_recall": "imrc",
    "delayed_recall": "dlrc",
    "serial7": "ser7",
    "income_earn": "itearn",
    "income_pension": "ipen",
    "social_work": "socwk",
    "cesd_score": "cesd10",
}

WAVE_ALT_SUFFIX = {
    "self_rated_health": {4: "shlta"},
}


def find_harmonized_dta(charls_root: Path | str) -> Path:
    root = Path(charls_root)
    matches = sorted(root.glob(HARMONIZED_GLOB))
    if not matches:
        raise FileNotFoundError(f"No Harmonized CHARLS dta found under {root}")
    return matches[0]


def _read_harmonized(path: Path) -> pd.DataFrame:
    needed = ["id"]
    for wave in range(1, 5):
        needed.append(f"r{wave}agey")
        needed.append("ragender")
        needed.append("raeduc_c")
        for out_name, raw_suffix in WAVE_FIELDS.items():
            raw_suffix = WAVE_ALT_SUFFIX.get(out_name, {}).get(wave, raw_suffix)
            needed.append(f"r{wave}{raw_suffix}")
        for cond in CHRONIC_FIELDS:
            needed.append(f"r{wave}{cond}")
    reader = pd.read_stata(path, convert_categoricals=False, iterator=True, chunksize=1)
    first = next(reader)
    available = set(first.columns)
    needed = [c for c in sorted(set(needed)) if c in available]
    return pd.read_stata(path, convert_categoricals=False, columns=needed)


def _melt_waves(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for wave in range(1, 5):
        cols = {"id": "id", "ragender": "gender", "raeduc_c": "education"}
        for out_name, raw_suffix in WAVE_FIELDS.items():
            raw_suffix = WAVE_ALT_SUFFIX.get(out_name, {}).get(wave, raw_suffix)
            col = f"r{wave}{raw_suffix}"
            if col in df.columns:
                cols[col] = out_name
        for cond in CHRONIC_FIELDS:
            cols[f"r{wave}{cond}"] = f"chronic_{cond}"
        sub = df[list(cols)].rename(columns=cols)
        sub["wave"] = wave
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    chronic_cols = [f"chronic_{c}" for c in CHRONIC_FIELDS]
    for c in chronic_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([-999, -998], np.nan)
    df["comorbidity_count"] = df[chronic_cols].eq(1).sum(axis=1, min_count=1)
    df["cesd_score"] = pd.to_numeric(df["cesd_score"], errors="coerce")
    df["depression"] = (df["cesd_score"] >= CESD_THRESHOLD).astype(int)
    return df


def build_standardized(
    charls_root: Path | str | None = None,
    min_age: int = MIN_AGE,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    root = Path(charls_root) if charls_root else None
    if root is None:
        from ..config import CHARLS_ROOT

        root = CHARLS_ROOT
    dta_path = find_harmonized_dta(root)
    df = _read_harmonized(dta_path)
    df = _melt_waves(df)
    df = _add_derived(df)

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["cesd_score"] = pd.to_numeric(df["cesd_score"], errors="coerce")
    df = df[df["age"] >= min_age].copy()
    df = df[df["cesd_score"].notna()].copy()
    df = df.sort_values(["id", "wave"]).reset_index(drop=True)

    if output_path is None:
        output_path = DATA_DIR / "charls_standardized.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df
