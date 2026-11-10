"""Compare official MPDD label2 with the legacy PHQ-9 threshold under one protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.data.mpdd import mpdd_label_audit
from elderly_depression.models.multimodal import train_subject_late_fusion
from elderly_depression.smoothing import aggregate_mpdd_subjects


def main() -> None:
    utterance_csv = PROJECT_ROOT / "data" / "mpdd_utterance_features.csv"
    raw = pd.read_csv(utterance_csv)
    raw["subject_id"] = raw["subject_id"].astype(str)
    subject = aggregate_mpdd_subjects(utterance_csv, method="mean")

    official = subject.copy()
    official["_label"] = official["_label"].astype(int)

    legacy = subject.copy()
    phq9 = raw.groupby("subject_id")["phq9_score"].first()
    legacy["_label"] = (legacy["subject_id"].map(phq9) >= 8).astype(int)

    rows = []
    for protocol, frame in [
        ("official_label2", official),
        ("legacy_phq9_ge_8", legacy),
    ]:
        result = train_subject_late_fusion(
            frame,
            prefixes=["face", "audio", "gait", "personality"],
            label_col="_label",
            output_dir=None,
            n_splits=5,
            n_repeats=5,
            pca_dims=64,
            threshold_policy="fixed",
        )
        summary = result["summary"].copy()
        summary.insert(0, "label_protocol", protocol)
        rows.append(summary)

    report = pd.concat(rows, ignore_index=True)
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(
        output_dir / "mpdd_label_protocol_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit = mpdd_label_audit(raw)
    payload = {
        "label_audit": audit,
        "protocol": "subject_mean__pca64__l2_logistic__equal_fusion__threshold_0.5",
        "repeats": 5,
        "splits": 5,
    }
    (output_dir / "mpdd_label_protocol_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        report[
            [
                "label_protocol",
                "modality",
                "accuracy_mean",
                "f1_mean",
                "auc_mean",
                "pr_auc_mean",
                "brier_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
