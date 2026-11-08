"""Compute fold-level Wilcoxon tests with Holm correction for MPDD fusion."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def holm_adjust(raw_p: list[float]) -> list[float]:
    n = len(raw_p)
    order = sorted(range(n), key=lambda i: raw_p[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, raw_p[idx] * (n - rank))
        adjusted[idx] = min(running, 1.0)
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-report",
        default=str(PROJECT_ROOT / "output" / "multimodal_subject_fold_report.csv"),
    )
    parser.add_argument("--output", default=str(PROJECT_ROOT / "output"))
    args = parser.parse_args()

    report_path = Path(args.fold_report)
    if not report_path.exists():
        print(f"fold report not found: {report_path}", file=sys.stderr)
        raise SystemExit(1)

    rows = read_rows(report_path)
    by_key: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (row["repeat"], row["fold"])
        by_key[key][row["modality"]] = {
            "auc": float(row["auc"]),
            "brier": float(row["brier"]),
        }

    keys = sorted(by_key)
    if not keys:
        raise SystemExit("no fold rows found")

    results: list[dict[str, float | str]] = []
    for metric in ("auc", "brier"):
        fusion = np.array([by_key[k]["fusion"][metric] for k in keys])
        for modality in ("face", "audio", "gait", "personality"):
            other = np.array([by_key[k][modality][metric] for k in keys])
            stat, p_value = stats.wilcoxon(fusion, other)
            direction = "fusion_higher" if fusion.mean() > other.mean() else "fusion_lower"
            results.append(
                {
                    "metric": metric,
                    "other_modality": modality,
                    "n_pairs": len(keys),
                    "wilcoxon_statistic": float(stat),
                    "raw_p": float(p_value),
                    "direction": direction,
                    "fusion_mean": float(fusion.mean()),
                    "other_mean": float(other.mean()),
                }
            )

    adjusted = holm_adjust([float(r["raw_p"]) for r in results])
    for row, p_value in zip(results, adjusted):
        row["holm_p"] = float(p_value)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "multimodal_wilcoxon_holm.csv"
    json_path = output_dir / "multimodal_wilcoxon_holm.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    json_path.write_text(
        json.dumps(
            {
                "fold_report": str(report_path),
                "n_pairs": len(keys),
                "method": "Wilcoxon signed-rank test, Holm correction over 8 comparisons",
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {csv_path}")
    print(f"saved {json_path}")


if __name__ == "__main__":
    main()
