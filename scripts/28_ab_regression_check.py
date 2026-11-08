"""Run the A/B comparison and append a regression-check history row."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    script = PROJECT_ROOT / "scripts" / "20_ab_comparison.py"
    subprocess.run([sys.executable, str(script)], check=True, cwd=PROJECT_ROOT)
    summary = json.loads(
        (PROJECT_ROOT / "output" / "ab_comparison_summary.json").read_text(encoding="utf-8")
    )
    tabular = {row["source"]: row for row in summary["tabular"]}
    multimodal = {row["source"]: row for row in summary["multimodal"]}
    row = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "tabular_real_auc": tabular.get("real", {}).get("mean_auc"),
        "tabular_synthetic_auc": tabular.get("synthetic", {}).get("mean_auc"),
        "multimodal_real_auc": multimodal.get("real", {}).get("auc_mean"),
        "multimodal_synthetic_auc": multimodal.get("synthetic", {}).get("auc_mean"),
        "dialogue_new_text_auc": summary["dialogue"]["new_text_auc"],
    }
    history = PROJECT_ROOT / "output" / "ab_check_history.csv"
    header = not history.exists()
    import csv

    with history.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if header:
            writer.writeheader()
        writer.writerow(row)
    print("A/B regression check appended:", row)


if __name__ == "__main__":
    main()
