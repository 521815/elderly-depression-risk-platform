"""Re-run real-vs-synthetic comparison after profile-guided generation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.services import generate_real_profile, generate_synthetic


def main() -> None:
    profile = generate_real_profile()
    print("Real profile generated:", profile["tabular"]["rows"], "tabular rows")
    generate_synthetic(dialogue_n=80, tabular_n=400, multimodal_n=300, seed=42)

    v1_path = PROJECT_ROOT / "scripts" / "10_compare_synthetic.py"
    spec = importlib.util.spec_from_file_location("compare_synthetic_v1", v1_path)
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)

    report = pd.DataFrame(v1.compare_tabular() + v1.compare_multimodal())
    out = PROJECT_ROOT / "output" / "synthetic_comparison_v2_report.csv"
    report.to_csv(out, index=False, encoding="utf-8-sig")
    print(report.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
