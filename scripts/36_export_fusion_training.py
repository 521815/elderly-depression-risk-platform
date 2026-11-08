"""Export user assessments with independent follow-up labels as fusion training data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.records import USER_DIR, list_assessments, list_users


def main() -> None:
    rows = []
    for user in list_users():
        label = user.get("followup_label")
        if label not in (0, 1):
            continue
        for record in list_assessments(user["id"]):
            rows.append(
                {
                    "user_id": user["id"],
                    "record_id": record.get("id"),
                    "created_at": record.get("created_at"),
                    "text_risk": record.get("text_risk"),
                    "scale_risk": record.get("scale_risk"),
                    "context_risk": record.get("context_risk"),
                    "expression_risk": record.get("expression_risk"),
                    "mpdd_risk": record.get("mpdd_risk"),
                    "depression": label,
                }
            )
    out = PROJECT_ROOT / "data" / "fusion_training_data.csv"
    df = pd.DataFrame(rows)
    if df.empty:
        pd.DataFrame(
            columns=[
                "user_id",
                "record_id",
                "created_at",
                "text_risk",
                "scale_risk",
                "context_risk",
                "expression_risk",
                "mpdd_risk",
                "depression",
            ]
        ).to_csv(out, index=False, encoding="utf-8-sig")
        print("暂无带独立标签的用户记录，已生成空模板。")
    else:
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(df.to_string(index=False))
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
