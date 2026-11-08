#!/usr/bin/env python3
"""Build the de-identified subject-level CMDC feature table."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elderly_depression.config import CMDC_ROOT
from elderly_depression.services import CMDC_AUDIT_JSON, CMDC_CSV, build_cmdc_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(CMDC_ROOT))
    parser.add_argument("--output", default=str(CMDC_CSV))
    parser.add_argument("--audit", default=str(CMDC_AUDIT_JSON))
    parser.add_argument(
        "--password-env",
        default="CMDC_ARCHIVE_PASSWORD",
        help="环境变量名；密码不会写入命令历史、项目或日志",
    )
    args = parser.parse_args()

    password = os.getenv(args.password_env)
    if not password:
        password = getpass.getpass("CMDC 归档密码（仅本次使用）: ")
    if not password:
        raise SystemExit("未提供 CMDC 归档密码")

    from elderly_depression.data.cmdc import build_cmdc_feature_table

    def progress(value: float, stage: str) -> None:
        print(f"{value * 100:5.1f}% {stage}", flush=True)

    df, audit = build_cmdc_feature_table(
        cmdc_root=Path(args.root),
        output_path=Path(args.output),
        audit_path=Path(args.audit),
        password=password,
        progress=progress,
    )
    print(
        json.dumps(
            {
                "rows": len(df),
                "columns": len(df.columns),
                "hc": audit["hc_subjects"],
                "mdd": audit["mdd_subjects"],
                "complete_multimodal": audit["complete_multimodal_subjects"],
                "output": str(Path(args.output)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
