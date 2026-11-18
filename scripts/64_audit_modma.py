#!/usr/bin/env python3
"""Audit the authorized MODMA auxiliary audio cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

from elderly_depression.data.modma import (
    MODMA_VERIFIED_ARCHIVE,
    build_modma_subject_index,
    materialize_modma_audio_root,
    write_modma_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "modma"
DEFAULT_METADATA = (
    DEFAULT_ROOT / "metadata" / "subjects_information_audio_lanzhou_2015.xlsx"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "modma"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-wav-headers",
        action="store_true",
        help="Only count filenames; skip WAV header readability checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with materialize_modma_audio_root(args.root) as (audio_root, extracted):
        index, audit = build_modma_subject_index(
            audio_root,
            args.metadata,
            inspect_wav_headers=not args.skip_wav_headers,
        )
        if extracted:
            index["subject_path"] = index["subject_id"].map(
                lambda subject_id: (
                    f"archive://{MODMA_VERIFIED_ARCHIVE}/{subject_id}"
                )
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "modma_subject_index.csv"
    audit_path = args.output_dir / "modma_audio_audit.json"
    index.to_csv(index_path, index=False)
    write_modma_audit(audit, audit_path)
    print(f"subject index: {index_path}")
    print(f"audit: {audit_path}")
    print(
        "audio complete: "
        f"{audit['audio']['complete_dataset']}; "
        f"subjects: {audit['audio']['complete_subjects']}/"
        f"{audit['metadata']['subjects']}; "
        f"WAV: {audit['audio']['actual_wav_files']}/"
        f"{audit['audio']['expected_wav_files']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
