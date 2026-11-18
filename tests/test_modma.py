import tempfile
import unittest
import wave
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np

from elderly_depression.data.modma import (
    build_modma_subject_index,
    load_modma_metadata,
    materialize_modma_audio_root,
)
from elderly_depression.features.modma_audio import (
    extract_modma_wav_features,
    modma_task_type,
)
from elderly_depression.models.modma_transfer import run_modma_to_mpdd_transfer


def _write_metadata(path: Path, subjects: int = 52) -> Path:
    rows = []
    for offset in range(subjects):
        group = "MDD" if offset < 23 else "HC"
        rows.append(
            {
                "subject id": 2010001 + offset,
                "type": group,
                "age": 18 + offset % 35,
                "gender": "F" if offset % 2 else "M",
                "education（years）": 12,
                "PHQ-9": 15 if group == "MDD" else 2,
                "CTQ-SF": 45,
                "LES": -10,
                "SSRS": 35,
                "GAD-7": 8 if group == "MDD" else 1,
                "PSQI": 10 if group == "MDD" else 3,
            }
        )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Sheet1", index=False)
    return path


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00")


class MODMAAuditTests(unittest.TestCase):
    def test_loads_official_metadata_columns_and_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = _write_metadata(Path(temp) / "metadata.xlsx", subjects=4)
            metadata = load_modma_metadata(path)
        self.assertEqual(len(metadata), 4)
        self.assertEqual(metadata["label"].tolist(), [1, 1, 1, 1])
        self.assertIn("phq9_score", metadata.columns)
        self.assertIn("psqi_score", metadata.columns)

    def test_audits_complete_29_segment_audio_structure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "modma"
            root.mkdir()
            metadata_path = _write_metadata(Path(temp) / "metadata.xlsx")
            for subject_index in range(52):
                subject_dir = root / f"0{2010001 + subject_index}"
                subject_dir.mkdir()
                for segment in range(1, 30):
                    _write_wav(subject_dir / f"{segment}.wav")
            index, audit = build_modma_subject_index(root, metadata_path)
        self.assertEqual(len(index), 52)
        self.assertTrue(audit["audio"]["complete_dataset"])
        self.assertEqual(audit["audio"]["actual_wav_files"], 1508)
        self.assertEqual(audit["audio"]["complete_subjects"], 52)
        self.assertEqual(audit["metadata"]["group_counts"], {"HC": 29, "MDD": 23})
        self.assertFalse(audit["scope"]["is_older_adult_primary_dataset"])
        self.assertEqual(audit["warnings"], [])

    def test_reports_missing_audio_and_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "modma"
            root.mkdir()
            metadata_path = _write_metadata(Path(temp) / "metadata.xlsx", subjects=3)
            subject_dir = root / "02010001"
            subject_dir.mkdir()
            _write_wav(subject_dir / "1.wav")
            _, audit = build_modma_subject_index(root, metadata_path)
        self.assertFalse(audit["audio"]["complete_dataset"])
        self.assertTrue(audit["warnings"])
        self.assertIn(
            "2010002",
            audit["pairing"]["metadata_subjects_missing_audio"],
        )

    def test_extracts_finite_acoustic_features_and_task_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.wav"
            sample_rate = 16000
            time = np.arange(sample_rate) / sample_rate
            signal = (0.2 * np.sin(2 * np.pi * 220 * time) * 32767).astype(
                np.int16
            )
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(signal.tobytes())
            features = extract_modma_wav_features(path)
        self.assertGreater(features["duration_seconds"], 0.9)
        self.assertEqual(features["sample_rate"], sample_rate)
        self.assertTrue(np.isfinite(features["mfcc_00_mean"]))
        self.assertEqual(modma_task_type(1), "interview")
        self.assertEqual(modma_task_type(19), "passage_reading")
        self.assertEqual(modma_task_type(20), "word_reading")
        self.assertEqual(modma_task_type(26), "picture_description")

    def test_transfer_uses_shared_target_splits_and_all_conditions(self):
        rng = np.random.default_rng(7)
        source_rows = []
        for subject in range(24):
            label = subject % 2
            source_rows.append(
                {
                    "source": "MODMA",
                    "subject_id": f"s{subject}",
                    "label": label,
                    "acoustic_a_task_mean": label + rng.normal(0, 0.4),
                    "acoustic_b_task_mean": rng.normal(0, 1),
                }
            )
        target_rows = []
        for subject in range(36):
            label = subject % 2
            target_rows.append(
                {
                    "source": "MPDD",
                    "subject_id": f"t{subject}",
                    "mpdd_binary": label,
                    "acoustic_a_task_mean": label + 0.3 + rng.normal(0, 0.5),
                    "acoustic_b_task_mean": rng.normal(0.2, 1),
                }
            )
        fold_report, comparisons, meta = run_modma_to_mpdd_transfer(
            pd.DataFrame(source_rows),
            pd.DataFrame(target_rows),
            n_splits=3,
            n_repeats=2,
        )
        self.assertEqual(
            set(fold_report["condition"]),
            {
                "target_only",
                "modma_zero_shot",
                "modma_probability_stacking",
                "joint_source_target",
            },
        )
        self.assertEqual(len(comparisons), 3)
        self.assertEqual(meta["source_subjects"], 24)
        self.assertEqual(meta["target_subjects"], 36)
        self.assertTrue(meta["same_target_splits"])

    def test_materializes_verified_archive_and_cleans_temporary_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "modma"
            subject = root / "02010001"
            subject.mkdir(parents=True)
            _write_wav(subject / "01.wav")
            archive = root / "audio_lanzhou_2015_verified.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.write(subject / "01.wav", "02010001/01.wav")
            subject.joinpath("01.wav").unlink()
            subject.rmdir()
            with materialize_modma_audio_root(root) as (audio_root, extracted):
                self.assertTrue(extracted)
                self.assertTrue((audio_root / "02010001" / "01.wav").exists())
                extracted_root = audio_root
            self.assertFalse(extracted_root.exists())


if __name__ == "__main__":
    unittest.main()
