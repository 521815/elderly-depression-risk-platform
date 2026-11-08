import io
import json
import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from elderly_depression.data.cmdc import build_cmdc_feature_table
from elderly_depression.models.cmdc import train_cmdc_models


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _arff_bytes(size: int, value: float) -> bytes:
    fields = ["unknown", *[f"{value + index * 0.01:.6f}" for index in range(size)], "0"]
    return (
        "@relation opensmile\n"
        "@attribute name string\n"
        f"@attribute f0 numeric\n@data\n{','.join(fields)}\n"
    ).encode("utf-8")


class CMDCLoaderTests(unittest.TestCase):
    def test_builds_deidentified_subject_level_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "part1.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("HC01/Q1.txt", "这是不应进入派生表的原始转写")
                zf.writestr(
                    "HC01/vector_Q1.csv",
                    ",".join(str(value) for value in np.arange(768) / 100),
                )
                zf.writestr("HC01/Q1.npy", _npy_bytes(np.arange(768) / 200))
                zf.writestr(
                    "HC01/Q1.pkl",
                    pickle.dumps(np.arange(128, dtype=np.float32).reshape(1, 128)),
                )
                zf.writestr("HC01/Q1.wav.csv", _arff_bytes(88, 1.0))

                zf.writestr("MDD01/Q1.txt", "另一条不应进入派生表的原始转写")
                zf.writestr(
                    "MDD01/vector_Q1.csv",
                    ",".join(str(value) for value in np.arange(768) / 50),
                )
                zf.writestr(
                    "MDD01/Q1.pkl",
                    pickle.dumps(np.arange(128, dtype=np.float32).reshape(1, 128)),
                )
                zf.writestr("MDD01/Q1.wav.csv", _arff_bytes(88, 2.0))

            output = root / "cmdc_features.csv"
            audit_path = root / "cmdc_audit.json"
            df, audit = build_cmdc_feature_table(
                cmdc_root=root,
                output_path=output,
                audit_path=audit_path,
                password=None,
            )
            saved_audit = json.loads(audit_path.read_text(encoding="utf-8"))

        self.assertEqual(len(df), 2)
        self.assertEqual(set(df["group"]), {"HC", "MDD"})
        self.assertEqual(
            int(df.loc[df["subject_id"] == "HC01", "complete_multimodal"].iloc[0]),
            1,
        )
        self.assertEqual(
            int(df.loc[df["subject_id"] == "MDD01", "complete_multimodal"].iloc[0]),
            0,
        )
        self.assertTrue(any(column.startswith("text_mean_") for column in df.columns))
        self.assertFalse(any("不应进入" in str(value) for value in df.to_numpy().ravel()))
        self.assertFalse(saved_audit["raw_data_copied"])
        self.assertFalse(saved_audit["password_persisted"])


class CMDCModelTests(unittest.TestCase):
    def test_subject_aware_benchmark_writes_report_and_model(self):
        rng = np.random.default_rng(42)
        rows = []
        for index in range(30):
            label = index % 2
            row = {
                "subject_id": f"S{index:02d}",
                "label": label,
                "group": "MDD" if label else "HC",
                "text_questions": 3,
                "visual_questions": 0,
                "audio_deep_questions": 3,
                "acoustic_questions": 3,
                "complete_multimodal": 0,
            }
            for feature in range(8):
                row[f"text_mean_{feature:03d}"] = rng.normal() + label * 1.5
                row[f"text_std_{feature:03d}"] = rng.normal()
            for feature in range(6):
                row[f"audio_deep_mean_{feature:03d}"] = rng.normal() + label
                row[f"audio_deep_std_{feature:03d}"] = rng.normal()
            for feature in range(5):
                row[f"acoustic_mean_{feature:03d}"] = rng.normal() + label
                row[f"acoustic_std_{feature:03d}"] = rng.normal()
            rows.append(row)
        df = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = train_cmdc_models(
                df=df,
                report_path=root / "report.csv",
                summary_path=root / "summary.json",
                model_dir=root / "models",
                repeats=2,
            )
            report = pd.read_csv(root / "report.csv")
            model_files = list((root / "models").glob("*.pkl"))

        self.assertTrue(summary["ready"])
        self.assertIn("text_audio", set(report["variant_name"]))
        self.assertTrue(model_files)
        self.assertIn("age_scope_warning", summary)


if __name__ == "__main__":
    unittest.main()
