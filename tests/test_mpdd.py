import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from elderly_depression.data.mpdd import (
    add_mpdd_binary_target,
    build_feature_table,
    build_utterance_feature_table,
    mpdd_label_audit,
)


class MPDDLoaderTests(unittest.TestCase):
    def test_official_label2_takes_precedence_over_score_threshold(self):
        df = pd.DataFrame(
            {
                "subject_id": ["a", "b", "c"],
                "label2": [1, 0, np.nan],
                "phq9_score": [5, 2, 12],
                "hamd_score": [5, 2, 12],
            }
        )

        labeled = add_mpdd_binary_target(df)

        self.assertEqual(labeled["mpdd_binary"].tolist(), [1, 0, 1])
        self.assertEqual(
            labeled["mpdd_label_source"].tolist(),
            ["label2", "label2", "phq9_score>=8"],
        )
        audit = mpdd_label_audit(labeled)
        self.assertEqual(audit["labeled_subjects"], 2)
        self.assertEqual(audit["unlabeled_subjects"], 1)
        self.assertEqual(audit["label2_positive"], 1)
        self.assertEqual(audit["threshold_positive"], 1)
        self.assertEqual(audit["threshold_disagreement"], 1)

    def test_build_table_from_official_npy_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mpdd"
            elder = root / "Train-MPDD-Elder" / "Elder"
            audio = elder / "Audio" / "train" / "mfcc" / "1"
            video = elder / "Video" / "train" / "densenet" / "1"
            imu = elder / "IMU" / "train" / "1"
            audio.mkdir(parents=True)
            video.mkdir(parents=True)
            imu.mkdir(parents=True)

            np.save(audio / "A_1.npy", np.ones((3, 2), dtype=np.float32) * 2)
            np.save(audio / "A_2.npy", np.ones((2, 2), dtype=np.float32) * 4)
            np.save(video / "V_1.npy", np.arange(6, dtype=np.float32).reshape(2, 3))
            np.save(imu / "1.npy", np.array([[1, 2], [3, 4]], dtype=np.float32))

            labels = elder / "split_labels_train.csv"
            pd.DataFrame(
                [{"split": "train", "ID": 1, "PHQ-9": 5, "label2": 0, "label3": 0}]
            ).to_csv(labels, index=False, encoding="utf-8-sig")

            embeddings = np.array(
                [{"id": 1, "embedding": np.arange(4, dtype=np.float32)}],
                dtype=object,
            )
            np.save(elder / "descriptions_embeddings_with_ids.npy", embeddings)

            df = build_feature_table(mpdd_root=root, output_path=root / "out.csv")

        self.assertEqual(len(df), 1)
        self.assertEqual(int(df.loc[0, "hamd_score"]), 5)
        self.assertEqual(int(df.loc[0, "mpdd_binary"]), 0)
        self.assertEqual(df.loc[0, "mpdd_label_source"], "label2")
        self.assertIn("audio_mfcc_mean_0", df.columns)
        self.assertIn("audio_mfcc_max_1", df.columns)
        self.assertIn("face_densenet_std_2", df.columns)
        self.assertIn("gait_imu_min_1", df.columns)
        self.assertIn("personality_3", df.columns)

    def test_build_utterance_table_from_official_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mpdd"
            elder = root / "Train-MPDD-Elder" / "Elder"
            audio = elder / "Audio" / "train" / "mfcc" / "1"
            video = elder / "Video" / "train" / "densenet" / "1"
            imu = elder / "IMU" / "train" / "1"
            audio.mkdir(parents=True)
            video.mkdir(parents=True)
            imu.mkdir(parents=True)

            np.save(audio / "A_1.npy", np.ones((2, 2), dtype=np.float32))
            np.save(audio / "A_2.npy", np.ones((3, 2), dtype=np.float32) * 3)
            np.save(video / "V_1.npy", np.ones((2, 2), dtype=np.float32) * 5)
            np.save(imu / "1.npy", np.ones((2, 2), dtype=np.float32) * 7)

            labels = elder / "split_labels_train.csv"
            pd.DataFrame(
                [{"split": "train", "ID": 1, "PHQ-9": 5, "label2": 0, "label3": 0}]
            ).to_csv(labels, index=False, encoding="utf-8-sig")
            embeddings = np.array(
                [{"id": 1, "embedding": np.arange(2, dtype=np.float32)}],
                dtype=object,
            )
            np.save(elder / "descriptions_embeddings_with_ids.npy", embeddings)

            df = build_utterance_feature_table(mpdd_root=root, output_path=root / "utter.csv")

        self.assertEqual(len(df), 4)
        self.assertEqual(set(df["modality"]), {"audio_mfcc", "face_densenet", "gait_imu"})
        self.assertEqual(set(df["fold"]), {"train"})
        self.assertEqual(int(df.loc[0, "hamd_score"]), 5)
        self.assertEqual(int(df.loc[0, "mpdd_binary"]), 0)
        self.assertIn("personality_1", df.columns)


if __name__ == "__main__":
    unittest.main()
