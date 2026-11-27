import unittest

import numpy as np
import pandas as pd

from elderly_depression.features.comorbidity_trajectory import (
    COMORBIDITY_TRAJECTORY_FEATURES,
    append_comorbidity_trajectory_features,
)


def _trajectory_fixture() -> pd.DataFrame:
    rows = []
    for subject, wave, active in [
        ("A", 1, {"arthre"}),
        ("A", 2, {"arthre", "hibpe"}),
        ("A", 3, {"arthre", "hibpe", "dyslipe"}),
        ("B", 1, set()),
        ("B", 2, {"hearte"}),
        ("B", 3, {"hearte", "stroke"}),
        ("C", 1, {"arthre", "hibpe"}),
    ]:
        row = {
            "id": subject,
            "wave": wave,
            "depression": 0,
        }
        for disease in (
            "arthre",
            "hibpe",
            "dyslipe",
            "hearte",
            "stroke",
            "diabe",
            "memrye",
        ):
            row[f"chronic_{disease}"] = int(disease in active)
        rows.append(row)
    return pd.DataFrame(rows)


class ComorbidityTrajectoryTests(unittest.TestCase):
    def test_wave_trajectory_features_follow_first_observed_order(self):
        result = append_comorbidity_trajectory_features(_trajectory_fixture())
        subject_a = result[result["id"] == "A"].set_index("wave")

        self.assertEqual(subject_a.loc[1, "trajectory_active_burden"], 1)
        self.assertEqual(subject_a.loc[1, "trajectory_new_diagnoses_this_wave"], 1)
        self.assertEqual(subject_a.loc[1, "trajectory_observation_years"], 0)
        self.assertEqual(subject_a.loc[1, "trajectory_first_is_arthritis"], 1)
        self.assertTrue(
            np.isnan(subject_a.loc[1, "trajectory_second_delay_years"])
        )

        self.assertEqual(subject_a.loc[2, "trajectory_post_entry_burden"], 1)
        self.assertEqual(subject_a.loc[2, "trajectory_second_delay_years"], 2)
        self.assertEqual(
            subject_a.loc[2, "trajectory_arthritis_to_hypertension"],
            1,
        )

        self.assertEqual(subject_a.loc[3, "trajectory_observation_years"], 4)
        self.assertEqual(subject_a.loc[3, "trajectory_central_burden"], 2)
        self.assertEqual(subject_a.loc[3, "trajectory_progression_rate"], 0.5)
        self.assertEqual(
            subject_a.loc[
                3,
                "trajectory_arthritis_hypertension_to_dyslipidemia",
            ],
            1,
        )
        self.assertTrue(
            set(COMORBIDITY_TRAJECTORY_FEATURES).issubset(result.columns)
        )

    def test_features_do_not_use_future_wave_diagnoses(self):
        data = _trajectory_fixture()
        full = append_comorbidity_trajectory_features(data)
        truncated = append_comorbidity_trajectory_features(
            data[(data["id"] != "A") | (data["wave"] == 1)]
        )
        full_first = full[(full["id"] == "A") & (full["wave"] == 1)].iloc[0]
        truncated_first = truncated[
            (truncated["id"] == "A") & (truncated["wave"] == 1)
        ].iloc[0]
        for feature in COMORBIDITY_TRAJECTORY_FEATURES:
            if pd.isna(full_first[feature]):
                self.assertTrue(pd.isna(truncated_first[feature]))
            else:
                self.assertEqual(full_first[feature], truncated_first[feature])

        subject_b_wave_2 = full[(full["id"] == "B") & (full["wave"] == 2)].iloc[0]
        self.assertEqual(subject_b_wave_2["trajectory_terminal_burden"], 0)

    def test_same_wave_multiple_diseases_are_marked_ambiguous(self):
        result = append_comorbidity_trajectory_features(_trajectory_fixture())
        subject_c = result[result["id"] == "C"].iloc[0]
        self.assertEqual(subject_c["trajectory_entry_burden"], 2)
        self.assertEqual(subject_c["trajectory_first_order_unambiguous"], 0)

    def test_missing_wave_column_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "wave"):
            append_comorbidity_trajectory_features(
                _trajectory_fixture().drop(columns=["wave"])
            )


if __name__ == "__main__":
    unittest.main()
