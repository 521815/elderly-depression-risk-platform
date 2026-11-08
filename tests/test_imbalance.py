import unittest

import numpy as np
import pandas as pd

from elderly_depression.imbalance import best_threshold, resample_train, scale_pos_weight


class ImbalanceTests(unittest.TestCase):
    def test_scale_pos_weight(self):
        y = pd.Series([0, 0, 0, 1])
        self.assertEqual(scale_pos_weight(y), 3.0)

    def test_resample_none(self):
        X = pd.DataFrame({"a": [1, 2, 3, 4]})
        y = pd.Series([0, 0, 0, 1])
        X2, y2 = resample_train(X, y, method="none")
        self.assertEqual(len(X2), 4)

    def test_threshold_f1(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])
        t = best_threshold(y, p, metric="f1")
        self.assertGreaterEqual(t, 0.4)
        self.assertLessEqual(t, 0.6)


if __name__ == "__main__":
    unittest.main()
