import unittest

from elderly_depression.scales import score_cesd10, score_gds15, score_phq9


class ScaleTests(unittest.TestCase):
    def test_cesd10_reverse(self):
        self.assertEqual(score_cesd10([0] * 10), 6)
        self.assertEqual(score_cesd10([3] * 10), 24)

    def test_phq9(self):
        self.assertEqual(score_phq9([1, 1, 1, 1, 1, 1, 1, 1, 1]), 9)

    def test_gds15(self):
        self.assertEqual(score_gds15([0] * 15), 5)
        self.assertEqual(score_gds15([1] * 15), 10)


if __name__ == "__main__":
    unittest.main()
