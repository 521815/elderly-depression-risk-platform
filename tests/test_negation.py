import unittest

from elderly_depression.negation import (
    has_any_unnegated_keyword,
    has_unnegated_keyword,
)


class NegationTests(unittest.TestCase):
    def test_negated_keyword_is_ignored(self):
        self.assertFalse(has_unnegated_keyword("我没有失眠", "失眠"))
        self.assertTrue(has_unnegated_keyword("最近经常失眠", "失眠"))

    def test_any_keyword(self):
        self.assertFalse(
            has_any_unnegated_keyword("没有胸痛，也没有喘不上气", ["胸痛", "喘不上气"])
        )
        self.assertTrue(
            has_any_unnegated_keyword("胸口疼，还喘不上气", ["胸痛", "胸口疼"])
        )


if __name__ == "__main__":
    unittest.main()
