
import unittest
from stats.summary import mean, median


class SummaryTests(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(mean([1, 2, 3]), 2)

    def test_mean_of_empty_is_zero(self):
        self.assertEqual(mean([]), 0.0)

    def test_median(self):
        self.assertEqual(median([5, 1, 3]), 3)
        self.assertEqual(median([1, 2, 3, 4]), 2.5)
