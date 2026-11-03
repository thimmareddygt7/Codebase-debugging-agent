
import unittest
from mathlib.series import sum_to


class SeriesTests(unittest.TestCase):
    def test_small(self):
        self.assertEqual(sum_to(4), 10)

    def test_one(self):
        self.assertEqual(sum_to(1), 1)

    def test_zero(self):
        self.assertEqual(sum_to(0), 0)

    def test_hundred(self):
        self.assertEqual(sum_to(100), 5050)
