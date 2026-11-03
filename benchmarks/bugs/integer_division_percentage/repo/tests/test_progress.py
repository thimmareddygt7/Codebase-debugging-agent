
import unittest
from reports.progress import percent


class PercentTests(unittest.TestCase):
    def test_quarter(self):
        self.assertEqual(percent(1, 4), 25.0)

    def test_complete(self):
        self.assertEqual(percent(3, 3), 100.0)

    def test_zero_whole(self):
        self.assertEqual(percent(5, 0), 0.0)

    def test_fractional(self):
        self.assertAlmostEqual(percent(1, 3), 33.3333, places=3)
