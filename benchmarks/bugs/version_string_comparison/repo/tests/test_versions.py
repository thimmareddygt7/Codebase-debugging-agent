
import unittest
from updater.versions import is_newer


class VersionTests(unittest.TestCase):
    def test_minor_ten_beats_nine(self):
        self.assertTrue(is_newer("1.10.0", "1.9.0"))

    def test_older_is_not_newer(self):
        self.assertFalse(is_newer("1.9.0", "1.10.0"))

    def test_major_bump(self):
        self.assertTrue(is_newer("2.0.0", "1.9.9"))

    def test_equal_is_not_newer(self):
        self.assertFalse(is_newer("1.0.0", "1.0.0"))
