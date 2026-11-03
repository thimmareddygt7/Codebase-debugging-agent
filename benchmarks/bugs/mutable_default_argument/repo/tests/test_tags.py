
import unittest
from tagging.tags import add_tag


class TagTests(unittest.TestCase):
    def test_new_list_each_call(self):
        self.assertEqual(add_tag("a"), ["a"])
        self.assertEqual(add_tag("b"), ["b"])

    def test_existing_list_is_extended(self):
        existing = ["x"]
        self.assertEqual(add_tag("y", existing), ["x", "y"])
