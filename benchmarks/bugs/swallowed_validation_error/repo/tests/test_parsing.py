
import unittest
from orders.parsing import parse_quantity


class ParsingTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse_quantity("5"), 5)
        self.assertEqual(parse_quantity(7), 7)

    def test_garbage_raises(self):
        for bad in ("abc", "", None, "3.5"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_quantity(bad)

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            parse_quantity("-1")
