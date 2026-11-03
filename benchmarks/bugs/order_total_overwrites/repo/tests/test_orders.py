
import unittest
from shop.orders import order_total


class OrderTotalTests(unittest.TestCase):
    def test_single_item(self):
        self.assertEqual(order_total([{"price": 2.5, "qty": 2}]), 5.0)

    def test_multiple_items(self):
        items = [{"price": 2.5, "qty": 2}, {"price": 10.0, "qty": 1}, {"price": 0.99, "qty": 3}]
        self.assertEqual(order_total(items), 17.97)

    def test_empty(self):
        self.assertEqual(order_total([]), 0)
