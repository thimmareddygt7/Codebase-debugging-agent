
import unittest
from shop.shipping import shipping_cost


class ShippingTests(unittest.TestCase):
    def test_below_threshold_pays(self):
        self.assertEqual(shipping_cost(49.99), 5.99)

    def test_exact_threshold_is_free(self):
        self.assertEqual(shipping_cost(50.0), 0.0)

    def test_above_threshold_is_free(self):
        self.assertEqual(shipping_cost(100.0), 0.0)
