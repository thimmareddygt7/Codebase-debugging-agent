
import unittest
from calendarx.dates import days_in_month, is_leap_year


class DateTests(unittest.TestCase):
    def test_regular_years(self):
        self.assertTrue(is_leap_year(2024))
        self.assertFalse(is_leap_year(2023))

    def test_century_years(self):
        self.assertFalse(is_leap_year(1900))
        self.assertTrue(is_leap_year(2000))

    def test_february_length(self):
        self.assertEqual(days_in_month(2000, 2), 29)
        self.assertEqual(days_in_month(1900, 2), 28)

    def test_other_months(self):
        self.assertEqual(days_in_month(2023, 4), 30)
        self.assertEqual(days_in_month(2023, 12), 31)
