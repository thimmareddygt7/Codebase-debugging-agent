
import unittest
from textutils.checks import is_palindrome


class PalindromeTests(unittest.TestCase):
    def test_simple(self):
        self.assertTrue(is_palindrome("Racecar"))

    def test_sentence_with_punctuation(self):
        self.assertTrue(is_palindrome("A man, a plan, a canal: Panama"))

    def test_not_a_palindrome(self):
        self.assertFalse(is_palindrome("hello"))

    def test_empty_string(self):
        self.assertTrue(is_palindrome(""))
