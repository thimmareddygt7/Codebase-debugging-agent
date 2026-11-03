
import unittest
from game.leaderboard import top_scores

ENTRIES = [
    {"name": "zoe", "score": 90},
    {"name": "adam", "score": 40},
    {"name": "mia", "score": 90},
    {"name": "ben", "score": 75},
]


class LeaderboardTests(unittest.TestCase):
    def test_highest_scores_first(self):
        self.assertEqual([e["name"] for e in top_scores(ENTRIES, 2)], ["mia", "zoe"])

    def test_ties_break_alphabetically(self):
        self.assertEqual([e["name"] for e in top_scores(ENTRIES, 3)], ["mia", "zoe", "ben"])

    def test_n_larger_than_list(self):
        self.assertEqual(len(top_scores(ENTRIES, 10)), 4)
