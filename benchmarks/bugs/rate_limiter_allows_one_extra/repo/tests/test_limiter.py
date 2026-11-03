
import unittest
from gateway.limiter import RateLimiter


class LimiterTests(unittest.TestCase):
    def test_blocks_after_limit(self):
        rl = RateLimiter(limit=3, window=10)
        self.assertEqual([rl.allow(t) for t in (0, 1, 2)], [True, True, True])
        self.assertFalse(rl.allow(3))

    def test_allows_again_after_window(self):
        rl = RateLimiter(limit=2, window=10)
        rl.allow(0)
        rl.allow(1)
        self.assertFalse(rl.allow(5))
        self.assertTrue(rl.allow(10))

    def test_denied_requests_do_not_count(self):
        rl = RateLimiter(limit=1, window=10)
        self.assertTrue(rl.allow(0))
        self.assertFalse(rl.allow(1))
        self.assertFalse(rl.allow(2))
        self.assertTrue(rl.allow(10))
