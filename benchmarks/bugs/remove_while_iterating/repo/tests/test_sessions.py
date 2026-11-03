
import unittest
from auth.sessions import remove_expired


def _s(name, expires_at):
    return {"name": name, "expires_at": expires_at}


class SessionTests(unittest.TestCase):
    def test_keeps_live_sessions(self):
        live = [_s("a", 100), _s("b", 200)]
        self.assertEqual(remove_expired(live, now=50), [_s("a", 100), _s("b", 200)])

    def test_removes_consecutive_expired_sessions(self):
        sessions = [_s("a", 1), _s("b", 2), _s("c", 3), _s("d", 100)]
        self.assertEqual(remove_expired(sessions, now=10), [_s("d", 100)])

    def test_expiry_boundary_is_expired(self):
        self.assertEqual(remove_expired([_s("a", 10)], now=10), [])
