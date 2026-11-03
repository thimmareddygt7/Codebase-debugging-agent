
import unittest
from app import cache
from app.service import get_profile


def loader(user_id, resource):
    return f"profile:{user_id}/{resource}"


class ServiceTests(unittest.TestCase):
    def setUp(self):
        cache.clear()

    def test_loads_and_caches(self):
        calls = []

        def counting_loader(u, r):
            calls.append((u, r))
            return loader(u, r)

        self.assertEqual(get_profile(1, "avatar", counting_loader), "profile:1/avatar")
        self.assertEqual(get_profile(1, "avatar", counting_loader), "profile:1/avatar")
        self.assertEqual(len(calls), 1)

    def test_different_users_never_share_entries(self):
        self.assertEqual(get_profile(1, "23x", loader), "profile:1/23x")
        self.assertEqual(get_profile(12, "3x", loader), "profile:12/3x")
