
import unittest
from service.config import DEFAULTS, build_config


class ConfigTests(unittest.TestCase):
    def test_extra_hosts_are_added(self):
        cfg = build_config(extra_hosts=["c.example"])
        self.assertEqual(cfg["hosts"], ["a.example", "b.example", "c.example"])

    def test_requests_are_isolated(self):
        build_config(extra_hosts=["leak.example"])
        self.assertEqual(build_config()["hosts"], ["a.example", "b.example"])

    def test_defaults_untouched(self):
        build_config(extra_hosts=["x.example"])
        self.assertEqual(DEFAULTS["hosts"], ["a.example", "b.example"])

    def test_overrides(self):
        self.assertEqual(build_config(retries=5)["retries"], 5)
