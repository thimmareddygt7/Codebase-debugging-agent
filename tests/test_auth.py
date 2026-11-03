"""Pure-logic tests for API-key helpers (no FastAPI needed)."""
from src.auth import key_is_valid, parse_api_keys

GOOD_A = "a" * 32
GOOD_B = "b" * 32


def test_parse_api_keys_splits_and_trims():
    assert parse_api_keys(f" {GOOD_A} , {GOOD_B} ,") == [GOOD_A, GOOD_B]


def test_parse_api_keys_drops_empty_and_weak_keys():
    assert parse_api_keys("") == []
    assert parse_api_keys(None) == []
    assert parse_api_keys("short,tiny") == []
    assert parse_api_keys(f"short,{GOOD_A}") == [GOOD_A]


def test_key_is_valid_accepts_any_configured_key():
    keys = [GOOD_A, GOOD_B]
    assert key_is_valid(GOOD_A, keys)
    assert key_is_valid(GOOD_B, keys)


def test_key_is_valid_rejects_missing_wrong_and_prefix_keys():
    keys = [GOOD_A]
    assert not key_is_valid(None, keys)
    assert not key_is_valid("", keys)
    assert not key_is_valid("wrong" * 8, keys)
    assert not key_is_valid(GOOD_A[:-1], keys)      # prefix of a real key
    assert not key_is_valid(GOOD_A + "x", keys)     # real key + suffix
    assert not key_is_valid(GOOD_A, [])             # nothing configured
