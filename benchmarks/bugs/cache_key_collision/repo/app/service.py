
from app import cache


def get_profile(user_id, resource, loader):
    """Return the cached profile for (user, resource), loading it on a miss."""
    key = cache.make_key(user_id, resource)
    value = cache.get(key)
    if value is None:
        value = loader(user_id, resource)
        cache.put(key, value)
    return value
