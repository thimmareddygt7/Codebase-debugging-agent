
_STORE = {}


def make_key(user_id, resource):
    return f"{user_id}:{resource}"


def get(key):
    return _STORE.get(key)


def put(key, value):
    _STORE[key] = value


def clear():
    _STORE.clear()
