
def parse_version(version):
    """'1.10.0' -> comparable version key."""
    return tuple(int(part) for part in version.split("."))


def is_newer(candidate, current):
    return parse_version(candidate) > parse_version(current)
