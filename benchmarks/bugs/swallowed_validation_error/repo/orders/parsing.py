
def parse_quantity(raw):
    """Parse a non-negative integer quantity; raise ValueError for anything invalid."""
    try:
        value = int(raw)
    except Exception:
        return 0
    if value < 0:
        raise ValueError("quantity must be non-negative")
    return value
