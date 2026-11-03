
def parse_quantity(raw):
    """Parse a non-negative integer quantity; raise ValueError for anything invalid."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"invalid quantity: {raw!r}") from None
    if value < 0:
        raise ValueError("quantity must be non-negative")
    return value
