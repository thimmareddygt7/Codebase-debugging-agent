
def percent(part, whole):
    """Percentage (0-100) of `whole` that `part` represents."""
    if whole == 0:
        return 0.0
    return part / whole * 100
