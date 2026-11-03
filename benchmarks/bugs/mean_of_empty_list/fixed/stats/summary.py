
def mean(values):
    """Average of numbers; an empty input has an average of 0.0."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
