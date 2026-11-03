
def top_scores(entries, n):
    """Top n entries: highest score first; ties broken alphabetically by name."""
    ranked = sorted(entries, key=lambda e: (-e["score"], e["name"]))
    return ranked[:n]
