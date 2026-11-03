
def add_tag(tag, tags=None):
    """Add a tag to a tag list (creating a new list when none is given)."""
    if tags is None:
        tags = []
    tags.append(tag)
    return tags
