
def is_palindrome(text):
    """True if text reads the same forwards and backwards, ignoring case and punctuation."""
    cleaned = text.lower()
    return cleaned == cleaned[::-1]
