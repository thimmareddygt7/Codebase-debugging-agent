
def is_palindrome(text):
    """True if text reads the same forwards and backwards, ignoring case and punctuation."""
    cleaned = "".join(ch.lower() for ch in text if ch.isalnum())
    return cleaned == cleaned[::-1]
