
def remove_expired(sessions, now):
    """Return only the sessions that have not expired yet."""
    return [s for s in sessions if s["expires_at"] > now]
