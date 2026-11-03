
def remove_expired(sessions, now):
    """Return only the sessions that have not expired yet."""
    for session in sessions:
        if session["expires_at"] <= now:
            sessions.remove(session)
    return sessions
