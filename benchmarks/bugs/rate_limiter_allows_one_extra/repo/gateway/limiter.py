
class RateLimiter:
    """Sliding-window limiter: at most `limit` requests per `window` seconds."""

    def __init__(self, limit, window):
        self.limit = limit
        self.window = window
        self.hits = []

    def allow(self, now):
        self.hits = [t for t in self.hits if now - t < self.window]
        if len(self.hits) > self.limit:
            return False
        self.hits.append(now)
        return True
