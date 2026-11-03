
DEFAULTS = {"retries": 3, "timeout": 10, "hosts": ["a.example", "b.example"]}


def build_config(extra_hosts=None, **overrides):
    """Per-request config: defaults + overrides + extra hosts. Must not alter DEFAULTS."""
    config = DEFAULTS.copy()
    config.update(overrides)
    for host in extra_hosts or []:
        config["hosts"].append(host)
    return config
