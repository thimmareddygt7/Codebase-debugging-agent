"""
Verify that the configured Groq models actually exist for your API key.

    python -m scripts.check_models

Model catalogs change (models get deprecated); a wrong default makes every request fail.
"""
import sys

from src.config import settings


def main() -> int:
    if not settings.GROQ_API_KEY:
        print("GROQ_API_KEY is not set.", file=sys.stderr)
        return 2
    from groq import Groq

    available = {m.id for m in Groq(api_key=settings.GROQ_API_KEY).models.list().data}
    ok = True
    for label, model in (("GROQ_MODEL", settings.GROQ_MODEL), ("FALLBACK_MODEL", settings.FALLBACK_MODEL)):
        found = model in available
        ok &= found
        print(f"{'OK     ' if found else 'MISSING'} {label} = {model}")
    if not ok:
        print("\nAvailable models:\n  " + "\n  ".join(sorted(available)))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
