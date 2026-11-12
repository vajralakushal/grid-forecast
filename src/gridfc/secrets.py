"""Read the EIA key from the environment or .env. Never print it."""
import os

from dotenv import load_dotenv

from .config import ROOT


def get_eia_key() -> str:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("EIA_API_KEY", "").strip()
    if not key or key == "paste-your-key-here":
        raise SystemExit(
            "EIA_API_KEY is not set. Run: cp .env.example .env  then paste your key into .env"
        )
    return key


def redact(text: str, key: str) -> str:
    """Remove the key from any message before it is printed or logged."""
    return text.replace(key, "***") if key else text
