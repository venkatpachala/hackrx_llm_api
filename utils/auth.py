import os


def verify_token(token: str) -> bool:
    """Check if the provided token matches the API token."""
    expected = os.getenv("API_TOKEN")
    if expected is None:
        return False
    return token == expected
