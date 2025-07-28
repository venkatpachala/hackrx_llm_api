import os



def verify_token(token: str) -> bool:
    """Check if the provided token matches the API token."""
    expected = os.getenv("API_TOKEN")
    if expected is None:
        return False
    return token == expected


def get_api_token() -> str:
    """Return the API token from environment variables."""
    token = os.getenv("API_TOKEN")
    if not token:
        raise ValueError("API_TOKEN not found in environment variables.")
    return token
