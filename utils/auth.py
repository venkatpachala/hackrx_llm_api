import os

def verify_bearer_token(token: str) -> bool:
    expected = f"Bearer {os.getenv('HACKRX_AUTH_TOKEN')}"
    return token == expected
