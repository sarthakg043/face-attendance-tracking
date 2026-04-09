from functools import wraps
from fastapi import Request
from fastapi.responses import RedirectResponse
from config import ADMIN_USERNAME, ADMIN_PASSWORD
import hmac


def verify_admin(username: str, password: str) -> bool:
    return (
        hmac.compare_digest(username, ADMIN_USERNAME)
        and hmac.compare_digest(password, ADMIN_PASSWORD)
    )


def require_admin(func):
    """Decorator for route handlers that need admin access."""
    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        if not request.session.get("admin"):
            return RedirectResponse("/admin/login", status_code=302)
        return await func(request, *args, **kwargs)
    return wrapper
