from datetime import datetime
from urllib.parse import urlparse

import requests
from flask import g, request, session
from flask_login import LoginManager, UserMixin, current_user, login_user


_user_registry = {}


class User(UserMixin):
    def __init__(self, user_id, email=None, slug=None, full_name=None):
        self.id = user_id
        self.email = email
        self.slug = slug
        self.full_name = full_name


def init_auth(app, *, directus_base_url, refresh_cookie_name, safe_user_slug, ensure_user_folders):
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = "login"
    login_manager.login_message = None

    @login_manager.user_loader
    def load_user(user_id):
        return _user_registry.get(str(user_id))

    def auth_cookie_domain():
        hostname = urlparse(directus_base_url or "").hostname or ""
        if hostname:
            return f".{hostname}"
        return None

    def auth_cookie_secure():
        return request.is_secure or request.headers.get("X-Forwarded-Proto") == "https"

    def directus_payload(resp):
        try:
            data = resp.json()
        except Exception:
            return {}
        return data.get("data") or data or {}

    def register_user(email, slug=None, full_name=None):
        if not email:
            return None

        slug = slug or safe_user_slug(email)
        user = _user_registry.get(str(slug))
        if user is None:
            user = User(
                user_id=slug,
                email=email,
                slug=slug,
                full_name=full_name or f"@{slug}",
            )
            _user_registry[str(user.id)] = user

        ensure_user_folders(slug)
        session["user_email"] = email
        session["user_slug"] = slug
        login_user(user)
        return user

    def store_shared_auth(tokens):
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")

        expires_ms = tokens.get("expires")
        try:
            expires_sec = float(expires_ms) / 1000.0 if expires_ms else 900.0
        except (TypeError, ValueError):
            expires_sec = 900.0

        if access_token:
            session["access_token"] = access_token
            session["access_expires_at"] = datetime.now().timestamp() + expires_sec
            g.access_token = access_token

        if refresh_token:
            g.new_refresh_token = refresh_token

    def cached_access_token():
        token = session.get("access_token")
        expires_at = session.get("access_expires_at")
        if not token or not expires_at:
            return None
        if datetime.now().timestamp() >= float(expires_at) - 30:
            return None
        return token

    def fetch_current_identity(access_token):
        try:
            resp = requests.get(
                f"{directus_base_url}/users/me",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=8,
            )
        except Exception as exc:
            app.logger.warning(f"Directus users/me failed: {exc}")
            return None

        if not resp.ok:
            return None

        data = resp.json().get("data", {})
        email = data.get("email") or session.get("user_email")
        slug = safe_user_slug(email) if email else session.get("user_slug")
        if email:
            register_user(email, slug=slug, full_name=f"@{slug}")
        return email

    @app.before_request
    def shared_cookie_auth_gate():
        path = request.path
        if path.startswith("/static/"):
            return None

        g.new_refresh_token = None
        g.clear_refresh_cookie = False

        if current_user.is_authenticated:
            return None

        token = cached_access_token()
        if token:
            g.access_token = token
            email = session.get("user_email") or fetch_current_identity(token)
            if email:
                register_user(email, slug=session.get("user_slug") or safe_user_slug(email))
            return None

        refresh_token = request.cookies.get(refresh_cookie_name)
        if not refresh_token:
            return None

        try:
            resp = requests.post(
                f"{directus_base_url}/auth/refresh",
                json={"refresh_token": refresh_token},
                timeout=8,
            )
        except Exception as exc:
            app.logger.warning(f"Directus refresh failed: {exc}")
            g.clear_refresh_cookie = True
            session.clear()
            return None

        if not resp.ok:
            g.clear_refresh_cookie = True
            session.clear()
            return None

        tokens = directus_payload(resp)
        store_shared_auth(tokens)

        email = fetch_current_identity(session.get("access_token"))
        if email:
            register_user(email, slug=session.get("user_slug") or safe_user_slug(email))

        return None

    @app.after_request
    def shared_cookie_after_request(resp):
        domain = auth_cookie_domain()

        if getattr(g, "clear_refresh_cookie", False):
            resp.delete_cookie(refresh_cookie_name, path="/", domain=domain)
            return resp

        new_refresh_token = getattr(g, "new_refresh_token", None)
        if new_refresh_token:
            resp.set_cookie(
                refresh_cookie_name,
                new_refresh_token,
                httponly=True,
                secure=auth_cookie_secure(),
                samesite="Lax",
                path="/",
                domain=domain,
            )

        return resp

    return {
        "login_manager": login_manager,
        "register_user": register_user,
        "store_shared_auth": store_shared_auth,
        "cached_access_token": cached_access_token,
        "fetch_current_identity": fetch_current_identity,
        "auth_cookie_domain": auth_cookie_domain,
        "auth_cookie_secure": auth_cookie_secure,
        "directus_payload": directus_payload,
        "user_registry": _user_registry,
    }