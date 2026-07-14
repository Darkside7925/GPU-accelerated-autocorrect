"""Check GitHub for a newer version.

Compares the local VERSION to the VERSION in app/version.py on the repo's main
branch (fetched raw, no auth, no API rate limit). Returns a small dict the tray
and dashboard use to tell you an update is available. Never raises; on any
network/TLS problem it just reports "not available" with an error string.
"""

from __future__ import annotations

import logging
import re
import time

log = logging.getLogger("updater")

REPO = "Darkside7925/GPU-accelerated-autocorrect"
RAW_VERSION_URL = (
    f"https://raw.githubusercontent.com/{REPO}/main/app/version.py"
)
REPO_URL = f"https://github.com/{REPO}"

_cache = {"ts": 0.0, "result": None}
_CACHE_TTL = 3600  # re-check at most hourly


def _vtuple(v: str):
    return tuple(int(x) for x in re.findall(r"\d+", v))


def _parse_version(text: str):
    m = re.search(r'VERSION\s*=\s*["\']([0-9.]+)["\']', text)
    return m.group(1) if m else None


def check_for_update(timeout: float = 8.0, force: bool = False) -> dict:
    """Return {available, current, latest, url[, error]}. Cached hourly."""
    from app.version import VERSION as current
    now = time.time()
    if not force and _cache["result"] and now - _cache["ts"] < _CACHE_TTL:
        return _cache["result"]
    result = {"available": False, "current": current, "latest": current, "url": REPO_URL}
    try:
        import truststore
        truststore.inject_into_ssl()   # this machine intercepts TLS; use the OS store
        import urllib.request
        text = urllib.request.urlopen(RAW_VERSION_URL, timeout=timeout).read().decode(
            "utf-8", "replace")
        latest = _parse_version(text)
        if latest:
            result["latest"] = latest
            result["available"] = _vtuple(latest) > _vtuple(current)
    except Exception as e:                # offline, TLS, rate, parse: never fatal
        result["error"] = f"{type(e).__name__}: {e}"
    _cache.update(ts=now, result=result)
    return result
