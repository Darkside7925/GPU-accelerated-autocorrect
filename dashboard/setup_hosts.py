"""One-time setup: point grammer.local at this machine so the dashboard opens
at http://grammer.local.

Run this ONCE, as Administrator (editing the hosts file needs elevation):

    Right-click PowerShell -> Run as administrator, then:
    python -m dashboard.setup_hosts

To undo, run with --remove. The dashboard also always works at
http://127.0.0.1 (or :8080), with no hosts edit at all.

We use grammer.local, not grammer.com, on purpose: grammer.com is a real site
and browsers force it to https (HSTS), which would break a plain-http local
server. The .local name is reserved for local use and never HSTS-blocked.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

HOSTS = Path(r"C:\Windows\System32\drivers\etc\hosts")
HOSTNAME = "grammer.local"
LINE = f"127.0.0.1 {HOSTNAME}"
MARK = "  # sumizome dashboard"


def _is_admin() -> bool:
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False


def add() -> None:
    text = HOSTS.read_text(encoding="utf-8")
    if HOSTNAME in text:
        print(f"{HOSTNAME} already present in hosts, nothing to do.")
        return
    with HOSTS.open("a", encoding="utf-8") as fh:
        fh.write(f"\n{LINE}{MARK}\n")
    print(f"Added: {LINE}")
    print("Open the dashboard at http://grammer.local (start the app first).")


def remove() -> None:
    lines = HOSTS.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [ln for ln in lines if HOSTNAME not in ln]
    HOSTS.write_text("".join(kept), encoding="utf-8")
    print(f"Removed {HOSTNAME} from hosts. Run 'ipconfig /flushdns' to clear the cache.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Map grammer.local to 127.0.0.1")
    ap.add_argument("--remove", action="store_true", help="undo the hosts entry")
    args = ap.parse_args()
    if not _is_admin():
        print("This needs Administrator rights. Re-run from an elevated shell:")
        print("    python -m dashboard.setup_hosts" + (" --remove" if args.remove else ""))
        return 1
    try:
        remove() if args.remove else add()
    except OSError as e:
        print(f"Could not edit the hosts file: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
