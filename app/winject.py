"""Atomic keystroke injection via a single Win32 SendInput call.

Why this matters for a fast typist: SendInput guarantees the whole batch of
events it is given is inserted into the input stream without being interspersed
with the user's own keyboard input (MSDN). So sending "8 backspaces + retype the
word" as ONE call means the user's next keystroke cannot land in the middle of a
correction and scramble it. pynput's Controller sends one key per SendInput
call, which leaves gaps the user's fast typing races into. This module sends the
whole correction in a single call.

Our injected events carry LLKHF_INJECTED, so the engine's keyboard hook detects
and ignores them (no feedback loop). Falls back to a pynput Controller if the
Win32 path is unavailable (non-Windows, or an unexpected error).
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

_ok = sys.platform == "win32"

# dwExtraInfo markers so the hook can tell OUR injections apart from any other
# injected input on the system (clipboard tools, AutoHotkey, ...)
CORRECTION_MAGIC = 0x53554D49   # "SUMI": a correction batch
REPLAY_MAGIC = 0x53554D52       # "SUMR": user keys held during a txn, replayed

if _ok:
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_UNICODE = 0x0004
    INPUT_KEYBOARD = 1
    VK_BACK = 0x08

    # ULONG_PTR is a pointer-SIZED unsigned integer VALUE (not a pointer): the
    # hook receives this exact value in KBDLLHOOKSTRUCT.dwExtraInfo
    ULONG_PTR = ctypes.c_size_t

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = (("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                    ("dwExtraInfo", ULONG_PTR))

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = (("dx", wintypes.LONG), ("dy", wintypes.LONG),
                    ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR))

    class _HARDWAREINPUT(ctypes.Structure):
        _fields_ = (("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                    ("wParamH", wintypes.WORD))

    class _INPUTunion(ctypes.Union):
        _fields_ = (("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT), ("hi", _HARDWAREINPUT))

    class _INPUT(ctypes.Structure):
        _fields_ = (("type", wintypes.DWORD), ("u", _INPUTunion))

    _SendInput = ctypes.windll.user32.SendInput
    _SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int)
    _SendInput.restype = wintypes.UINT

    def _key_event(vk: int, scan: int, flags: int, extra: int = CORRECTION_MAGIC) -> _INPUT:
        inp = _INPUT()
        inp.type = INPUT_KEYBOARD
        inp.u.ki = _KEYBDINPUT(vk, scan, flags, 0, extra)
        return inp


_fallback = None


def _get_fallback():
    global _fallback
    if _fallback is None:
        from pynput import keyboard
        _fallback = keyboard.Controller()
    return _fallback


def send(backspaces: int, text: str, magic: int = CORRECTION_MAGIC) -> None:
    """Send `backspaces` backspaces then type `text`, atomically in one call.
    Every event is stamped with `magic` in dwExtraInfo so the hook can identify
    the batch."""
    if not _ok:
        _send_pynput(backspaces, text)
        return
    events = []
    for _ in range(max(0, backspaces)):
        events.append(_key_event(VK_BACK, 0, 0, magic))
        events.append(_key_event(VK_BACK, 0, KEYEVENTF_KEYUP, magic))
    for ch in text:
        code = ord(ch)
        events.append(_key_event(0, code, KEYEVENTF_UNICODE, magic))
        events.append(_key_event(0, code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, magic))
    if not events:
        return
    n = len(events)
    arr = (_INPUT * n)(*events)
    try:
        sent = _SendInput(n, arr, ctypes.sizeof(_INPUT))
        if sent != n:                       # partial send, fall back for the rest
            _send_pynput(0, text if backspaces == 0 else "")
    except Exception:
        _send_pynput(backspaces, text)


def send_unicode_chars(chars: str, magic: int = REPLAY_MAGIC) -> None:
    """Replay held user characters as unicode down+up pairs stamped `magic`."""
    send(0, chars, magic=magic)


def vk_to_char(vk: int, scan: int):
    """Resolve a keydown to exactly one printable character using the current
    shift/caps state, without disturbing dead-key state (wFlags bit 2). Returns
    None when the key does not map cleanly to one printable char; the caller
    must then let the key through instead of holding it."""
    if not _ok:
        return None
    try:
        user32 = ctypes.windll.user32
        state = (ctypes.c_ubyte * 256)()
        if user32.GetAsyncKeyState(0x10) & 0x8000:   # SHIFT physically down
            state[0x10] = 0x80
        if user32.GetKeyState(0x14) & 1:             # CAPS LOCK toggled
            state[0x14] = 0x01
        buf = ctypes.create_unicode_buffer(8)
        layout = user32.GetKeyboardLayout(0)
        n = user32.ToUnicodeEx(vk, scan, state, buf, 8, 0x4, layout)
        if n == 1 and buf[0].isprintable():
            return buf[0]
    except Exception:
        pass
    return None


def _send_pynput(backspaces: int, text: str) -> None:
    from pynput.keyboard import Key
    kb = _get_fallback()
    for _ in range(max(0, backspaces)):
        kb.tap(Key.backspace)
    if text:
        kb.type(text)
