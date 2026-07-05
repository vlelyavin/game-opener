#!/usr/bin/env python3
"""
hotkey_trigger.py — fire the can-opener action (play tank sound + launch World of
Tanks) on a key press instead of on a real can. Handy for filming a clip: open
your beer on camera, tap the hotkey, edit the two together.

The terminal output is IDENTICAL to the real detector (can_listener.py): same
startup banner, same "[time] Я СЛЫШУ ПИВО (P=0.9xx) -> ЗАПУСКАЮ" line —
so on camera it looks exactly like it heard the can. It just starts instantly
(no TensorFlow, no microphone) and fires on a key instead of on sound.

Usage (Windows):
    .venv\\Scripts\\python.exe hotkey_trigger.py                 # global hotkey ctrl+alt+t
    .venv\\Scripts\\python.exe hotkey_trigger.py --hotkey f8      # use F8 instead
    .venv\\Scripts\\python.exe hotkey_trigger.py --enter          # trigger on Enter in this window

Same env vars as the detector work here too:
    $env:CAN_NO_VOICE="1"        # launch the game with no sound
    $env:CAN_LAUNCH_CMD="..."    # run a different launch command (e.g. open Game Center)
    $env:CAN_SOUND="C:\\x.wav"    # different sound
"""

import argparse
import json
import os
import random
import sys
import time

# Print UTF-8 so the Russian detection line renders correctly on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from actions import launch_target

HERE = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(HERE, "models", "meta.json")

# Match can_listener.py's defaults so the banner reads one-to-one with the real
# detector. Cooldown text mirrors the detector's default; presses aren't actually
# rate-limited here, so retakes fire every time.
COOLDOWN_TEXT = "10.0"


def print_startup_banner():
    """Reproduce can_listener.py's exact startup output."""
    meta = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}
    threshold = meta.get("threshold", 0.9)
    sm = meta.get("smoothing", {"k": 2, "n": 3})
    k, n = sm.get("k", 2), sm.get("n", 3)

    print("Loading YAMNet + beer classifier...")
    time.sleep(1.5)   # mimic the real model-load pause so it looks authentic on camera
    print("\n=== DETECT MODE ===")
    print(f"Listening for beer. threshold={threshold:.3f}, trigger on {k} of "
          f"last {n} windows. Cooldown {COOLDOWN_TEXT}s. Ctrl+C to stop.\n")


def trigger():
    """Print the exact detection line the detector prints, then run the action."""
    ts = time.strftime("%H:%M:%S")
    p = random.uniform(0.91, 0.99)   # a realistic high P(can), like a real detection
    print(f"[{ts}] Я СЛЫШУ ПИВО (P={p:.3f}) -> ЗАПУСКАЮ")
    launch_target()


def run_enter_loop():
    """Fallback trigger: press Enter in this console window to fire."""
    try:
        while True:
            input()
            trigger()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")


def run_global_hotkey(hotkey):
    """Global hotkey via the `keyboard` package — works even when this window
    isn't focused (trigger from a numpad off-camera). Falls back to Enter if the
    package isn't installed."""
    try:
        import keyboard
    except ImportError:
        print("(`keyboard` not installed — press ENTER in this window to trigger instead.)")
        run_enter_loop()
        return

    keyboard.add_hotkey(hotkey, trigger)
    try:
        keyboard.wait()
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hotkey", default="ctrl+alt+t",
                    help="global hotkey combo (default: ctrl+alt+t)")
    ap.add_argument("--enter", action="store_true",
                    help="trigger on ENTER in this window instead of a global hotkey")
    args = ap.parse_args()

    # Output is intentionally identical to can_listener.py — no extra "manual mode"
    # lines — so a recording of this terminal is indistinguishable from a real
    # detection. The chosen hotkey is silent by design; you set it, you know it.
    print_startup_banner()

    if args.enter:
        run_enter_loop()
    else:
        run_global_hotkey(args.hotkey)


if __name__ == "__main__":
    main()
