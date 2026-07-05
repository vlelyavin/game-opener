#!/usr/bin/env python3
"""
actions.py — the on-trigger action: play the sound, then launch the game.

Shared by can_listener.py (fires it on can detection) and hotkey_trigger.py
(fires it on a key press). Deliberately has NO TensorFlow import, so the hotkey
path starts instantly.

Config via environment variables:
    CAN_SOUND        path to the sound file (default: assets/tank.wav)
    CAN_NO_VOICE=1   skip the sound
    CAN_LAUNCH_PATH  Windows: exe to launch (default: World of Tanks)
    CAN_LAUNCH_CMD   run this shell command instead of the per-OS default
"""

import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM = platform.system()   # 'Windows', 'Darwin' (macOS), 'Linux'

# Sound played on trigger. Override with CAN_SOUND=/path/to/file (wav/ogg/...).
SOUND_PATH = os.environ.get("CAN_SOUND") or os.path.join(HERE, "assets", "tank.wav")

# Default launch target per OS. On Windows this starts World of Tanks directly
# (WGC comes up in the background for auth/patching). Override with CAN_LAUNCH_PATH.
WOT_EXE = r"C:\Games\World_of_Tanks_EU\WorldOfTanks.exe"


def play_sound(path=SOUND_PATH, block=False):
    """Play a sound file (wav/ogg/...) cross-platform.

    Uses sounddevice (already a dependency). Decodes via soundfile if available
    (any format), else falls back to scipy for WAV. Non-blocking by default;
    set block=True to wait until the sound finishes.
    """
    if not os.path.exists(path):
        print(f"[sound] file not found: {path}", file=sys.stderr)
        return
    try:
        import sounddevice as sd
        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype="float32")
        except ImportError:
            import scipy.io.wavfile as wavio
            sr, data = wavio.read(path)
        sd.play(data, sr)            # plays on the default output device
        if block:
            sd.wait()
    except Exception as e:
        print(f"[sound] could not play {path}: {e}", file=sys.stderr)


def launch_game():
    """Launch the game (no sound). Resolution order:

      1. CAN_LAUNCH_CMD  — run this shell command verbatim (any OS).
      2. Windows  — open CAN_LAUNCH_PATH (default: World of Tanks exe).
      3. macOS    — open -a "Google Chrome" (original behavior).
      4. Linux    — xdg-open the path if set, else no-op.
    """
    cmd = os.environ.get("CAN_LAUNCH_CMD")
    if cmd:
        subprocess.Popen(cmd, shell=True)
        return

    if SYSTEM == "Windows":
        target = os.environ.get("CAN_LAUNCH_PATH", WOT_EXE)
        if os.path.exists(target):
            os.startfile(target)     # noqa: T003 (Windows-only, launches detached)
        else:
            print(f"[launch] target not found: {target}\n"
                  f"         set CAN_LAUNCH_PATH or CAN_LAUNCH_CMD.", file=sys.stderr)
    elif SYSTEM == "Darwin":
        target = os.environ.get("CAN_LAUNCH_PATH")
        subprocess.run(["open", "-a", target or "Google Chrome"], check=False)
    else:  # Linux / other
        target = os.environ.get("CAN_LAUNCH_PATH")
        if target:
            subprocess.Popen(["xdg-open", target])


def launch_target():
    """The full on-trigger action: play the sound, then launch the game."""
    if not os.environ.get("CAN_NO_VOICE"):
        play_sound()
    launch_game()
