#!/usr/bin/env python3
"""
can_listener.py — Listen to the Mac microphone and launch a program when it
hears a metal can being opened (the "psst / crack / fizz" of a soda can).

How it works: each ~0.5s it feeds the latest 1s of audio to YAMNet, takes the
1024-d embedding, and runs a small trained classifier (models/can_model.keras)
that outputs P(can). A trigger requires the probability to clear the threshold on
2 of the last 3 windows (temporal smoothing) — this rejects one-off clicks (keys,
mouse) while still catching a real ~1s can event. See train.py for how the model
is built (YAMNet embeddings + augmented can events + ESC-50 hard negatives).

Usage:
    python can_listener.py                 # run the detector
    python can_listener.py --calibrate     # print live P(can) (tuning mode)
    python can_listener.py --analyze FILE   # score a recording (mp3/m4a/wav) and exit
    python can_listener.py --list-devices  # list microphones
    python can_listener.py --threshold 0.9 # more sensitive (default from meta.json)

Runs on macOS, Windows and Linux. While the detector is running you can also fire
the trigger by hand with a global hotkey (default Cmd+Shift+J on macOS, Ctrl+Alt+J
elsewhere) — handy for testing or launching on demand without opening a can.
Customise it with --hotkey and the launched program with --launch.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
import queue
import shutil
import subprocess
import sys
import time
from collections import deque

import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "can_model.keras")
META_PATH = os.path.join(HERE, "models", "meta.json")

# Audio / framing constants (must match train.py).
SAMPLE_RATE = 16000
WINDOW_SEC = 1.0
HOP_SEC = 0.48
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)
HOP_SAMPLES = int(HOP_SEC * SAMPLE_RATE)
BLOCK_SAMPLES = int(0.1 * SAMPLE_RATE)
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"

COOLDOWN_SEC = 10.0

IS_MAC = sys.platform == "darwin"
IS_WINDOWS = sys.platform.startswith("win")

# Default manual-trigger hotkey. macOS has a Cmd key; Windows/Linux use Ctrl+Alt
# (Ctrl+Shift+J is often a browser DevTools shortcut, so we avoid it).
DEFAULT_HOTKEY = "<cmd>+<shift>+j" if IS_MAC else "<ctrl>+<alt>+j"

VOICE_PATH = os.path.join(HERE, "assets", "voice.wav")


def default_launch_cmd():
    """Placeholder launch target per OS — override with --launch / $CAN_LAUNCH_CMD.

    This is the single "game launch path". Point it at your game, e.g.
        macOS:    open -a 'World of Tanks'
        Windows:  start "" "C:\\Games\\World_of_Tanks\\WorldOfTanks.exe"
        Linux:    /path/to/wot.sh    (or  xdg-open steam://rungameid/...)
    """
    if IS_MAC:
        return 'open -a "Google Chrome"'
    if IS_WINDOWS:
        return 'start "" chrome'
    return "xdg-open https://www.google.com"


def announce():
    """Play the announcement (non-blocking), cross-platform.

    Preferred path plays through sounddevice (already a dependency, works on
    macOS/Windows/Linux with no external audio binary). Falls back to a system
    player if that fails. Regenerate/replace the clip via make_voice.sh or by
    dropping your own assets/voice.wav.
    """
    if not os.path.exists(VOICE_PATH):
        return
    try:
        import scipy.io.wavfile as wavio
        sr, data = wavio.read(VOICE_PATH)
        sd.play(data, sr)               # non-blocking; own output stream
        return
    except Exception:
        pass
    if IS_WINDOWS:
        try:
            import winsound
            winsound.PlaySound(VOICE_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
            return
        except Exception:
            return
    player = (["afplay"] if IS_MAC else
              next(([p] for p in ("paplay", "aplay") if shutil.which(p)),
                   (["ffplay", "-nodisp", "-autoexit"] if shutil.which("ffplay") else None)))
    if player:
        subprocess.Popen(player + [VOICE_PATH],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def launch_target(launch_cmd=None):
    """Action performed on a trigger: announce, then launch the target program.

    launch_cmd (from --launch) wins; then $CAN_LAUNCH_CMD; then the per-OS default.
    Set CAN_NO_VOICE=1 to skip the announcement.
    """
    if not os.environ.get("CAN_NO_VOICE"):
        announce()
    cmd = launch_cmd or os.environ.get("CAN_LAUNCH_CMD") or default_launch_cmd()
    subprocess.run(cmd, shell=True, check=False)


def start_hotkey(combo, on_fire):
    """Run a global hotkey listener in a background thread (needs `pynput`).

    Returns the listener, or None if pynput isn't installed / can't grab keys
    (audio detection keeps working either way). macOS asks for Accessibility /
    Input Monitoring permission the first time — allow it for the hotkey to work.
    """
    try:
        from pynput import keyboard
    except ImportError:
        print("(hotkey off: `pip install pynput` to enable the manual trigger)")
        return None
    try:
        listener = keyboard.GlobalHotKeys({combo: on_fire})
    except Exception as e:
        print(f"(hotkey off: could not register {combo!r}: {e})")
        return None
    listener.daemon = True
    listener.start()
    print(f"Manual trigger hotkey: {combo}")
    return listener


def load_models():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model not found at {MODEL_PATH}. Run:  python train.py")
    print("Loading YAMNet + can classifier...")
    yamnet = hub.load(YAMNET_HANDLE)
    clf = tf.keras.models.load_model(MODEL_PATH)
    meta = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}
    # Warm up both models so the first real inference isn't slow.
    _s, emb, _sp = yamnet(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
    clf.predict(emb.numpy(), verbose=0)
    return yamnet, clf, meta


def window_prob(yamnet, clf, buffer):
    """P(can) for the current audio window (max over YAMNet frames in it)."""
    waveform = np.asarray(buffer, dtype=np.float32)
    _scores, emb, _spec = yamnet(waveform)
    probs = clf.predict(emb.numpy(), verbose=0).ravel()
    return float(probs.max()) if len(probs) else 0.0


def make_audio_queue(device):
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        q.put(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=BLOCK_SAMPLES, device=device, callback=callback,
    )
    return stream, q


# ---------------------------------------------------------------------------
# Offline analysis of a recorded file (empirical tuning / testing).
# ---------------------------------------------------------------------------
def analyze_file(yamnet, clf, meta, path):
    import shutil, tempfile
    import scipy.io.wavfile as wavio
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found; needed to decode audio files.")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE), tmp],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _sr, x = wavio.read(tmp); os.unlink(tmp)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = np.abs(x).max()
    if peak > 0:
        x = x / peak
    _scores, emb, _spec = yamnet(x)
    p = clf.predict(emb.numpy(), verbose=0).ravel()
    thr = meta.get("threshold", 0.9)
    fires = any((p[i:i+3] >= thr).sum() >= 2 for i in range(max(0, len(p) - 2))) \
        if len(p) >= 3 else (p >= thr).sum() >= 2
    print(f"\n=== ANALYZE: {path} ===")
    print(f"frames={len(p)}  max P(can)={p.max():.3f}  threshold={thr:.3f}")
    print(f"-> would {'FIRE ✅' if fires else 'NOT fire ❌'} (needs 2 of 3 frames >= threshold)")
    print("per-frame P(can): " + " ".join(f"{v:.2f}" for v in p))


def run(args, yamnet, clf, meta):
    threshold = args.threshold if args.threshold is not None else meta.get("threshold", 0.9)
    sm = meta.get("smoothing", {"k": 2, "n": 3})
    k, n = sm.get("k", 2), sm.get("n", 3)

    stream, q = make_audio_queue(args.device)
    buffer = deque(maxlen=WINDOW_SAMPLES)
    hist = deque(maxlen=n)
    samples_since_infer = 0
    last_trigger = 0.0

    mode = "CALIBRATE" if args.calibrate else "DETECT"
    print(f"\n=== {mode} MODE ===")
    if args.calibrate:
        print("Open a can in front of the mic and watch P(can).\n")
    else:
        # Manual hotkey trigger (skip in calibrate mode — nothing to launch there).
        def fire_manually():
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] MANUAL TRIGGER (hotkey) -> launching target")
            launch_target(args.launch)
        if args.hotkey.lower() != "off":
            start_hotkey(args.hotkey, fire_manually)
        print(f"Listening for a can. threshold={threshold:.3f}, trigger on {k} of "
              f"last {n} windows. Cooldown {args.cooldown}s. Ctrl+C to stop.\n")

    with stream:
        try:
            while True:
                block = q.get()
                buffer.extend(block)
                samples_since_infer += len(block)
                if len(buffer) < WINDOW_SAMPLES or samples_since_infer < HOP_SAMPLES:
                    continue
                samples_since_infer = 0

                p = window_prob(yamnet, clf, buffer)
                hist.append(p)

                if args.calibrate:
                    bar = "#" * int(p * 40)
                    print(f"P(can)={p:.3f} |{bar:<40}|")
                    continue

                now = time.monotonic()
                triggered = sum(v >= threshold for v in hist) >= k
                if triggered and (now - last_trigger) >= args.cooldown:
                    last_trigger = now
                    hist.clear()
                    ts = time.strftime("%H:%M:%S")
                    print(f"[{ts}] CAN DETECTED (P={p:.3f}) -> launching target")
                    launch_target(args.launch)
                elif args.verbose:
                    print(f"P(can)={p:.3f}", end="\r", flush=True)
        except KeyboardInterrupt:
            print("\nStopped.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibrate", action="store_true", help="print live P(can)")
    p.add_argument("--analyze", metavar="FILE", help="score a recorded clip and exit")
    p.add_argument("--list-devices", action="store_true", help="list audio input devices")
    p.add_argument("--device", type=int, default=None, help="input device index")
    p.add_argument("--threshold", type=float, default=None, help="override P(can) threshold")
    p.add_argument("--cooldown", type=float, default=COOLDOWN_SEC, help="re-arm delay after a trigger (s)")
    p.add_argument("--launch", metavar="CMD", default=None,
                   help="shell command to run on a trigger (e.g. \"open -a 'World of Tanks'\"); "
                        "overrides $CAN_LAUNCH_CMD and the built-in default")
    p.add_argument("--hotkey", metavar="COMBO", default=DEFAULT_HOTKEY,
                   help=f"global hotkey for a manual trigger (default {DEFAULT_HOTKEY}); \"off\" to disable")
    p.add_argument("--verbose", action="store_true", help="print rolling P(can) in detect mode")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    yamnet, clf, meta = load_models()

    if args.analyze:
        analyze_file(yamnet, clf, meta, args.analyze)
        return

    run(args, yamnet, clf, meta)


if __name__ == "__main__":
    main()
