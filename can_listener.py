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
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")     # silence TF C++ INFO/WARNING logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")    # silence the oneDNN startup notice

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)   # set before TF import so import-time warnings are filtered too

import warnings
warnings.filterwarnings("ignore")                      # silence pkg_resources / deprecation noise

import argparse
import json
import queue
import subprocess
import sys
import time
from collections import deque

# Print UTF-8 so the Russian detection line renders correctly on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub

# Silence TensorFlow's Python-side deprecation + GPU chatter (keep real errors).
tf.get_logger().setLevel(logging.ERROR)

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

# The on-detection action (play sound + launch game) lives in actions.py so the
# manual hotkey trigger can share it without importing TensorFlow.
from actions import launch_target


def load_models():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model not found at {MODEL_PATH}. Run:  python train.py")
    print("Loading YAMNet + beer classifier...")
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
        print(f"Listening for beer. threshold={threshold:.3f}, trigger on {k} of "
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
                    # Show BOTH the raw mic input level and P(can): if the mic bar
                    # never moves when you tap/talk, the mic is muted or it's the
                    # wrong --device. If mic moves but P(can) stays low, the sound
                    # just doesn't read as a can (try a real can crack near the mic).
                    level = float(np.sqrt(np.mean(np.square(
                        np.asarray(buffer, dtype=np.float32)))))
                    lbar = "#" * min(30, int(level * 300))
                    pbar = "#" * int(p * 30)
                    print(f"mic|{lbar:<30}|  P(can)={p:.3f} |{pbar:<30}|")
                    continue

                now = time.monotonic()
                triggered = sum(v >= threshold for v in hist) >= k
                if triggered and (now - last_trigger) >= args.cooldown:
                    last_trigger = now
                    hist.clear()
                    ts = time.strftime("%H:%M:%S")
                    print(f"[{ts}] Я СЛЫШУ ПИВО (P={p:.3f}) -> ЗАПУСКАЮ")
                    launch_target()
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
