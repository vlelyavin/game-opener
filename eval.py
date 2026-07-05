#!/usr/bin/env python3
"""
eval.py — Evaluate the trained can detector end-to-end (model + threshold + the
temporal trigger rule) on real clips, and print an operating-point table so the
decision threshold / smoothing rule can be chosen from evidence, not guesswork.

Positives : references/*.mp4 + ESC-50 can_opening.
Confusers : the ESC-50 classes most like a can crack (keyboard, mouse, clicks,
            liquid, glass...). We report the fraction of clips that would trigger.

Usage:  .venv/bin/python eval.py
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import csv
import glob
import subprocess
import tempfile

import numpy as np
import scipy.io.wavfile as wavio
import tensorflow as tf
import tensorflow_hub as hub

HERE = os.path.dirname(os.path.abspath(__file__))
ESC = os.path.join(HERE, "data", "esc50")
CONFUSERS = ["keyboard_typing", "mouse_click", "clock_tick", "glass_breaking",
             "pouring_water", "footsteps", "door_wood_knock", "drinking_sipping"]


def read(p):
    _sr, x = wavio.read(p); x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = np.abs(x).max()
    return x / peak if peak > 0 else x


def decode(p):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(["ffmpeg", "-y", "-i", p, "-ac", "1", "-ar", "16000", tmp],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    x = read(tmp); os.unlink(tmp); return x


def fires_kofn(p, thr, k, n):
    """Trigger rule: k of any n consecutive frames >= thr."""
    if len(p) < n:
        return int((p >= thr).sum() >= k)
    return int(any((p[i:i+n] >= thr).sum() >= k for i in range(len(p) - n + 1)))


def fires_peak(p, hi, msum):
    """Trigger: a strong peak (>= hi) whose 3-frame neighborhood sums to >= msum.

    A can is a peak with a decaying tail (e.g. 0.91, 0.75, 0.64 -> nbhd 2.3);
    an isolated keyboard/mouse click is a lone spike (0.95, 0, 0 -> nbhd 0.95).
    """
    p = np.asarray(p)
    for i in range(len(p)):
        if p[i] >= hi:
            nb = p[max(0, i-1):i+2].sum()
            if nb >= msum:
                return 1
    return 0


def main():
    ym = hub.load("https://tfhub.dev/google/yamnet/1")
    clf = tf.keras.models.load_model(os.path.join(HERE, "models", "can_model.keras"))

    def probs(x):
        _s, e, _sp = ym(x.astype(np.float32))
        return clf.predict(e.numpy(), verbose=0).ravel()

    meta = {r["filename"]: r["category"]
            for r in csv.DictReader(open(os.path.join(ESC, "meta", "esc50.csv")))}
    by_cat = {}
    for fn, c in meta.items():
        by_cat.setdefault(c, []).append(os.path.join(ESC, "audio", fn))

    print("Scoring positives (references + ESC-50 can_opening)...")
    pos = [probs(decode(f)) for f in sorted(glob.glob(os.path.join(HERE, "references", "*.mp4")))]
    pos += [probs(read(f)) for f in by_cat["can_opening"]]
    print("Scoring confusers...")
    conf = {c: [probs(read(f)) for f in by_cat[c]] for c in CONFUSERS}
    allneg = [(c, probs(read(f))) for c in by_cat if c != "can_opening" for f in by_cat[c][:6]]

    def report(label, fn):
        rec = np.mean([fn(p) for p in pos])
        cf = [np.mean([fn(p) for p in conf[c]]) for c in CONFUSERS]
        an = np.mean([fn(p) for _c, p in allneg])
        print(f"{label:16s} {rec:.2f}   | " + " ".join(f"{x:.2f} " for x in cf) + f"|  {an:.3f}")

    print("\nrule              recall | " + " ".join(f"{c[:5]}" for c in CONFUSERS) + " | allneg")
    for thr in [0.85, 0.90, 0.93]:
        report(f"2of3@{thr}", lambda p, t=thr: fires_kofn(p, t, 2, 3))
    print()
    for hi, ms in [(0.90, 1.6), (0.90, 2.0), (0.93, 1.6), (0.93, 2.0), (0.95, 1.8)]:
        report(f"peak{hi}/nb{ms}", lambda p, h=hi, m=ms: fires_peak(p, h, m))


if __name__ == "__main__":
    main()
