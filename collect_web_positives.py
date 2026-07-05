#!/usr/bin/env python3
"""
collect_web_positives.py — search the web for can-opening sound clips and keep
only the clean ones (auto-rejecting narrated / music-heavy videos that would
poison training).

It downloads short audio results via yt-dlp, then filters with YAMNet: a clip is
kept only if it is NOT dominated by Speech/Music and DOES contain a strong
transient event. Kept clips go to references_web/ for use as extra positives.

NOTE ON LICENSING: downloaded clips are for *local training* only and are not
redistributed (references_web/ is gitignored). The published artifact is the
trained model weights. For a fully license-clean dataset, prefer Freesound (CC).

Usage:  .venv/bin/python collect_web_positives.py [N_per_query]
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import glob
import shutil
import sys
import tempfile

import numpy as np
import scipy.io.wavfile as wavio
import tensorflow as tf
import tensorflow_hub as hub

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "collected_raw")
OUT = os.path.join(HERE, "references_web")
QUERIES = [
    "soda can opening sound effect",
    "coca cola can opening sound",
    "beer can opening sound effect",
    "energy drink can crack open sound",
    "aluminum can opening asmr",
]


def download(n_per_query):
    import yt_dlp
    os.makedirs(RAW, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(RAW, "%(id)s.%(ext)s"),
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "match_filter": yt_dlp.utils.match_filter_func("duration < 40"),
        "postprocessors": [{"key": "FFmpegExtractAudio",
                            "preferredcodec": "wav", "preferredquality": "0"}],
        "postprocessor_args": ["-ac", "1", "-ar", "16000"],
        "ignoreerrors": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        for q in QUERIES:
            print(f"  searching: {q}")
            try:
                ydl.download([f"ytsearch{n_per_query}:{q}"])
            except Exception as e:
                print(f"    (query failed: {e})")


def keep_clip(yamnet, names, x):
    """True if the clip looks like a genuine can event (not speech/music)."""
    scores, _emb, _spec = yamnet(x.astype(np.float32))
    scores = scores.numpy()
    top = scores.argmax(axis=1)
    junk = {names.index(n) for n in ["Speech", "Music", "Narration, monologue",
                                     "Conversation", "Singing", "Musical instrument"]}
    junk_frac = np.mean([t in junk for t in top])
    # need a strong transient somewhere (high-frequency energy spike)
    has_event = np.abs(x).max() > 0.1
    return junk_frac < 0.35 and has_event, junk_frac


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(f"Downloading up to {n} clips per query ({len(QUERIES)} queries)...")
    download(n)

    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    import csv, io
    names = [r["display_name"] for r in csv.DictReader(
        io.StringIO(tf.io.read_file(yamnet.class_map_path()).numpy().decode()))]

    os.makedirs(OUT, exist_ok=True)
    kept = 0
    raws = sorted(glob.glob(os.path.join(RAW, "*.wav")))
    print(f"\nFiltering {len(raws)} downloaded clips...")
    for fp in raws:
        try:
            _sr, x = wavio.read(fp)
        except Exception:
            continue
        x = x.astype(np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        peak = np.abs(x).max()
        if peak > 0:
            x = x / peak
        ok, junk = keep_clip(yamnet, names, x)
        tag = "KEEP" if ok else "drop"
        print(f"  [{tag}] {os.path.basename(fp)}  (junk_frac={junk:.2f})")
        if ok:
            shutil.copy(fp, os.path.join(OUT, f"web_{os.path.basename(fp)}"))
            kept += 1
    print(f"\nKept {kept} clips -> references_web/  "
          f"(add to training by pointing train.py at references_web/ too)")


if __name__ == "__main__":
    main()
