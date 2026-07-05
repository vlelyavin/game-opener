#!/usr/bin/env python3
"""
train.py — Train a can-opening sound detector on top of YAMNet embeddings.

Why this instead of hand-picking YAMNet output classes: a can-open sound has no
stable AudioSet label — it scatters across Glass / Chink / Coin / Keys / Crack /
Hiss depending on the can and mic, and those same labels fire for keys, coins and
dishes. So we drop the labels and learn from YAMNet's 1024-d *embedding* instead.

Data:
  positives  = can-open events, heavily augmented (background-mixed at varied SNR,
               gain/speed jitter). Sources:
                 - references/*.mp4          (your clips)
                 - ESC-50 'can_opening' class (40 real recordings)
  negatives  = the other 49 ESC-50 classes — real hard confusers:
               pouring_water, crackling_fire, glass_breaking, clock_tick,
               keyboard_typing, mouse_click, drinking_sipping, ...
  features   = YAMNet embeddings (1024-d per ~0.5s frame)
  model      = small MLP -> P(can)

Honest evaluation: grouped K-fold CV by SOURCE clip (an event never appears in
both train and test), so reported recall isn't inflated by event leakage.

Outputs:
  models/can_model.keras
  models/meta.json   (decision threshold, smoothing policy, frame timing)

Usage:
  .venv/bin/python train.py
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import csv
import glob
import json
import subprocess
import tempfile
import urllib.request
import zipfile

import numpy as np
import scipy.io.wavfile as wavio
import scipy.signal as sps
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import GroupKFold

SR = 16000
WIN = 15360                      # YAMNet analysis window (0.96s) in samples
PAD = 16000                      # feed 1.0s so we reliably get >=1 embedding frame
MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIRS = [os.path.join(HERE, "references"),        # your curated clips
            os.path.join(HERE, "references_web")]     # collect_web_positives.py output
ESC_DIR = os.path.join(HERE, "data", "esc50")
MODEL_DIR = os.path.join(HERE, "models")
REF_EXTS = ("*.mp4", "*.m4a", "*.mp3", "*.wav", "*.webm", "*.ogg", "*.opus")

RNG = np.random.default_rng(1234)   # fixed seed: reproducible dataset

TARGET_POSITIVES = 3000          # augmented positives spread across all sources
NEG_FRAMES_PER_CLIP = 3          # embeddings sampled per ESC-50 negative clip


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def decode(path):
    """Decode any media file to 16 kHz mono float32 in [-1, 1] via ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(SR), tmp],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _sr, x = wavio.read(tmp)
    os.unlink(tmp)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = np.abs(x).max()
    return x / peak if peak > 0 else x


def read_wav(path):
    _sr, x = wavio.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = np.abs(x).max()
    return x / peak if peak > 0 else x


def extract_event(x):
    """Return the WIN-sample window with the most acoustic energy (the can event)."""
    # High-frequency-weighted energy: a can crack/hiss is broadband/high-freq, which
    # separates it from low-freq rumble/speech.
    hp = sps.butter(4, 1500, "hp", fs=SR, output="sos")
    e = sps.sosfilt(hp, x) ** 2
    k = np.ones(int(0.05 * SR)) / int(0.05 * SR)     # ~50ms smoothing
    env = np.convolve(e, k, mode="same")
    peak = int(np.argmax(env))
    start = max(0, min(len(x) - WIN, peak - int(0.25 * WIN)))  # peak ~25% into window
    core = x[start:start + WIN]
    if len(core) < WIN:
        core = np.pad(core, (0, WIN - len(core)))
    return core


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


def apply_channel(x):
    """Simulate a random capture channel: mic/speaker band-limiting + room reverb.

    This is what makes the model robust to *how* the sound is captured — different
    mics, distances and rooms (and the speaker->mic case) all change the spectrum
    and add reverb. Training through randomized channels closes that gap.
    """
    y = x.astype(np.float32)
    lo, hi = RNG.uniform(40, 250), RNG.uniform(3500, 7800)     # band-limit
    y = sps.sosfilt(sps.butter(2, [lo, hi], btype="band", fs=SR, output="sos"), y)
    if RNG.random() < 0.6:                                     # room reverb
        tau = RNG.uniform(0.02, 0.12)
        L = int(SR * RNG.uniform(0.05, 0.25))
        ir = np.exp(-np.arange(L) / SR / tau) * RNG.standard_normal(L)
        ir[0] = 1.0
        y = np.convolve(y, ir)[:len(x)]
    m = np.abs(y).max()
    return (y / m).astype(np.float32) if m > 0 else y.astype(np.float32)


def augment(core, bg_pool):
    """One augmented positive: speed/gain jitter + background mix at random SNR."""
    y = core.copy()
    s = RNG.uniform(0.9, 1.1)                          # speed/pitch jitter
    y = sps.resample(y, int(len(y) / s))
    y = np.pad(y, (0, WIN))[:WIN] if len(y) < WIN else y[:WIN]
    y = y * RNG.uniform(0.3, 1.0)                      # gain
    if RNG.random() < 0.7:                             # background mix
        bg = bg_pool[RNG.integers(len(bg_pool))]
        off = RNG.integers(0, max(1, len(bg) - WIN))
        bg = bg[off:off + WIN]
        bg = np.pad(bg, (0, WIN - len(bg))) if len(bg) < WIN else bg
        # keep the can clearly dominant (SNR 10-30 dB): low-SNR mixing taught the
        # model that background clicks/typing look like a can.
        snr = RNG.uniform(10, 30)
        y = y + bg * (rms(y) / (10 ** (snr / 20)) / rms(bg))
    if RNG.random() < 0.85:                            # random capture channel
        y = apply_channel(y)
    m = np.abs(y).max()
    if m > 1:
        y = y / m
    return np.pad(y, (0, PAD - WIN)).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset build
# ---------------------------------------------------------------------------
def ensure_esc50():
    if os.path.isdir(os.path.join(ESC_DIR, "audio")):
        return
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    print("Downloading ESC-50 (~600MB)...")
    zpath = os.path.join(HERE, "data", "esc50.zip")
    urllib.request.urlretrieve(
        "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip", zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(os.path.join(HERE, "data"))
    os.rename(os.path.join(HERE, "data", "ESC-50-master"), ESC_DIR)
    os.remove(zpath)


# Transient/click/liquid confusers that look most like a can crack — the model
# needs to see MANY of these as negatives to learn the boundary.
HARD_NEG = {
    "keyboard_typing", "mouse_click", "clock_tick", "clock_alarm",
    "glass_breaking", "pouring_water", "drinking_sipping", "door_wood_knock",
    "footsteps", "water_drops", "crackling_fire", "church_bells", "clapping",
}


def load_esc50():
    """Return (neg_waves, neg_cats, can_waves): ESC-50 split by role."""
    meta = {}
    with open(os.path.join(ESC_DIR, "meta", "esc50.csv")) as f:
        for row in csv.DictReader(f):
            meta[row["filename"]] = row["category"]
    neg, neg_cats, can = [], [], []
    for fp in sorted(glob.glob(os.path.join(ESC_DIR, "audio", "*.wav"))):
        cat = meta.get(os.path.basename(fp), "")
        if cat == "can_opening":
            can.append(read_wav(fp))
        else:
            neg.append(read_wav(fp)); neg_cats.append(cat)
    return neg, neg_cats, can


def embed(model, waveform):
    _scores, emb, _spec = model(waveform.astype(np.float32))
    return emb.numpy()


def build_dataset(model):
    print("Loading ESC-50...")
    neg_waves, neg_cats, can_waves = load_esc50()
    print(f"  {len(neg_waves)} negative clips, {len(can_waves)} ESC-50 can_opening clips")

    ref_files = sorted(f for d in REF_DIRS for ext in REF_EXTS
                       for f in glob.glob(os.path.join(d, ext)))
    ref_waves = [decode(fp) for fp in ref_files]
    print(f"  {len(ref_waves)} reference clips (incl. references_web/)")

    # Positive sources = reference clips + ESC-50 can clips. Each is one event.
    sources = [extract_event(w) for w in ref_waves + can_waves]
    aug_per = max(20, TARGET_POSITIVES // len(sources))
    print(f"Augmenting {len(sources)} positive sources x{aug_per} "
          f"(background pool = {len(neg_waves)} negatives)...")

    pos_emb, pos_src = [], []
    for si, core in enumerate(sources):
        for _ in range(aug_per):
            pos_emb.append(embed(model, augment(core, neg_waves))[0])
            pos_src.append(si)
    pos_emb, pos_src = np.array(pos_emb), np.array(pos_src)

    print("Extracting negative embeddings (hard confusers oversampled x channels)...")
    neg_emb = []
    for w, cat in zip(neg_waves, neg_cats):
        if cat in HARD_NEG:
            # hard confusers (keyboard/mouse/clicks/liquid): take ALL frames from
            # the original AND 2 channel-augmented copies, so channel-warped clicks
            # are firmly labeled not-can.
            variants = [w, apply_channel(w), apply_channel(w)]
            for v in variants:
                e = embed(model, v)
                if len(e):
                    neg_emb.extend(e)
        else:
            # half of easy negatives also go through a random channel, so the model
            # can't use "band-limited/reverberant" as a shortcut for "can".
            wc = apply_channel(w) if RNG.random() < 0.5 else w
            e = embed(model, wc)
            if len(e):
                idx = RNG.choice(len(e), size=min(NEG_FRAMES_PER_CLIP, len(e)), replace=False)
                neg_emb.extend(e[idx])
    neg_emb = np.array(neg_emb)

    print(f"\nDataset: {len(pos_emb)} positives ({len(sources)} sources), "
          f"{len(neg_emb)} negatives")
    return pos_emb, pos_src, neg_emb, len(ref_files)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def make_model(x_train):
    norm = tf.keras.layers.Normalization(axis=-1)
    norm.adapt(x_train)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        norm,
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
    return model


def fit(model, X, y):
    cw = {0: 1.0, 1: float((y == 0).sum() / max(1, (y == 1).sum()))}
    es = tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                          patience=8, restore_best_weights=True)
    model.fit(X, y, validation_split=0.2, epochs=100, batch_size=64,
              class_weight=cw, callbacks=[es], verbose=0)
    return model


def train_on(pos, neg):
    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    p = RNG.permutation(len(X))
    return fit(make_model(X[p]), X[p], y[p])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ensure_esc50()
    model = hub.load(MODEL_HANDLE)
    pos_emb, pos_src, neg_emb, n_ref = build_dataset(model)

    # Shared negative test split for all CV folds.
    nperm = RNG.permutation(len(neg_emb))
    n_test, n_train = neg_emb[nperm[:2000]], neg_emb[nperm[2000:]]

    print("\n=== Grouped CV by source clip (no event leakage) ===")
    gkf = GroupKFold(n_splits=5)
    recalls, fp_rates = [], []
    oof_pos, oof_neg = [], []           # out-of-fold predictions -> honest threshold
    for k, (tr_i, te_i) in enumerate(gkf.split(pos_emb, groups=pos_src)):
        clf = train_on(pos_emb[tr_i], n_train)
        pos_prob = clf.predict(pos_emb[te_i], verbose=0).ravel()
        neg_prob = clf.predict(n_test, verbose=0).ravel()
        oof_pos.extend(pos_prob); oof_neg.extend(neg_prob)
        rec = float((pos_prob >= 0.5).mean())
        fpr = float((neg_prob >= 0.5).mean())
        recalls.append(rec); fp_rates.append(fpr)
        print(f"  fold {k+1}: held-out sources={len(set(pos_src[te_i]))}  "
              f"recall={rec:.2f}  neg-FP-rate={fpr:.3f}")
    print(f"  MEAN recall={np.mean(recalls):.2f}  MEAN neg-FP-rate={np.mean(fp_rates):.3f}")

    # Honest per-frame threshold: the value at which only ~2% of *held-out* negative
    # frames fire (the runtime 2-of-3 rule cuts false positives further). Picking on
    # out-of-fold predictions avoids the leakage that made the naive PR-curve pick a
    # uselessly high threshold (the final model memorizes its own training set).
    oof_pos, oof_neg = np.array(oof_pos), np.array(oof_neg)
    threshold = float(np.clip(np.quantile(oof_neg, 0.98), 0.5, 0.97))
    oof_recall = float((oof_pos >= threshold).mean())
    print(f"  chosen per-frame threshold = {threshold:.3f}  "
          f"(held-out per-frame recall {oof_recall:.2f})")

    print("\n=== Training final model on all data ===")
    final = train_on(pos_emb, neg_emb)

    os.makedirs(MODEL_DIR, exist_ok=True)
    final.save(os.path.join(MODEL_DIR, "can_model.keras"))
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump({
            "threshold": threshold,
            "sample_rate": SR,
            "yamnet_handle": MODEL_HANDLE,
            "smoothing": {"policy": "k_of_n", "k": 2, "n": 3},
            "cv_mean_recall": float(np.mean(recalls)),
            "cv_mean_fp_rate": float(np.mean(fp_rates)),
            "n_positives": int(len(pos_emb)),
            "n_negatives": int(len(neg_emb)),
        }, f, indent=2)
    print(f"\nSaved models/can_model.keras + models/meta.json")
    print(f"Summary: CV recall {np.mean(recalls):.2f}, "
          f"FP rate {np.mean(fp_rates):.3f}, threshold {threshold:.3f}")


if __name__ == "__main__":
    main()
