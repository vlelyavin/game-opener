#!/bin/bash
# make_voice.sh — generate the funny Russian announcement played on can detection.
# Uses macOS `say` (voice Milena, ru_RU) + ffmpeg pitch/tempo for the comedic effect.
# Re-run this to change the phrase or style.  Output: assets/voice.wav
set -e
cd "$(dirname "$0")"
mkdir -p assets

PHRASE="${1:-Ебать, ну наконец-то, запускаю, браток!}"
# Style B "deep/drunk bro": low pitch (asetrate x0.75) + slightly slower feel.
RATE_MULT="${VOICE_RATE_MULT:-0.75}"
TEMPO="${VOICE_TEMPO:-1.05}"

TMP="$(mktemp -t voice).aiff"
say -v Milena -o "$TMP" "$PHRASE"
ffmpeg -y -i "$TMP" \
  -af "asetrate=22050*${RATE_MULT},aresample=22050,atempo=${TEMPO},volume=1.3" \
  assets/voice.wav 2>/dev/null
rm -f "$TMP"
echo "wrote assets/voice.wav  (phrase: \"$PHRASE\", pitch x$RATE_MULT, tempo $TEMPO)"
