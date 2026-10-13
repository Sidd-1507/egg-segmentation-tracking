#!/usr/bin/env bash
# Downloads the Task 4 video ("Raw and Boiled Egg Spinning", MocomiKids) into task4_tracking/input.
# The default YouTube client refuses this video; the android client serves format 18 (640x360, 24 fps).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p task4_tracking/input
python -m yt_dlp --extractor-args "youtube:player_client=android" -f 18 \
  -o task4_tracking/input/eggs.mp4 "https://www.youtube.com/watch?v=Y4JyV94y2_A"
