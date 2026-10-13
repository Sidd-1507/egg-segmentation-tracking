#!/usr/bin/env bash
# Downloads the Roboflow "egg" v1 export (110 images, CC BY 4.0) into data/raw.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
gdown --fuzzy "https://drive.google.com/file/d/1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv/view?usp=sharing" -O data/eggs.zip
unzip -qo data/eggs.zip -d data/raw
echo "images: $(find data/raw -name '*.jpg' | wc -l)"
