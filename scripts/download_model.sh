#!/bin/bash
set -e

BASE="https://cloud.tsinghua.edu.cn"
DIR="outputs/"

if ! command -v aria2c >/dev/null 2>&1; then
  echo "aria2c is required but was not found in PATH."
  echo "Install it first, for example: sudo apt-get install aria2"
  exit 1
fi

mkdir -p $DIR

download() {
  local url="$BASE/$1/?dl=1"
  local out="$2"
  local out_dir
  local out_name
  out_dir="$(dirname "$out")"
  out_name="$(basename "$out")"
  if [[ -f "$out" ]]; then
    echo "Already exists, skipping: $out"
    return 0
  fi
  echo "Downloading $out..."
  aria2c \
    --allow-overwrite=true \
    --auto-file-renaming=false \
    --continue=true \
    --dir "$out_dir" \
    --out "$out_name" \
    --summary-interval=0 \
    "$url" || { echo "FAILED: $out"; exit 1; }
}

# Intra
# https://cloud.tsinghua.edu.cn/f/be070877300048f887fd/?dl=1
# Inter
# https://cloud.tsinghua.edu.cn/f/ac98b644dc814aa0bc46/?dl=1

# Download the intra-subject model
download "f/be070877300048f887fd" "$DIR/intra_model.zip"
# Download the inter-subject model
download "f/ac98b644dc814aa0bc46" "$DIR/inter_model.zip"

unzip -o "$DIR/intra_model.zip" -d "$DIR/intra_model"
unzip -o "$DIR/inter_model.zip" -d "$DIR/inter_model"

echo "Done."