#!/usr/bin/env bash
# Downloads MELD.Raw into data/meld/raw/, extracts it, fetches the label CSVs,
# and verifies every video clip is readable, logging corrupt/unreadable clips
# to data/meld/bad_clips.txt so preprocessing can exclude them consistently.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RAW_DIR="data/meld/raw"
ARCHIVE="$RAW_DIR/MELD.Raw.tar.gz"
EXTRACT_DIR="$RAW_DIR/MELD.Raw"
LABELS_DIR="$RAW_DIR/labels"
BAD_CLIPS_FILE="data/meld/bad_clips.txt"

PRIMARY_URL="https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"
MIRROR_URL="https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz"
# Sanity-check tolerance: the two mirrors should serve (near) identical bytes.
EXPECTED_ARCHIVE_SIZE=10878146150
SIZE_TOLERANCE=50000000

mkdir -p "$RAW_DIR" "$LABELS_DIR" "$(dirname "$BAD_CLIPS_FILE")"

download() {
  local url="$1"
  echo "[download_meld] trying $url"
  curl -fL --retry 5 --retry-delay 5 -C - -o "$ARCHIVE" "$url"
}

if [[ ! -s "$ARCHIVE" ]]; then
  download "$PRIMARY_URL" || download "$MIRROR_URL"
else
  echo "[download_meld] archive already present at $ARCHIVE, skipping download"
fi

actual_size=$(stat -c%s "$ARCHIVE")
echo "[download_meld] archive size: $actual_size bytes (expected ~$EXPECTED_ARCHIVE_SIZE)"
if (( actual_size < EXPECTED_ARCHIVE_SIZE - SIZE_TOLERANCE )); then
  echo "[download_meld] ERROR: archive size far below expected — download likely incomplete/corrupt" >&2
  exit 1
fi

if [[ ! -d "$EXTRACT_DIR" ]] || [[ -z "$(find "$EXTRACT_DIR" -maxdepth 3 -iname '*.mp4' -print -quit)" ]]; then
  echo "[download_meld] extracting top-level archive"
  mkdir -p "$EXTRACT_DIR"
  tar -xzf "$ARCHIVE" -C "$EXTRACT_DIR"

  # MELD.Raw.tar.gz contains nested per-split archives (train/dev/test); extract those too.
  while IFS= read -r -d '' nested; do
    echo "[download_meld] extracting $nested"
    tar -xzf "$nested" -C "$(dirname "$nested")"
  done < <(find "$EXTRACT_DIR" -iname '*.tar.gz' -print0)
else
  echo "[download_meld] $EXTRACT_DIR already populated with .mp4 files, skipping extraction"
fi

for split in train dev test; do
  csv="$LABELS_DIR/${split}_sent_emo.csv"
  if [[ ! -s "$csv" ]]; then
    echo "[download_meld] fetching ${split}_sent_emo.csv"
    curl -fL -o "$csv" \
      "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/${split}_sent_emo.csv"
  fi
done

echo "[download_meld] utterance counts vs. mp4 files found on disk:"
for split in train dev test; do
  csv="$LABELS_DIR/${split}_sent_emo.csv"
  n_rows=$(($(wc -l < "$csv") - 1))
  # Exclude macOS AppleDouble sidecar files (._dia*.mp4) bundled in the tar archives.
  n_mp4=$(find "$EXTRACT_DIR" -type f -iname '*.mp4' ! -iname '._*' -ipath "*${split}*" | wc -l)
  echo "  split=$split labeled_utterances=$n_rows mp4_files_found=$n_mp4"
done

echo "[download_meld] verifying clip readability with ffprobe (parallel over $(nproc) cores)..."
: > "$BAD_CLIPS_FILE"
find "$EXTRACT_DIR" -type f -iname '*.mp4' ! -iname '._*' -print0 \
  | xargs -0 -P "$(nproc)" -I{} bash -c '
      f="{}"
      if ! ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" >/dev/null 2>/dev/null; then
        echo "$f"
      fi
    ' >> "$BAD_CLIPS_FILE"

n_bad=$(wc -l < "$BAD_CLIPS_FILE")
echo "[download_meld] found $n_bad unreadable/corrupt clip(s), logged to $BAD_CLIPS_FILE"
echo "[download_meld] done."
