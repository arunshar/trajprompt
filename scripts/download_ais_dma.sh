#!/usr/bin/env bash
# Pull Danish Maritime Authority daily AIS dumps for a chosen date range.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data/ais"
mkdir -p "$DATA"

START=${1:-2024-01-01}
END=${2:-2024-06-30}
echo "Fetching DMA AIS daily dumps from $START to $END..."

current=$(date -j -f "%Y-%m-%d" "$START" "+%s")
end_epoch=$(date -j -f "%Y-%m-%d" "$END" "+%s")
while [ $current -le $end_epoch ]; do
  d=$(date -j -f "%s" "$current" "+%Y-%m-%d")
  url="https://web.ais.dk/aisdata/aisdk-$d.zip"
  echo "  $d"
  curl -fsS "$url" -o "$DATA/aisdk-$d.zip" || echo "    skip (not available)"
  if [[ -f "$DATA/aisdk-$d.zip" ]]; then
    unzip -q -o "$DATA/aisdk-$d.zip" -d "$DATA"
  fi
  current=$((current + 86400))
done
echo "Done."
