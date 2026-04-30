#!/usr/bin/env bash
set -euo pipefail

# Quarantine incompatible FusionRAG cache entries reported by server startup logs.
# It only moves entries explicitly reported as incompatible:
#   [FusionRAG] skip incompatible chunk cache: path=...
#
# Usage:
#   ./quarantine_incompatible_cache.sh /path/to/sg_e2e_05b.log
#
# Optional env:
#   QUARANTINE_ROOT=/mnt/data3/xmy/fusionrag_tree_cache_quarantine

LOG_FILE="${1:-}"
QUARANTINE_ROOT="${QUARANTINE_ROOT:-/mnt/data3/xmy/fusionrag_tree_cache_quarantine}"

if [ -z "$LOG_FILE" ]; then
  echo "usage: $0 <server_log_file>" >&2
  exit 2
fi

if [ ! -f "$LOG_FILE" ]; then
  echo "log file not found: $LOG_FILE" >&2
  exit 2
fi

TENSOR_PATHS="$(
  rg -o 'path=[^ ]+' "$LOG_FILE" \
    | sed 's/^path=//' \
    | rg 'fusionrag_tree_cache/.*/\(raw_kv_cache\|preprocess_kv_cache\)/v2/.+/.+\.pt$' \
    | sort -u || true
)"

if [ -z "$TENSOR_PATHS" ]; then
  echo "no incompatible cache paths found in $LOG_FILE"
  exit 0
fi

TS="$(date +%Y%m%d_%H%M%S)"
DEST_ROOT="$QUARANTINE_ROOT/$TS"
mkdir -p "$DEST_ROOT"

printf "%s\n" "$TENSOR_PATHS" | while IFS= read -r tensor_path; do
  [ -n "$tensor_path" ] || continue
  cache_dir="$(dirname "$tensor_path")"
  if [ ! -d "$cache_dir" ]; then
    echo "skip missing dir: $cache_dir"
    continue
  fi
  rel="${cache_dir#/}"
  dest="$DEST_ROOT/$rel"
  mkdir -p "$(dirname "$dest")"
  mv "$cache_dir" "$dest"
  echo "moved: $cache_dir -> $dest"
done

moved="$(find "$DEST_ROOT" -type d | wc -l | tr -d ' ')"
if [ "$moved" -gt 0 ]; then
  moved=$((moved - 1))
fi
echo "quarantined $moved cache dirs into $DEST_ROOT"
