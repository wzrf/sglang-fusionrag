#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:30004}"
LOG_FILE="${LOG_FILE:-/mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline/sg_e2e_30004.log}"
TMP_DIR="${TMP_DIR:-/tmp/fusionrag_recompute_nonprefix_case}"
mkdir -p "$TMP_DIR"

NO_FUSION_JSON="$TMP_DIR/no_fusion.json"
KV_GEN_JSON="$TMP_DIR/kv_gen_raw.json"
FUSION_JSON="$TMP_DIR/fusion_chunk_recompute_nonprefix.json"

cat > "$NO_FUSION_JSON" <<'EOF'
{
  "text": "Alpha intro stays uncached. Anchor chunk says Paris is in France. Tail question: Where is Paris?",
  "sampling_params": {"temperature": 0, "max_new_tokens": 64}
}
EOF

cat > "$KV_GEN_JSON" <<'EOF'
{
  "text": "Anchor chunk says Paris is in France.",
  "sampling_params": {"temperature": 0, "max_new_tokens": 0},
  "fusionrag_params": {
    "enable": true,
    "mode": "kv_gen",
    "cache_policy": {
      "use_radix_cache": false,
      "use_chunk_cache": false,
      "save_to_radix_cache": false,
      "save_to_chunk_cache": true
    },
    "chunk_plan": [
      {
        "doc_id": "doc_demo_nonprefix",
        "chunk_id": "doc_demo_nonprefix_c0",
        "doc_text": "Anchor chunk says Paris is in France.",
        "cache_variant": "raw",
        "start_token": 4,
        "end_token": 12,
        "use_preprocess_cache": false,
        "recompute_token_idx_local": []
      }
    ],
    "kv_gen": {
      "target_doc_id": "doc_demo_nonprefix",
      "target_chunk_id": "doc_demo_nonprefix_c0",
      "save_variants": ["raw"],
      "strip_prefix": false
    }
  }
}
EOF

cat > "$FUSION_JSON" <<'EOF'
{
  "text": "Alpha intro stays uncached. Anchor chunk says Paris is in France. Tail question: Where is Paris?",
  "sampling_params": {"temperature": 0, "max_new_tokens": 64},
  "fusionrag_params": {
    "enable": true,
    "mode": "generate",
    "cache_policy": {
      "use_radix_cache": false,
      "use_chunk_cache": true,
      "save_to_radix_cache": false,
      "save_to_chunk_cache": false
    },
    "chunk_plan": [
      {
        "doc_id": "doc_demo_nonprefix",
        "chunk_id": "doc_demo_nonprefix_c0",
        "doc_text": "Anchor chunk says Paris is in France.",
        "cache_variant": "raw",
        "start_token": 4,
        "end_token": 12,
        "use_preprocess_cache": false,
        "recompute_token_idx_local": [1, 3]
      }
    ],
    "recompute_token_idx": [5, 7]
  }
}
EOF

echo "== Health =="
curl -sf "$BASE_URL/health" >/dev/null && echo "OK"

echo
echo "== Request JSON (human-readable) =="
for f in "$NO_FUSION_JSON" "$KV_GEN_JSON" "$FUSION_JSON"; do
  echo "----- $f -----"
  cat "$f"
done

echo
echo "== 1) no-fusion baseline =="
BASE_RESP=$(curl -sS -X POST "$BASE_URL/generate" -H "Content-Type: application/json" --data-binary @"$NO_FUSION_JSON")
echo "$BASE_RESP" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("text",""))'

echo
echo "== 2) kv_gen raw save =="
KV_RESP=$(curl -sS -X POST "$BASE_URL/generate" -H "Content-Type: application/json" --data-binary @"$KV_GEN_JSON")
echo "$KV_RESP" | python3 -c 'import sys,json; d=json.load(sys.stdin); m=d.get("meta_info",{}); print("id=",m.get("id")); print("prompt_tokens=",m.get("prompt_tokens"))'

echo
echo "== 3) fusion chunk+recompute (non-prefix) =="
FUSION_RESP=$(curl -sS -X POST "$BASE_URL/generate" -H "Content-Type: application/json" --data-binary @"$FUSION_JSON")
echo "$FUSION_RESP" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("text","")); m=d.get("meta_info",{}); c=m.get("customized_info",{}); print("fusionrag_chunk_lookup_hits", c.get("fusionrag_chunk_lookup_hits", [])); print("fusionrag_chunk_lookup_misses", c.get("fusionrag_chunk_lookup_misses", [])); print("fusionrag_recompute_tokens", c.get("fusionrag_recompute_tokens", [])); print("fusionrag_prefill_tokens", c.get("fusionrag_prefill_tokens", [])); print("fusionrag_fallback_reason", c.get("fusionrag_fallback_reason", []))'

echo
echo "== 4) log snippets =="
rg -n "req_summary|extend_shape|recompute_map|drop_prefix_recompute_idx_strict|drop_unmapped_recompute_idx|fusionrag_prefill_compute" "$LOG_FILE" | tail -n 80 || true

echo
echo "== done =="
