#!/usr/bin/env bash
set -euo pipefail

# FusionRAG demo runner for sa1 (or any similar environment).
#
# Requirements on the target machine:
# - `curl` available
# - `rg` (ripgrep) available
# - Python + transformers available (for computing token indices)
#
# This script avoids `jq` to keep dependencies minimal.

BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
LOG_FILE="${LOG_FILE:-/mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline/sg_e2e_05b.log}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/shm/sglang-fusionrag/venv-kt/bin/python3.10}"
PYTHONPATH_DIR="${PYTHONPATH_DIR:-/mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen2.5-32B-Instruct}"
E2_READ_DELAY_SEC="${E2_READ_DELAY_SEC:-0.15}"

say() { printf "\n== %s ==\n" "$*"; }
pass() { printf "PASS: %s\n" "$*"; }
fail() {
  printf "FAIL: %s\n" "$*" >&2
  exit 1
}

extract_id() {
  # Extracts the first `"id":"..."` from response json.
  # Works for both single and batch response bodies.
  sed -n 's/.*"id":"\([^"]*\)".*/\1/p' | head -n 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 2
  }
}

require_cmd curl
require_cmd rg

json_field() {
  local response="$1"
  local field="$2"
  RESP_JSON="$response" "$PYTHON_BIN" - "$field" <<'PY'
import json
import os
import sys

field = sys.argv[1]
try:
    obj = json.loads(os.environ["RESP_JSON"])
    if isinstance(obj, list):
        obj = obj[0]
    value = obj
    for part in field.split("."):
        value = value[part]
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    print(value)
except Exception:
    print("")
PY
}

assert_response_ok() {
  local name="$1"
  local response="$2"
  local rid
  rid="$(printf "%s" "$response" | extract_id)"
  [ -n "$rid" ] || fail "$name did not return a request id: $response"
  if printf "%s" "$response" | rg -q '"error"|"detail"'; then
    fail "$name response contains an error: $response"
  fi
  pass "$name returned id=$rid"
}

assert_num_ge() {
  local name="$1"
  local actual="$2"
  local expected="$3"
  [[ "$actual" =~ ^-?[0-9]+$ ]] || fail "$name is not numeric: '$actual'"
  [ "$actual" -ge "$expected" ] || fail "$name expected >= $expected, got $actual"
  pass "$name=$actual >= $expected"
}

assert_eq() {
  local name="$1"
  local actual="$2"
  local expected="$3"
  [ "$actual" = "$expected" ] || fail "$name expected '$expected', got '$actual'"
  pass "$name=$actual"
}

assert_fallback_allowed() {
  local name="$1"
  local actual="$2"
  case "$actual" in
    ""|"chunk_miss"|"metadata_mismatch")
      pass "$name fallback_reason='$actual'"
      ;;
    *)
      fail "$name unexpected fallback_reason='$actual'"
      ;;
  esac
}

say "Health"
if ! curl -sS -m 2 "$BASE_URL/model_info" >/dev/null; then
  fail "cannot reach model service at BASE_URL=$BASE_URL (override with BASE_URL=http://<host>:<port>)"
fi
echo "BASE_URL=$BASE_URL"
echo "LOG_FILE=$LOG_FILE"
echo "E2_READ_DELAY_SEC=$E2_READ_DELAY_SEC"

say "A: kv_gen raw save"
RESP_A="$(
  curl -sS "$BASE_URL/generate" -H 'Content-Type: application/json' -d @- <<'JSON'
{
  "text":"Document RAW demo: Paris is in France.",
  "sampling_params":{"temperature":0,"max_new_tokens":0},
  "fusionrag_params":{
    "enable":true,
    "mode":"kv_gen",
    "cache_policy":{
      "use_radix_cache":false,
      "use_chunk_cache":false,
      "save_to_radix_cache":false,
      "save_to_chunk_cache":true
    },
    "kv_gen":{
      "target_doc_id":"doc_raw_demo",
      "target_chunk_id":"doc_raw_demo_c0",
      "save_variants":["raw"],
      "strip_prefix":false
    }
  }
}
JSON
)"
echo "$RESP_A"
RID_A="$(printf "%s" "$RESP_A" | extract_id)"
echo "RID_A=$RID_A"
assert_response_ok "A raw kv_gen" "$RESP_A"
rg -n "$RID_A|save to RAW cache|cache save path" "$LOG_FILE" | tail -n 60 || true
rg -q "save to RAW cache" "$LOG_FILE" || fail "A did not observe raw cache save in LOG_FILE"
pass "A observed raw cache save"

say "B: kv_gen preprocess save (strip_prefix)"
RESP_B="$(
  curl -sS "$BASE_URL/generate" -H 'Content-Type: application/json' -d @- <<'JSON'
{
  "text":"prefixP::The Eiffel Tower is in Paris.",
  "sampling_params":{"temperature":0,"max_new_tokens":0},
  "fusionrag_params":{
    "enable":true,
    "mode":"kv_gen",
    "cache_policy":{
      "use_radix_cache":true,
      "use_chunk_cache":true,
      "save_to_radix_cache":false,
      "save_to_chunk_cache":true
    },
    "kv_gen":{
      "target_doc_id":"doc_pre_demo",
      "target_chunk_id":"doc_pre_demo_c0",
      "save_variants":["preprocess"],
      "strip_prefix":true
    }
  }
}
JSON
)"
echo "$RESP_B"
RID_B="$(printf "%s" "$RESP_B" | extract_id)"
echo "RID_B=$RID_B"
assert_response_ok "B preprocess kv_gen" "$RESP_B"
rg -n "$RID_B|save to PREPROCESS cache|cache save path" "$LOG_FILE" | tail -n 80 || true
rg -q "save to PREPROCESS cache" "$LOG_FILE" || fail "B did not observe preprocess cache save in LOG_FILE"
pass "B observed preprocess cache save"

say "C: radix + chunk + recompute + prefill (dynamic token indices)"
TEXT_C="prefixZ::Document A says Paris is in France. text2 token tail"
DOC_C="Document A says Paris is in France."

mapfile -t _TOK_LINES < <(
  PYTHONPATH="$PYTHONPATH_DIR" "$PYTHON_BIN" - <<PY
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("$MODEL_PATH", trust_remote_code=True)
text="$TEXT_C"
prefix="prefixZ::"
print(len(tok.encode(prefix, add_special_tokens=False)))
print(len(tok.encode(text, add_special_tokens=False)))
PY
)
P_C="${_TOK_LINES[0]:-}"
TOTAL_C="${_TOK_LINES[1]:-}"
if ! [[ "$P_C" =~ ^[0-9]+$ ]] || ! [[ "$TOTAL_C" =~ ^[0-9]+$ ]]; then
  echo "failed to compute token lengths for C: P_C='$P_C' TOTAL_C='$TOTAL_C'" >&2
  exit 2
fi
if [ "$TOTAL_C" -le $((P_C + 4)) ]; then
  fail "C text is too short to force both chunk hit and tail prefill: P_C=$P_C TOTAL_C=$TOTAL_C"
fi
END_C=$((P_C + 6))
if [ "$END_C" -gt $((TOTAL_C - 2)) ]; then END_C=$((TOTAL_C - 2)); fi
echo "C token span: P_C=$P_C END_C=$END_C TOTAL_C=$TOTAL_C"

# Seed only the radix prefix. Seeding the full prompt would hide the tail gap.
RESP_C_RADIX="$(
  curl -sS "$BASE_URL/generate" -H 'Content-Type: application/json' -d @- <<JSON
{
  "text":"prefixZ::",
  "sampling_params":{"temperature":0,"max_new_tokens":1}
}
JSON
)"
RID_C_RADIX="$(printf "%s" "$RESP_C_RADIX" | extract_id)"
echo "RID_C_RADIX=$RID_C_RADIX"
assert_response_ok "C radix prefix seed" "$RESP_C_RADIX"

# Pre-generate preprocess chunk matching this request span (required for a deterministic chunk hit).
RESP_C_PRE="$(
  curl -sS "$BASE_URL/generate" -H 'Content-Type: application/json' -d @- <<JSON
{
  "text":"$TEXT_C",
  "sampling_params":{"temperature":0,"max_new_tokens":0},
  "fusionrag_params":{
    "enable":true,
    "mode":"kv_gen",
    "cache_policy":{
      "use_radix_cache":true,
      "use_chunk_cache":true,
      "save_to_radix_cache":false,
      "save_to_chunk_cache":true
    },
    "chunk_plan":[
      {
        "doc_id":"doc_pre_demo",
        "chunk_id":"doc_pre_demo_c0",
        "doc_text":"$DOC_C",
        "cache_variant":"preprocess",
        "start_token":$P_C,
        "end_token":$END_C,
        "use_preprocess_cache":true,
        "recompute_token_idx_local":[]
      }
    ],
    "kv_gen":{
      "target_doc_id":"doc_pre_demo",
      "target_chunk_id":"doc_pre_demo_c0",
      "save_variants":["preprocess"],
      "strip_prefix":true
    }
  }
}
JSON
)"
RID_C_PRE="$(printf "%s" "$RESP_C_PRE" | extract_id)"
echo "RID_C_PRE=$RID_C_PRE"
assert_response_ok "C preprocess chunk seed" "$RESP_C_PRE"

# Online request with chunk overlay + recompute.
RESP_C="$(
  curl -sS "$BASE_URL/generate" -H 'Content-Type: application/json' -d @- <<JSON
{
  "text":"$TEXT_C",
  "sampling_params":{"temperature":0,"max_new_tokens":4},
  "fusionrag_params":{
    "enable":true,
    "mode":"generate",
    "cache_policy":{
      "use_radix_cache":true,
      "use_chunk_cache":true,
      "save_to_radix_cache":true,
      "save_to_chunk_cache":false
    },
    "chunk_plan":[
      {
        "doc_id":"doc_pre_demo",
        "chunk_id":"doc_pre_demo_c0",
        "doc_text":"$DOC_C",
        "cache_variant":"preprocess",
        "start_token":$P_C,
        "end_token":$END_C,
        "use_preprocess_cache":true,
        "recompute_token_idx_local":[0,2]
      }
    ],
    "recompute_token_idx":[$P_C,$((P_C+1))]
  }
}
JSON
)"
echo "$RESP_C"
RID_C="$(printf "%s" "$RESP_C" | extract_id)"
echo "RID_C=$RID_C"
assert_response_ok "C mixed online" "$RESP_C"
assert_num_ge "C chunk lookup hits" "$(json_field "$RESP_C" "meta_info.fusionrag_chunk_lookup_hits")" 1
assert_num_ge "C chunk hit tokens" "$(json_field "$RESP_C" "meta_info.fusionrag_chunk_hit_tokens")" 1
assert_num_ge "C recompute tokens" "$(json_field "$RESP_C" "meta_info.fusionrag_recompute_tokens")" 1
assert_num_ge "C prefill tokens" "$(json_field "$RESP_C" "meta_info.fusionrag_prefill_tokens")" 1
assert_num_ge "C compute tokens" "$(json_field "$RESP_C" "meta_info.fusionrag_compute_tokens")" 1
assert_eq "C fallback reason" "$(json_field "$RESP_C" "meta_info.fusionrag_fallback_reason")" ""
rg -n "$RID_C|fusionrag_radix_hit_tokens|fusionrag_chunk_hit_tokens|fusionrag_recompute_tokens|fusionrag_prefill_tokens|fusionrag_fallback_reason" "$LOG_FILE" | tail -n 80 || true

say "D: preprocess stage reuses chunks"
TEXT_D="systemX:: [DOC1] [DOC2] [DOC3]"
mapfile -t _D_TOK_LINES < <(
  PYTHONPATH="$PYTHONPATH_DIR" "$PYTHON_BIN" - <<PY
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("$MODEL_PATH", trust_remote_code=True)
text="$TEXT_D"
ids=tok.encode(text, add_special_tokens=False)

def find_span(piece):
    pids=tok.encode(piece, add_special_tokens=False)
    for i in range(0, len(ids)-len(pids)+1):
        if ids[i:i+len(pids)] == pids:
            return i, i+len(pids)
    raise SystemExit(f"cannot find token span for {piece!r}; ids={ids}, pids={pids}")

for piece in (" [DOC1]", " [DOC2]", " [DOC3]"):
    print(*find_span(piece))
PY
)
D1_START="$(printf "%s" "${_D_TOK_LINES[0]:-}" | awk '{print $1}')"
D1_END="$(printf "%s" "${_D_TOK_LINES[0]:-}" | awk '{print $2}')"
D2_START="$(printf "%s" "${_D_TOK_LINES[1]:-}" | awk '{print $1}')"
D2_END="$(printf "%s" "${_D_TOK_LINES[1]:-}" | awk '{print $2}')"
D3_START="$(printf "%s" "${_D_TOK_LINES[2]:-}" | awk '{print $1}')"
D3_END="$(printf "%s" "${_D_TOK_LINES[2]:-}" | awk '{print $2}')"
echo "D token spans: DOC1=[$D1_START,$D1_END) DOC2=[$D2_START,$D2_END) DOC3=[$D3_START,$D3_END)"

for doc in " [DOC1]" " [DOC2]"; do
  DOC_TOKEN_LEN="$(
    PYTHONPATH="$PYTHONPATH_DIR" "$PYTHON_BIN" - <<PY
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("$MODEL_PATH", trust_remote_code=True)
print(len(tok.encode("$doc", add_special_tokens=False)))
PY
  )"
  [[ "$DOC_TOKEN_LEN" =~ ^[0-9]+$ ]] || fail "failed to compute token length for $doc: '$DOC_TOKEN_LEN'"
  RESP_D_PRE="$(
    curl -sS "$BASE_URL/generate" -H 'Content-Type: application/json' -d @- <<JSON
{
  "text":"$doc",
  "sampling_params":{"temperature":0,"max_new_tokens":0},
  "fusionrag_params":{
    "enable":true,
    "mode":"kv_gen",
    "cache_policy":{
      "use_radix_cache":false,
      "use_chunk_cache":false,
      "save_to_radix_cache":false,
      "save_to_chunk_cache":true
    },
    "chunk_plan":[
      {
        "doc_id":"$doc",
        "chunk_id":"${doc}_c0",
        "doc_text":"$doc",
        "cache_variant":"preprocess",
        "start_token":0,
        "end_token":$DOC_TOKEN_LEN,
        "use_preprocess_cache":true,
        "recompute_token_idx_local":[]
      }
    ],
    "kv_gen":{
      "target_doc_id":"$doc",
      "target_chunk_id":"${doc}_c0",
      "save_variants":["preprocess"],
      "strip_prefix":true
    }
  }
}
JSON
  )"
  assert_response_ok "D seed $doc preprocess chunk" "$RESP_D_PRE"
done

RESP_D="$(
  curl -sS "$BASE_URL/generate" -H 'Content-Type: application/json' -d @- <<JSON
{
  "text":"$TEXT_D",
  "sampling_params":{"temperature":0,"max_new_tokens":0},
  "fusionrag_params":{
    "enable":true,
    "mode":"kv_gen",
    "cache_policy":{
      "use_radix_cache":true,
      "use_chunk_cache":true,
      "save_to_radix_cache":false,
      "save_to_chunk_cache":true
    },
    "chunk_plan":[
      {
        "doc_id":" [DOC1]",
        "chunk_id":" [DOC1]_c0",
        "doc_text":" [DOC1]",
        "cache_variant":"preprocess",
        "start_token":$D1_START,
        "end_token":$D1_END,
        "use_preprocess_cache":true,
        "recompute_token_idx_local":[]
      },
      {
        "doc_id":" [DOC2]",
        "chunk_id":" [DOC2]_c0",
        "doc_text":" [DOC2]",
        "cache_variant":"preprocess",
        "start_token":$D2_START,
        "end_token":$D2_END,
        "use_preprocess_cache":true,
        "recompute_token_idx_local":[]
      },
      {
        "doc_id":" [DOC3]",
        "chunk_id":" [DOC3]_c0",
        "doc_text":" [DOC3]",
        "cache_variant":"preprocess",
        "start_token":$D3_START,
        "end_token":$D3_END,
        "use_preprocess_cache":true,
        "recompute_token_idx_local":[]
      }
    ],
    "kv_gen":{
      "target_doc_id":" [DOC3]",
      "target_chunk_id":" [DOC3]_c0",
      "save_variants":["preprocess"],
      "strip_prefix":true
    }
  }
}
JSON
)"
echo "$RESP_D"
RID_D="$(printf "%s" "$RESP_D" | extract_id)"
echo "RID_D=$RID_D"
assert_response_ok "D preprocess reuse kv_gen" "$RESP_D"
assert_num_ge "D chunk lookup hits" "$(json_field "$RESP_D" "meta_info.fusionrag_chunk_lookup_hits")" 2
assert_num_ge "D chunk hit tokens" "$(json_field "$RESP_D" "meta_info.fusionrag_chunk_hit_tokens")" 2
assert_num_ge "D compute tokens" "$(json_field "$RESP_D" "meta_info.fusionrag_compute_tokens")" 0
rg -n "$RID_D|save to PREPROCESS cache|cache save path|fusionrag_chunk_lookup" "$LOG_FILE" | tail -n 120 || true

say "E1: concurrent reads of the same chunk"
TMP_E1_1="/tmp/fusionrag_e1_1_$$.json"
TMP_E1_2="/tmp/fusionrag_e1_2_$$.json"
for i in 1 2; do
  out_var="TMP_E1_${i}"
  out_file="${!out_var}"
  curl -sS "$BASE_URL/generate" -H 'Content-Type: application/json' -d @- >"$out_file" <<JSON &
{
  "text":"$TEXT_C",
  "sampling_params":{"temperature":0,"max_new_tokens":4},
  "fusionrag_params":{
    "enable":true,
    "mode":"generate",
    "cache_policy":{
      "use_radix_cache":true,
      "use_chunk_cache":true,
      "save_to_radix_cache":true,
      "save_to_chunk_cache":false
    },
    "chunk_plan":[
      {
        "doc_id":"doc_pre_demo",
        "chunk_id":"doc_pre_demo_c0",
        "doc_text":"$DOC_C",
        "cache_variant":"preprocess",
        "start_token":$P_C,
        "end_token":$END_C,
        "use_preprocess_cache":true,
        "recompute_token_idx_local":[]
      }
    ]
  }
}
JSON
done
wait
RESP_E1_1="$(cat "$TMP_E1_1")"
RESP_E1_2="$(cat "$TMP_E1_2")"
echo "$RESP_E1_1"
echo "$RESP_E1_2"
assert_response_ok "E1 concurrent read #1" "$RESP_E1_1"
assert_response_ok "E1 concurrent read #2" "$RESP_E1_2"
assert_num_ge "E1 #1 chunk lookup hits" "$(json_field "$RESP_E1_1" "meta_info.fusionrag_chunk_lookup_hits")" 1
assert_num_ge "E1 #2 chunk lookup hits" "$(json_field "$RESP_E1_2" "meta_info.fusionrag_chunk_lookup_hits")" 1
assert_eq "E1 #1 fallback reason" "$(json_field "$RESP_E1_1" "meta_info.fusionrag_fallback_reason")" ""
assert_eq "E1 #2 fallback reason" "$(json_field "$RESP_E1_2" "meta_info.fusionrag_fallback_reason")" ""

say "E2: build/read race for the same chunk"
RACE_DOC="race_doc_$$"
RACE_CHUNK="${RACE_DOC}_c0"
TMP_E2_BUILD="/tmp/fusionrag_e2_build_$$.json"
TMP_E2_READ="/tmp/fusionrag_e2_read_$$.json"
curl -sS "$BASE_URL/generate" -H 'Content-Type: application/json' -d @- >"$TMP_E2_BUILD" <<JSON &
{
  "text":"Race build document for FusionRAG concurrent cache construction.",
  "sampling_params":{"temperature":0,"max_new_tokens":0},
  "fusionrag_params":{
    "enable":true,
    "mode":"kv_gen",
    "cache_policy":{
      "use_radix_cache":false,
      "use_chunk_cache":false,
      "save_to_radix_cache":false,
      "save_to_chunk_cache":true
    },
    "chunk_plan":[
      {
        "doc_id":"$RACE_DOC",
        "chunk_id":"$RACE_CHUNK",
        "doc_text":"Race build document for FusionRAG concurrent cache construction.",
        "cache_variant":"preprocess",
        "start_token":0,
        "end_token":8,
        "use_preprocess_cache":true,
        "recompute_token_idx_local":[]
      }
    ],
    "kv_gen":{
      "target_doc_id":"$RACE_DOC",
      "target_chunk_id":"$RACE_CHUNK",
      "save_variants":["preprocess"],
      "strip_prefix":true
    }
  }
}
JSON
sleep "$E2_READ_DELAY_SEC"
curl -sS "$BASE_URL/generate" -H 'Content-Type: application/json' -d @- >"$TMP_E2_READ" <<JSON &
{
  "text":"Race build document for FusionRAG concurrent cache construction.",
  "sampling_params":{"temperature":0,"max_new_tokens":2},
  "fusionrag_params":{
    "enable":true,
    "mode":"generate",
    "cache_policy":{
      "use_radix_cache":false,
      "use_chunk_cache":true,
      "save_to_radix_cache":false,
      "save_to_chunk_cache":false
    },
    "chunk_plan":[
      {
        "doc_id":"$RACE_DOC",
        "chunk_id":"$RACE_CHUNK",
        "doc_text":"Race build document for FusionRAG concurrent cache construction.",
        "cache_variant":"preprocess",
        "start_token":0,
        "end_token":8,
        "use_preprocess_cache":true,
        "recompute_token_idx_local":[]
      }
    ]
  }
}
JSON
wait
RESP_E2_BUILD="$(cat "$TMP_E2_BUILD")"
RESP_E2_READ="$(cat "$TMP_E2_READ")"
echo "$RESP_E2_BUILD"
echo "$RESP_E2_READ"
assert_response_ok "E2 build side" "$RESP_E2_BUILD"
assert_response_ok "E2 read side" "$RESP_E2_READ"
assert_num_ge "E2 read side compute tokens" "$(json_field "$RESP_E2_READ" "meta_info.fusionrag_compute_tokens")" 0
assert_fallback_allowed "E2 read side" "$(json_field "$RESP_E2_READ" "meta_info.fusionrag_fallback_reason")"

say "Done"
