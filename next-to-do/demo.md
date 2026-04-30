# FusionRAG Demo 与测试标准（可执行版）

## 1. 目标
这份文档给出：
1. 代表请求（planner-only 与 e2e）
2. 可直接运行的 `curl` 命令
3. 日志检查命令
4. 通过判据

> 默认服务地址：`http://127.0.0.1:30002`

---

## 2. 前置条件

### 2.1 启动服务
示例（按当前项目脚本）：
```bash
cd /mnt/data/shm/sglang-fusionrag
bash qwen_sglang.sh
```

### 2.2 环境变量
```bash
export BASE_URL="http://127.0.0.1:30002"
export LOG_FILE="/mnt/data/shm/sglang-fusionrag/sg_qwen2.5.log"
```

### 2.3 健康检查
```bash
curl -sS "$BASE_URL/model_info" | jq .
```
通过标准：HTTP 200 且返回 model 信息。

---

## 3. 通用观测指标
每次请求必须能从返回体或日志提取：
- `radix_hit_tokens`
- `chunk_hit_tokens`
- `prefill_tokens`
- `recompute_tokens`
- `cached_tokens`
- `save_actions`
- `fallback_reason`（可空）

建议日志统一关键字：`FusionRAG` / `cache_finished_req` / `highlight_recompute_tokens`。

---

## 4. Planner-only 测试（不启动 sglang）

> 这里不跑模型，重点验证 plan 逻辑。若有本地 planner 脚本，可按以下最小输入组织 JSON 并断言。

### P1: 纯 radix
- 输入：无 `chunk_plan`
- 期望：`chunk_hit_tokens=0`，`radix_hit_tokens>0`

### P2: radix + chunk
- 输入：有 `chunk_plan` 且服务端计算后可命中对应 chunk cache
- 期望：`chunk_hit_tokens>0` 且 `prefill_tokens` 下降

### P3: chunk + recompute
- 输入：`recompute_token_idx_local` 非空
- 期望：`recompute_tokens>0`

### P4: chunk miss 回退
- 输入：服务端计算后无法命中对应 chunk cache
- 期望：`fallback_reason=chunk_miss`（或等价）

### P5: kv_gen doc-only
- 输入：`mode=kv_gen` + `strip_prefix=true`
- 期望：save action 仅覆盖 doc 区间

---

## 5. 端到端测试（启动 sglang）


### 5.1 关于 doc_hash
- 推荐服务端自行从 `doc_text`（或服务端已知 doc 内容）计算 hash。
- 本文请求示例优先使用 `doc_text`，避免客户端手动构造 hash。
- 若客户端传了 `doc_hash`，服务端可用于快速 hint，但应以服务端计算结果为准。


## 5.0 通用函数（可选）
```bash
extract_id() {
  echo "$1" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p'
}
```

### A. 生成 raw KV

#### A1 请求
```bash
RESP_A=$(curl -sS "$BASE_URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "text":"Document RAW demo: Paris is in France.",
    "sampling_params":{"temperature":0,"max_new_tokens":0},
    "fusionrag_param":{
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
  }')

echo "$RESP_A" | jq .
RID_A=$(extract_id "$RESP_A")
echo "RID_A=$RID_A"
```

#### A2 日志检查
```bash
rg -n "$RID_A|save to RAW cache|cache save path" "$LOG_FILE" | tail -n 30
```

#### A3 通过标准
- 请求成功返回（HTTP 200）
- 日志出现 `save to RAW cache`
- 有对应 `cache save path`

---

### B. 生成 preprocess KV（strip prefix）

#### B1 请求
```bash
RESP_B=$(curl -sS "$BASE_URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "text":"prefixP::The Eiffel Tower is in Paris.",
    "sampling_params":{"temperature":0,"max_new_tokens":0},
    "fusionrag_param":{
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
  }')

echo "$RESP_B" | jq .
RID_B=$(extract_id "$RESP_B")
echo "RID_B=$RID_B"
```

#### B2 日志检查
```bash
rg -n "$RID_B|save preprocess cache|save to PREPROCESS cache|cache save path" "$LOG_FILE" | tail -n 40
```

#### B3 通过标准
- 请求成功
- 日志出现 `save to PREPROCESS cache`
- 保存路径存在

---

### C. 在线混合命中（radix + chunk + recompute）

> 该 case 假设 A/B 已经产生可复用 cache。
>
> 注意：`chunk_plan.start_token/end_token` 是基于 **tokenizer 的 token 坐标**，不同模型/模板会导致 token 数不同。
> 因此不建议写死数字。下面给出“动态计算 token 索引”的做法（在 sa1 上执行）。

#### C0 计算 token 坐标（必做一次）
在 sa1 上用与服务一致的 tokenizer 计算 `prefix_tokens` 和 `total_tokens`，并据此构造区间：
```bash
PYTHONPATH=/mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline/python \
/mnt/data/shm/sglang-fusionrag/venv-kt/bin/python3.10 - <<'PY'
from transformers import AutoTokenizer
model="/mnt/data/models/Qwen2.5-0.5B-Instruct"  # 或替换为你实际 serving 的 model path
tok=AutoTokenizer.from_pretrained(model, trust_remote_code=True)
text="prefixZ::Document A says Paris is in France. text2 token tail"
prefix="prefixZ::"
ids=tok.encode(text, add_special_tokens=False)
pids=tok.encode(prefix, add_special_tokens=False)
print("prefix_tokens=", len(pids))
print("total_tokens=", len(ids))
PY
```
把输出的 `prefix_tokens` 记为 `P`，则建议：
- `start_token = P`
- `end_token = min(P + 16, total_tokens)`（按你的 doc 长度调整）

#### C1 第一次请求（用于建立 radix 前缀）
```bash
RESP_C1=$(curl -sS "$BASE_URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "text":"prefixZ::Document A says Paris is in France. text2 token tail",
    "sampling_params":{"temperature":0,"max_new_tokens":8}
  }')

echo "$RESP_C1" | jq .
RID_C1=$(extract_id "$RESP_C1")
```

#### C2 第二次请求（带 chunk_plan + recompute）
```bash
RESP_C2=$(curl -sS "$BASE_URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "text":"prefixZ::Document A says Paris is in France. text2 token tail",
    "sampling_params":{"temperature":0,"max_new_tokens":8},
    "fusionrag_param":{
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
          "doc_text":"Document A says Paris is in France.",
          "cache_variant":"preprocess",
          "start_token":<P>,
          "end_token":<P_plus_len>,
          "use_preprocess_cache":true,
          "recompute_token_idx_local":[0,2]
        }
      ],
      "recompute_token_idx":[10,11]
    }
  }')

echo "$RESP_C2" | jq .
RID_C2=$(extract_id "$RESP_C2")
```

#### C3 日志检查
```bash
rg -n "$RID_C1|$RID_C2|Prefill batch|cached-token|cache_finished_req|highlight_recompute_tokens|use_preprocess_cache" "$LOG_FILE" | tail -n 120
```

#### C4 通过标准
- C2 返回成功
- 日志可见 recompute 触发（如 `highlight_recompute_tokens`）
- C2 存在 cache 参与迹象（`cached-token > 0` 或 chunk/load 命中日志）
- 未覆盖区间仍有 prefill（`Prefill batch` 新 token > 0）

---

### D. preprocess 阶段复用已有 chunk

#### D1 请求
```bash
RESP_D=$(curl -sS "$BASE_URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "text":"systemX::doc1 doc2 doc3",
    "sampling_params":{"temperature":0,"max_new_tokens":0},
    "fusionrag_param":{
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
          "doc_id":"doc1",
          "chunk_id":"doc1_c0",
          "doc_text":"doc1",
          "cache_variant":"preprocess",
          "start_token":4,
          "end_token":12,
          "use_preprocess_cache":true,
          "recompute_token_idx_local":[]
        },
        {
          "doc_id":"doc2",
          "chunk_id":"doc2_c0",
          "doc_text":"doc2",
          "cache_variant":"preprocess",
          "start_token":12,
          "end_token":20,
          "use_preprocess_cache":true,
          "recompute_token_idx_local":[]
        }
      ],
      "kv_gen":{
        "target_doc_id":"doc3",
        "target_chunk_id":"doc3_c0",
        "save_variants":["preprocess"],
        "strip_prefix":true
      }
    }
  }')

echo "$RESP_D" | jq .
RID_D=$(extract_id "$RESP_D")
```

#### D2 日志检查
```bash
rg -n "$RID_D|chunk_hit|load text|save to PREPROCESS cache|Prefill batch" "$LOG_FILE" | tail -n 80
```

#### D3 通过标准
- 预处理请求成功
- 日志显示 preprocess 保存成功
- 若已有 chunk，可见 chunk 命中或 prefill token 降低迹象

---

### E. 并发一致性测试

#### E1 并发读取同一 chunk
```bash
for i in 1 2; do
  curl -sS "$BASE_URL/generate" -H 'Content-Type: application/json' -d '{
    "text":"prefixP::The Eiffel Tower is in Paris.",
    "sampling_params":{"temperature":0,"max_new_tokens":4},
    "fusionrag_param":{
      "enable":true,
      "mode":"generate",
      "cache_policy":{"use_radix_cache":true,"use_chunk_cache":true,"save_to_radix_cache":true,"save_to_chunk_cache":false},
      "chunk_plan":[{"doc_id":"doc_pre_demo","chunk_id":"doc_pre_demo_c0","doc_text":"The Eiffel Tower is in Paris.","cache_variant":"preprocess","start_token":8,"end_token":24,"use_preprocess_cache":true,"recompute_token_idx_local":[]}]
    }
  }' > /tmp/fusionrag_e1_$i.json &
done
wait
jq . /tmp/fusionrag_e1_1.json
jq . /tmp/fusionrag_e1_2.json
```

通过标准：
- 两个请求都成功
- 无明显污染/异常日志

#### E2 构建 + 读取竞争
- 在 kv_gen 构建同 key 的同时发 generate 读取同 key。
- 通过标准：
  - 读取请求要么等待并成功，要么按策略 fallback prefill 成功
  - 不出现半成品读取错误

---

## 6. 最终验收
满足以下即可判定“满足开发需求”：
1. A/B 能稳定生成 raw/preprocess。
2. C 能体现 radix+chunk+recompute+prefill 混合路径。
3. D 能体现 preprocess 阶段复用已有 chunk。
4. E 并发场景无污染、无崩溃。
5. 关键统计字段可观测、可复盘。
