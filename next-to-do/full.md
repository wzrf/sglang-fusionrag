# FusionRAG 需求与方案（审计版）

> 本文件保留原始需求说明。当前实现状态、风险与下一步请看 `HANDOFF.md`。

## 1. 范围与边界
- 只讨论方案设计、接口设计、测试设计。
- 不重写底层 attention/radix 核心结构。
- 改造重点：请求调度、cache 复用策略、save/load 策略。

## 2. 目标需求（确认版）
1. 在线请求支持 `radix + chunk cache` 协同：
- prefix 尽量命中 radix。
- radix 未命中区间可按 doc chunk 复用 KV。
- chunk 复用不依赖“字符串前缀贪心”，依赖请求传入的 doc 区间与 key。
- chunk 内可指定 `recompute` token 局部重算。
- 未覆盖区间必须完整 prefill。
- 请求结束后完整结果可写回 radix。

2. `kv_gen` 支持两类产物：
- `preprocess`：`prefix + doc` 计算后 strip prefix，仅保存 doc KV。
- `raw`：`doc` 裸请求生成 KV。

3. preprocess 生成阶段允许复用已有 chunk（减少离线生成开销）。

## 3. 执行顺序（统一流水线）
所有请求统一走：
`parse -> plan -> execute -> save`

其中 `generate` 与 `kv_gen` 的主要差异只在 save policy。

## 4. 接口契约（最小必需）

```json
{
  "fusionrag_param": {
    "enable": true,
    "mode": "generate",
    "cache_policy": {
      "use_radix_cache": true,
      "use_chunk_cache": true,
      "save_to_radix_cache": true,
      "save_to_chunk_cache": false
    },
    "chunk_plan": [],
    "recompute_token_idx": [],
    "kv_gen": null
  }
}
```

### 4.1 `mode`
- `generate`（默认）
- `kv_gen`

### 4.2 `chunk_plan`（关键）
每个元素代表一个可复用 doc chunk：

```json
{
  "doc_id": "doc_001",
  "chunk_id": "doc_001_chunk_0",
  "doc_hash": "sha256:optional_client_hint",
  "cache_variant": "raw",
  "start_token": 128,
  "end_token": 2048,
  "use_preprocess_cache": false,
  "recompute_token_idx_local": [3, 10]
}
```

### 4.3 `kv_gen`
仅在 `mode=kv_gen` 时使用：

```json
{
  "target_doc_id": "doc_001",
  "target_chunk_id": "doc_001_chunk_0",
  "save_variants": ["raw", "preprocess"],
  "strip_prefix": true
}
```


### 4.4 doc hash 计算原则
- 推荐由服务端基于标准化后的 doc 内容计算 `doc_hash`。
- 客户端传入的 `doc_hash` 仅作为可选 hint/校验，不应作为唯一真值。
- 推荐 hash 输入至少包含：标准化 doc 文本、tokenizer/version、model revision。

## 5. Token 索引语义（必须统一）
- 索引 0-based。
- 区间左闭右开 `[start, end)`。
- `start_token/end_token` 是相对当前请求 `input_ids` 的全局坐标。
- `recompute_token_idx_local` 是 chunk 局部坐标：
  - `global_idx = start_token + local_idx`

## 6. 覆盖与合并规则
1. 先 radix 命中前缀。
2. chunk 只在 `chunk_plan` 指定区间覆盖。
3. 未被 radix/chunk 覆盖的区间走 prefill。
4. recompute 对指定 token 强制重算覆盖。
5. 最终得到 request-local KV；共享 chunk KV 不允许被原地修改。

## 7. 并发一致性（简化约束）
- 共享 chunk cache 只读。
- 请求内重算只写 request-local KV。
- 推荐状态：`BUILDING | READY | FAILED | STALE`。
- 同 key 并发构建需幂等与互斥（至少单 writer）。

## 8. 保存策略
### 8.1 generate
- 默认：完整请求写回 radix。
- 是否写 chunk 由策略开关控制。

### 8.2 kv_gen
- 默认：只保存目标 doc chunk（raw/preprocess）。
- `strip_prefix=true` 时 preprocess 保存仅保留 doc 区间。

## 9. 校验规则（必须）
1. `mode` 合法。
2. `start_token < end_token`。
3. `chunk_plan` 区间默认不重叠。
4. `recompute_token_idx_local` 不越界。
5. `mode=kv_gen` 时 `kv_gen` 必填。

## 10. 错误码（建议）
- `FRG001` schema 不匹配
- `FRG002` mode 非法
- `FRG003` 区间非法
- `FRG004` 区间重叠
- `FRG005` recompute 越界
- `FRG006` kv_gen 字段缺失
- `FRG007` chunk 未找到
- `FRG008` metadata 不兼容

## 11. 测试口径
### 11.1 无 sglang（逻辑）
- P1 纯 radix
- P2 radix + chunk
- P3 chunk + recompute
- P4 全未命中
- P5 kv_gen doc-only

### 11.2 有 sglang（端到端）
- A 生成 raw kv
- B 生成 preprocess kv（strip prefix）
- C 在线混合命中（radix+chunk+recompute）
- D preprocess 阶段复用已有 chunk

### 11.3 观测字段
- `radix_hit_tokens`
- `chunk_hit_tokens`
- `recompute_tokens`
- `prefill_tokens`
- `save_actions`
- `cached_tokens`

## 12. 建议类与伪代码（实现参考）

> 说明：这是实现参考，不是强制类拆分。可按现有代码结构渐进落地。

### 13.1 FusionRAGParamParser
职责：
- 解析 `fusionrag_param`
- schema 校验
- `chunk_plan` 归一化
- 合并 `recompute` 全局坐标

```python
class FusionRAGParamParser:
    def parse(self, req_json) -> "FusionRAGParam":
        p = req_json.get("fusionrag_param", {})
        if not p.get("enable", False):
            return FusionRAGParam.disabled()

        mode = p.get("mode", "generate")
        assert mode in ("generate", "kv_gen"), "FRG002"

        chunk_plan = self._normalize_chunk_plan(p.get("chunk_plan", []))
        recompute_global = self._merge_recompute(
            p.get("recompute_token_idx", []), chunk_plan
        )

        kv_gen = p.get("kv_gen")
        if mode == "kv_gen":
            self._validate_kv_gen(kv_gen)

        return FusionRAGParam(
            enable=True,
            mode=mode,
            cache_policy=p.get("cache_policy", {}),
            chunk_plan=chunk_plan,
            recompute_global_idx=recompute_global,
            kv_gen=kv_gen,
        )
```

### 13.2 FusionRAGPlanner
职责：
- 先 radix 命中
- 再按 `chunk_plan` 区间做 chunk overlay
- 计算 prefill spans
- 应用 recompute overlay
- 产出 save actions

```python
class FusionRAGPlanner:
    def build_plan(self, input_ids, param, radix_cache, chunk_index) -> "FusionRAGPlan":
        if not param.enable:
            return FusionRAGPlan.fallback_prefill_all(len(input_ids))

        radix_hit = (
            radix_cache.match_prefix(input_ids)
            if param.cache_policy.get("use_radix_cache", True)
            else 0
        )

        chunk_loads = []
        for c in param.chunk_plan:
            node = chunk_index.lookup(c.doc_hash, c.cache_variant, c.use_preprocess_cache)
            if node:
                chunk_loads.append(ChunkLoad(c, node))

        covered = self._compose_coverage(len(input_ids), radix_hit, chunk_loads)
        prefill_spans = self._spans_not_covered(covered)
        recompute_spans = self._to_recompute_spans(param.recompute_global_idx)
        save_actions = self._save_actions(param.mode, param.kv_gen, len(input_ids), param.cache_policy)

        return FusionRAGPlan(
            radix_hit=radix_hit,
            chunk_loads=chunk_loads,
            prefill_spans=prefill_spans,
            recompute_spans=recompute_spans,
            save_actions=save_actions,
        )
```

### 13.3 FusionRAGExecutor
职责：
- 执行 plan（load radix/chunk、prefill、recompute）
- chunk 源只读
- 执行保存动作

```python
class FusionRAGExecutor:
    def run_prefill(self, req, plan, runtime) -> "ExecResult":
        kv_local = runtime.alloc_request_kv(req)

        if plan.radix_hit > 0:
            runtime.load_radix_prefix(req, kv_local, plan.radix_hit)

        for load in plan.chunk_loads:
            runtime.load_chunk_into_local(req, kv_local, load)

        for span in plan.prefill_spans:
            runtime.forward_prefill_span(req, kv_local, span)

        for span in plan.recompute_spans:
            runtime.forward_recompute_span(req, kv_local, span)

        runtime.apply_save_actions(req, kv_local, plan.save_actions)

        return ExecResult(
            kv_local=kv_local,
            stats=runtime.collect_stats(req, plan),
        )
```

### 13.4 ChunkCacheIndex
职责：
- hash 直达
- metadata 兼容性检查

```python
class ChunkCacheIndex:
    def lookup(self, doc_hash, cache_variant, use_preprocess_cache):
        # check READY + model/tokenizer/rope compatibility
        ...
```
