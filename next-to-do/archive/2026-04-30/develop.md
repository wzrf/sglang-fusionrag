# FusionRAG 开发说明（详细执行版）

## 1. 文档定位
- `full.md`：唯一权威需求与接口定义（source of truth）。
- `develop.md`：工程落地说明（详细步骤、注意事项、实现顺序）。
- `dev_plan.md`：阶段化推进计划与门禁。
- `demo.md`：测试样例与通过标准。

> 规则：若本文件与 `full.md` 冲突，以 `full.md` 为准，并先同步修正文档再开发。

---

## 2. 要解决的问题（工程视角）
你要落地的是“统一编排”，而不是两套流程：
- 统一处理 `generate` 与 `kv_gen`。
- 统一处理 `radix` 命中、`chunk` 复用、`recompute` 覆盖、`prefill` 补全、`save` 落库。

目标行为：
1. prefix 尽量走 radix cache。
2. 指定 doc 区间走 chunk cache（按 `chunk_plan` 显式范围，不靠字符串贪心）。
3. chunk 内指定 token 做 recompute。
4. 其余未覆盖区间完整 prefill。
5. 按模式执行保存策略（generate/kv_gen）。

---

## 3. 不可破坏约束
1. **接口命名统一**：
   - `chunk_plan`
   - `recompute_token_idx`
   - `recompute_token_idx_local`
2. **坐标语义统一**：0-based，区间 `[start, end)`。
3. **共享 chunk 只读**：recompute 只改 request-local KV。
4. **先 radix 后 chunk**：chunk 只覆盖 `chunk_plan` 指定区间。
5. **fallback 可解释**：miss/mismatch 时有统一回退与日志标记。

---

## 4. 推荐实现切分（逻辑层，不强制类名）

### 4.1 Parse 层
职责：
- 读取 `fusionrag_param`
- 校验模式与字段
- 归一化 `chunk_plan`
- 合并全局/局部 recompute 坐标

输入：原始请求 JSON
输出：标准化 `FusionRAGParam`

### 4.2 Plan 层
职责：
- 调 `radix` 获取 prefix 命中
- 依据 `chunk_plan` + cache index 决定 chunk_loads
- 形成 prefill spans
- 叠加 recompute spans
- 生成 save_actions

输入：`input_ids`, `FusionRAGParam`, cache state
输出：`FusionRAGPlan`

### 4.3 Execute 层
职责：
- 创建 request-local KV
- 执行 load radix / load chunk / prefill / recompute
- 执行 save actions
- 记录 debug 统计

输入：`FusionRAGPlan`
输出：执行结果 + debug 信息

---

## 5. 字段解释（工程口径）
详细字段定义见 `full.md`。这里补工程常见误区。

### 5.1 `chunk_plan`
每个 chunk 描述“请求中的一个 doc 片段可复用机会”：
- `start_token/end_token`: 当前 `input_ids` 全局坐标
- `doc_hash`: 可选客户端提示；服务端可忽略并自行计算。
- `cache_variant`: raw/preprocess
- `recompute_token_idx_local`: chunk 局部重算点

误区：
- 不要把 `start_token/end_token` 当字符位置。
- 不要混用全局与局部坐标。

### 5.2 `recompute_token_idx` vs `recompute_token_idx_local`
- `recompute_token_idx`: 全局 token 坐标
- `recompute_token_idx_local`: chunk 内坐标，需映射

建议流程：
1. `local -> global`
2. 与全局集合求并集
3. 去重排序
4. 边界裁剪

---

### 5.3 doc_hash 口径
- 由服务端按统一规则计算 `doc_hash`。
- 客户端 `doc_hash` 仅作可选 hint/校验。
- 这样可避免客户端算法不一致导致的错配。

## 6. 统一执行顺序（实现必须一致）
1. Parse `fusionrag_param`
2. Radix 命中（prefix）
3. Chunk 命中（仅 `chunk_plan` 区间）
4. 合并覆盖得到 prefill spans
5. 应用 recompute 覆盖
6. 执行 prefill + recompute
7. 执行 save policy

重点：
- recompute 是覆盖层，不是基础命中层。
- chunk miss 不应导致整个请求失败（除非策略要求）。

---

## 7. save policy（落库策略）

### 7.1 generate
默认：
- 请求完成后写回完整请求到 radix。

可选：
- 是否额外保存 chunk 由策略开关决定。

### 7.2 kv_gen
默认：
- 不写完整请求到 radix（除非显式打开）。
- 只保存目标 doc chunk 到 chunk cache。

`strip_prefix=true`：
- preprocess 保存仅包含 doc 区间 KV，不包含 prefix KV。

---

## 8. 并发与一致性

### 8.1 三副本模型
- `ChunkCacheKV`：共享、只读
- `RequestKV`：请求内可写
- `RadixKV`：请求结束后提交

### 8.2 状态建议
- `BUILDING`
- `READY`
- `FAILED`
- `STALE`

### 8.3 行为建议
- 同 key 并发构建：单 writer + 其余等待或回退。
- 禁止读取半成品（原子提交）。
- metadata mismatch 时走统一 fallback（并打标）。

---

## 9. fallback 口径（建议固定）
1. chunk miss：回退 prefill（默认）
2. metadata mismatch：回退 prefill（默认）
3. recompute 比例过高：可整段退化 prefill
4. planner 发现非法区间：直接报错（不回退）

并统一记录：
- `fallback_reason`
- `affected_chunk_id`

---

## 10. 可观测性（最低要求）
每个请求至少产出：
- `radix_hit_tokens`
- `chunk_hit_tokens`
- `prefill_tokens`
- `recompute_tokens`
- `save_actions`
- `cached_tokens`
- `fallback_reason`

建议统一日志前缀：`[FusionRAG]` 或 `[FusionRAGPlan]`。

---

## 11. 测试执行建议
完整判定看 `demo.md`，这里给执行顺序：
1. 先跑 planner-only（P1~P5）保证区间逻辑正确。
2. 再跑端到端（A~D）验证真实保存/命中。
3. 最后跑并发（E1/E2）验证一致性。

验收顺序不可反：
- 先逻辑正确，再性能验证。

---

## 12. 代码改造优先级建议
1. 参数解析和校验（最先）
2. planner 产物可解释性（第二）
3. executor 覆盖完整性（第三）
4. save policy 细化（第四）
5. 并发与回退（第五）

---

## 13. 常见错误清单
1. 把 chunk 命中当成前缀匹配（错误）。
2. local/global recompute 坐标混淆。
3. 对共享 chunk KV 做原地写（严重错误）。
4. 未覆盖区间漏 prefill。
5. kv_gen 保存时误把 prefix KV 一并保存。
6. 文档更新滞后导致实现与接口漂移。

---

## 14. 与权威文档对齐检查
提交前必须自检：
1. 字段名是否全部与 `full.md` 一致。
2. 坐标语义是否 `[start,end)`。
3. save policy 是否符合 `mode`。
4. `demo.md` case 是否全部可跑并可判定。

通过后再进入代码评审。
