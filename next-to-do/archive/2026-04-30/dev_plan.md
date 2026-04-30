# FusionRAG 开发 Plan（给执行 Agent）

## 阶段 0：基线盘点（0.5 天）
1. 定位当前 radix 命中入口、prefill 执行边界、kv_gen 入口。
2. 定位 chunk/hicache 现有实现与并发控制点。
3. 产出 `as-is` 流程图和可复用模块列表。

交付：`docs/fusionrag_as_is.md`

## 阶段 1：接口与数据结构冻结（0.5 天）
1. 冻结 `fusionrag_param` schema（含默认值、校验、错误码）。
2. 冻结 `FusionRAGPlan` schema。
3. 冻结 chunk metadata schema（含兼容字段）。

交付：
- `docs/fusionrag_param_schema.md`
- `docs/fusionrag_plan_schema.md`

## 阶段 2：Planner 抽象落地（1 天）
1. 新增 `FusionRAGParamParser`。
2. 新增 `FusionRAGPlanner`（仅决策，不做执行）。
3. 在现有入口接入 planner，但保持行为尽量等价。

交付：
- planner 单测（无 sglang）
- 关键日志字段（plan dump 开关）

## 阶段 3：Executor/SavePolicy 重构（1.5 天）
1. 新增/整理 `FusionRAGExecutor`：load radix -> load chunk -> recompute -> prefill。
2. 新增 `FusionRAGSavePolicy`：
   - generate: save radix（可配置）
   - kv_gen: save chunk(raw/preprocess), doc-only, strip_prefix
3. 明确 ChunkCacheKV 只读语义；RequestKV 可写语义。

交付：
- 执行路径时序图
- 并发副本规则文档

## 阶段 4：kv_gen 双产物能力（1 天）
1. 支持 `save_variants=[raw, preprocess]`。
2. 支持 preprocess strip_prefix 保存。
3. 支持 preprocess 阶段复用 prefix 内已有 chunk。

交付：
- kv_gen A/B/D 三类测试通过

## 阶段 5：端到端验证与回归（1 天）
1. 不启动 sglang 的逻辑测试全通过。
2. 启动 sglang 的 A/B/C/D 用例全通过。
3. 输出命中率/延迟/正确性报告。

交付：`docs/fusionrag_eval_report.md`

## 阶段 6：灰度与回滚（0.5 天）
1. 增加开关：`enable_fusionrag`、`mode`、`save_to_*`。
2. 明确快速回滚策略：关闭 fusionrag_param 即回退旧路径。

交付：`docs/fusionrag_rollout.md`

## DoD（完成定义）
- 统一入口支持 generate 与 kv_gen。
- planner 产出的 plan 可解释且可观测。
- 并发下无共享 chunk 被污染。
- 四类端到端 case 可稳定复现并通过。
- 文档/测试脚本完整可交接。

## 补充：接口冻结与测试门禁

### 接口冻结门禁（阶段 1 完成判定）
1. `fusionrag_param`/`chunk_plan`/`recompute_token_idx(_local)` 字段表完成且与 full.md 一致。
2. token 坐标语义（global/local/source）文档化并评审通过。
3. 错误码 `FRG001~FRG008` 冻结。

### 测试门禁（阶段 5 完成判定）
1. planner-only: P1~P5 全通过。
2. e2e: A~D 全通过。
3. 日志统一输出 `radix_hit_tokens/chunk_hit_tokens/recompute_tokens/prefill_tokens`。
4. 并发一致性 case 通过（共享 chunk 不被请求内重算污染）。
