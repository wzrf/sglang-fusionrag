# 下一步 TODO（带状态）

判定口径：
- `DONE`：在 sa1（或等价环境）实际跑通/观测到关键指标，且脚本/用例可复现
- `PARTIAL`：能力基本具备但仍缺稳定性/缺强断言/缺脚本化
- `TODO`：尚未完成

## DONE（本次已做）
1. `chunk_plan` v2 语义修正：chunk overlay 不要求紧贴 prefix hit
2. output-only logprob 不再误触发 input-logprob 降级；output-only 允许 chunk hit + recompute
3. kv_gen 保存区间按目标 chunk span（避免整段保存导致错配）
4. `demo.md` C 用例补充：动态计算 token 坐标（不写死 start/end）
5. sa1 一键脚本：`run_demo_sa1.sh`（A/B/C/D），已在 sa1 试跑通过

## PARTIAL（需要补稳定性/脚本判定）
1. demo C 稳定体现 “radix + chunk + recompute + prefill”
   - 现状：脚本能稳定做到 chunk hit + recompute；但 `prefill_tokens>0` 需要更稳定的构造（让 chunk_plan 不覆盖全部剩余 prompt）
2. demo D “preprocess 阶段复用已有 chunk”
   - 现状：能保存 preprocess；复用命中与 token 降低需要更明确观测/断言（避免仅靠日志感觉）

## TODO（下一位重点）
1. preprocess cache `unsupport layout detected` 告警
   - 需要定位：旧 artifact / kv layout 版本 / shape 兼容性；并制定版本化 key 或隔离目录策略
2. demo E 并发一致性（E1/E2）
   - 固化成脚本并给出通过判据：同 key 并发构建/读取下不读半成品，不污染共享 chunk
3. single-pass prefill（一次性前向）设计与实现
   - 目标：把 gap + recompute 尽量合并到一次 EXTEND forward（不改 causal 语义）
   - 落点：planner 输出 `compute_positions`；alloc/write 支持按 position 写 `req_to_token_pool`；forward 支持显式 `position_ids` + `out_cache_loc`
   - 参考算子：`ktransformers/operators/sparse_attention.py:selected_query_sparse_attention`（仅供思路，仍需接入完整 transformer forward）
4. `input logprob + cache hit` 是否需要支持
   - 现状：稳定降级 `input_logprob_cache_hit_unsupported`
   - 需产品确认：接受降级 or 需要实现“cache hit 也返回 input logprob”

