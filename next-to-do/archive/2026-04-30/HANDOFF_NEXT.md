# Handoff（给下一位 Agent）

日期：2026-04-30

## 1) 你接手要完成什么
把 FusionRAG v2（`full.md`）从“能跑”推进到“可稳定验收”：
- `generate` 与 `kv_gen` 共用统一管线：`parse -> plan -> execute -> save`
- `radix + chunk + recompute + prefill + save` 行为与 `full.md` 一致
- `demo.md` A~D（+E 并发）变成稳定可复现

## 2) 当前代码状态（关键点）
分支：`fusionrag_unified_pipeline`

已落地的关键修复/能力（相对 handoff.md 的后续进展）：
1. chunk overlay v2 语义修正：不再要求 chunk span 必须紧贴 prefix hit
   - `python/sglang/srt/fusionrag_plan.py:select_prefix_compatible_chunk_hits`
2. output-only logprob 不再误触发 input-logprob 降级；output-only 允许 chunk hit + recompute
   - `python/sglang/srt/managers/schedule_batch.py:requests_input_logprobs`
   - `python/sglang/srt/managers/schedule_batch.py:apply_fusionrag_runtime_guards`
3. kv_gen 保存区间修正：按 plan 的目标 chunk span 保存（避免整段 prompt 保存导致错配）
   - `python/sglang/srt/mem_cache/fusionrag_cache.py:cache_finished_req`
4. demo C 修正：不再写死 token 坐标，改为动态算 `start_token/end_token`
   - `next-to-do/demo.md`
5. 新增 sa1 一键脚本（A/B/C/D）
   - `next-to-do/run_demo_sa1.sh`

## 3) sa1 运行与验证（已跑通）
### 3.1 runtime smoke（之前 handoff 提到的）
- 命令在旧 handoff.md 里已有；现已能通过断言：
  - output-only logprob：允许 chunk hit + recompute
  - input-logprob：稳定降级 `input_logprob_cache_hit_unsupported`

### 3.2 e2e（轻量服务，端口 30003）
为了稳定跑 demo，用过 0.5B 单卡服务：
- `BASE_URL=http://127.0.0.1:30003`
- `LOG_FILE=/mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline/sg_e2e_05b.log`
一键跑：
- `cd /mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline/next-to-do && ./run_demo_sa1.sh`

注意：
- 该脚本的 C 用例会先 kv_gen 生成匹配的 preprocess chunk，再在线命中 chunk（用于稳定复现 chunk hit + recompute）。

## 4) 未完成事项（下一位必须继续）
1. preprocess cache `unsupport layout detected` 告警：需要定位 shape/layout 兼容性或旧 artifact 影响，并制定版本化/隔离策略。
2. demo C 的“prefill_tokens>0”稳定展示：当前脚本能跑到 chunk hit + recompute，但是否能稳定体现 gap prefill 需要进一步构造（让 chunk_plan 不覆盖全部剩余 prompt）。
3. demo E 并发一致性：未固化成脚本与通过判据（需要跑 E1/E2 并补 fallback reason）。
4. `input logprob + cache hit` 降级是否可接受：目前实现是稳定降级（而非支持该组合），需确认产品需求。

## 6) single-pass prefill（一次性前向）设计方向
目标与落点已整理到 `NEXT_PLAN.md`（TODO 列表里），这里不重复展开。

## 5) 同步信息（本地与 sa1）
- 本地仓库：`/Users/hming/code/sglang-fusionrag` 分支 `fusionrag_unified_pipeline`
- sa1 worktree：`/mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline`（手动同步过关键文件）

建议后续用 `git` 做正式同步（避免 scp 漂移），并在 sa1 上用同一 worktree 跑验收脚本。
