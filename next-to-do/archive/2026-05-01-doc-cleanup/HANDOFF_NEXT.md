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
6. 本轮新增：sa1 脚本升级为 A/B/C/D/E 强断言
   - C：只 seed radix prefix，chunk 覆盖中间 span，保留 tail gap，断言 `chunk_hit_tokens/recompute_tokens/prefill_tokens`
   - D：先预生成 doc1/doc2 preprocess chunk，再在 doc3 kv_gen 中断言 chunk 复用
   - E1/E2：并发读同一 chunk、构建/读取竞争
7. 本轮新增：preprocess/raw cache v2 隔离与兼容加载
   - 新路径：`raw_kv_cache/v2`、`preprocess_kv_cache/v2`
   - 新 metadata：`fusionrag_cache_format_version`、`kv_shape`、`layer_num`
   - 启动加载遇到旧/不兼容 tensor 时结构化 warning + skip，不再打印 `unsupport layout detected`
8. 本轮新增：`kv_gen` v2 分支支持按 `chunk_plan` 查找已有 chunk
   - 之前只检查目标 save action 是否已存在，无法强验证“preprocess 阶段复用已有 chunk”
   - 现在会返回 matched chunk nodes 与 lookup stats，同时保留目标 chunk 已存在时 `no_need_to_run` 的语义

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
- 新版脚本尚需在 sa1 重新实跑；本地已做 `bash -n` 语法检查。

## 4) 未完成事项（下一位必须继续）
1. 在 sa1 重新跑新版 `next-to-do/run_demo_sa1.sh`，确认 A/B/C/D/E 强断言全部通过。
2. 重启服务后观察 cache 加载日志，确认 v2 cache 隔离后不再出现旧 artifact layout 噪音。
3. 若 E2 在真实并发下出现非预期 fallback，需要进一步补单 writer/BUILDING 状态互斥；当前脚本先固化“不崩溃、不读半成品”的验收口径。
4. `input logprob + cache hit` 降级是否可接受：目前实现是稳定降级（而非支持该组合），需确认产品需求。

## 6) single-pass prefill（一次性前向）设计方向
目标与落点已整理到 `NEXT_PLAN.md`（TODO 列表里），这里不重复展开。

## 5) 同步信息（本地与 sa1）
- 本地仓库：`/Users/hming/code/sglang-fusionrag` 分支 `fusionrag_unified_pipeline`
- sa1 worktree：`/mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline`（手动同步过关键文件）

建议后续用 `git` 做正式同步（避免 scp 漂移），并在 sa1 上用同一 worktree 跑验收脚本。
