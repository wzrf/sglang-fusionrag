# FusionRAG Handoff

本文是 `next-to-do/` 目录里唯一的“当前交接文档”。

规则：
- 原始需求只看 `full.md`。
- 当前状态、已修问题、已知风险、下一步只看本文件。
- 历史 handoff/status/todo 已统一挪到 `archive/`。

## 1. 项目目标
把 FusionRAG v2 从“功能打通”推进到“行为正确、性能可接受、可稳定验收”：
- `generate` 与 `kv_gen` 共用统一管线。
- `radix + chunk + recompute + prefill + save` 行为对齐 `full.md`。
- `demo.md` 和 `run_demo_sa1.sh` 能稳定复现关键 case。

## 2. 稳定入口
按这个顺序看：
1. `full.md`：原始需求与接口定义，只保留这一份。
2. `develop.md`：工程实现说明。
3. `dev_plan.md`：阶段计划与门禁。
4. `demo.md`：验收用例与观测指标。
5. `HANDOFF.md`：当前状态与下一步。

## 3. 当前代码状态
主分支/工作树：
- 本地仓库：`/Users/hming/code/sglang-fusionrag`
- 本地分支：`fusionrag_unified_pipeline`
- sa1 工作树：`/mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline`

近期已完成的主线能力：
- FusionRAG 参数归一化、plan 构建、统一执行链已接入。
- `chunk_plan` v2 语义修正：chunk span 是显式 overlay，不再强制与 prefix 连续。
- output-only `return_logprob` 不再误触发 input-logprob 降级。
- `kv_gen` 保存区间按目标 chunk span 保存，不再错误保存整段 prompt。
- preprocess/raw chunk cache 已做 `v2` 路径隔离与兼容加载跳过。
- single-pass prefill 主路径已打通，稀疏写路径支持 Triton，回退时仍可走 torch。
- sa1 验收脚本已扩展到 A/B/C/D/E，并增加更强的 token/compute 指标断言。

近期补上的运行时修复：
- recompute 索引生命周期和映射逻辑修正，避免 stale state 与 prefix-space 误写。
- `forward_batch_info` / `flashinfer_backend` 对齐真实 compute stream 长度。
- `mem_cache/common.py` 的 KV 分配改为按真实 `compute_positions` 对齐。
- `scheduler.py` 修复 overlap 调度时把 `(batch, no_run_list)` 当成 batch 使用的问题。
- `schedule_policy.py` 把 `_remap_loaded_chunk_rope_positions` 正确放回 `PrefillAdder` 可访问位置。
- `fusionrag_cache.py` 修复错误的字符串 `raise`，改为真正异常。
- `fusionrag_cache.py` 修复 `.ready` 哨兵文件并发删除/发布竞态，避免多 TP 写同一 cache key 时崩溃。
- 新增 `qwen_sglang.sh` 启动脚本，便于直接用仓库内框架拉起 Qwen。

本地最近提交：
- `39a1ce9b0` `fusionrag: sync local fixes and sa1 handoff state`
- `92c5d3736` `fusionrag: harden cache ready sentinel writes`
- `77afa1a6f` `fix: restore contiguous prefill path for normal cache hits`

## 3.1 本轮新增结论与修改
本轮优先处理的是“评测输出乱码/同 prompt 第二次命中 cache 后输出漂移”。

已经确认的关键现象：
- 这不是单纯的 Qwen prompt 格式问题。
- 现象可在普通 prefix cache 路径复现，不依赖 FusionRAG chunk load-back。
- 典型复现是：第一次请求 `cached_tokens=0` 输出正常；第二次请求 `cached_tokens=N-1`、只新算 1 个 token，但输出发生明显漂移。

当前最强怀疑点：
- 重构后普通请求也被并入了 `compute_positions` 的 sparse KV 分配/写回路径。
- 对没有 `recompute_idx`、没有 `fusionrag_plan` 的普通请求，这条路径不该启用；它们应继续走旧的连续 tail prefill 路径。
- 在 `page_size > 1` 时，radix/page 对齐会导致第二次请求常见为“命中 N-1 个 token，再新算 1 个 tail token”，如果这个 tail token 被按 sparse 语义错误处理，就会出现 cache pollution / 输出乱码。

本轮已改的本地代码：
- `python/sglang/srt/managers/schedule_batch.py`
  - 只有请求真的带 `recompute_idx` 或 `fusionrag_plan` 时，才启用 sparse `compute_positions`。
  - 普通请求把 `self.compute_positions` 设为 `None`。
- `python/sglang/srt/mem_cache/common.py`
  - `alloc_for_extend()` 新增普通分支：
  - 当 `compute_positions is None` 时，直接恢复旧的连续 KV 分配与连续写回逻辑。
  - 只有真正需要 recompute/FusionRAG 的请求才走 sparse 分配。

补充说明：
- sa1 评测侧 Qwen prompt 已恢复到原来的 `<|im_start|>...` 模板，但该修改发生在评测仓库，不在本仓库 git 历史内。
- 这次修改目前只在本地仓库完成并提交，尚未在当前环境里直接覆盖到 sa1 运行目录。

## 4. 已验证状态
已确认：
- `test_fusionrag_params.py` 本地通过。
- `.ready` 并发删除竞态已有最小回归测试覆盖。
- sa1 上已同步关键修复文件，并通过 `py_compile` 级验证。
- `run_demo_sa1.sh` 所覆盖的 A/B/C/D/E 路径已有前序 agent 的通过记录。
- 本轮两处本地修改已通过：
  - `python3 -m py_compile python/sglang/srt/managers/schedule_batch.py python/sglang/srt/mem_cache/common.py`
  - `git diff --check`

未完全闭环：
- sa1 当前 Python 环境缺少 `pytest`，部分回归测试不能直接在远端运行。
- sa1 工作树是直接覆盖同步，不是干净 git 历史；远端 `git status` 会显示已修改文件。
- 当前环境没有挂载 `/mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline`，因此本轮没法直接在这里替 sa1 重启服务并复跑在线最小复现。

本轮已验证过的推断：
- 普通请求误走 sparse `compute_positions` 路径，足以解释“性能回归 + cache hit 后输出漂移”这两个症状为何会同时出现。
- `radix_cache.py` 的 page 对齐语义可以解释为什么复现里经常是第二次命中 `N-1` 个 token，而不是 `N` 个 token。

本轮尚未验证、但明确值得继续探究的点：
- `77afa1a6f` 应用到 sa1 服务后，是否能消除“同 prompt 第二次请求输出漂移/乱码”。
- 如果污染依旧存在，下一嫌疑点是 `radix_cache.py` / `hiradix_cache.py` 在 `cache_unfinished_req` / `cache_finished_req` 里的 page-aligned tail 处理。
- 还没有重新做性能对比，暂时不能断言本轮修复是否已显著恢复吞吐/时延。

## 5. 当前主要风险
1. 性能回归
- 现状：重构后性能明显落后于旧分支，说明 scheduler / prefill / recompute / cache 装载路径仍可能有额外开销或错误预算。
- 重点怀疑点：
  - 普通请求误走 sparse KV 分配/写回。
  - 仍有部分调度预算按 `extend_input_len` 而不是真实 `extend_compute_len` 在工作。

2. 质量回归
- `iter_rag` 数据集在 `rate=0.15` 上出现明显答案质量下降。
- 表现不是简单截断，而是长篇跑偏、实体漂移、无关扩写。
- 当前判断这更像运行时对齐 / cache pollution 问题，不像单纯 prompt 问题。

3. 兼容性边界
- `input logprob + cache hit` 目前仍是稳定降级，不是完整支持。
- 历史旧 cache 目录虽然已做 `v2` 隔离，但重启服务后仍应继续观察是否还有兼容噪音。

## 6. 下一步优先级
1. 先把 `77afa1a6f` 同步到 sa1 并做最小复现
- 重启服务后，连续发两次完全相同的 prompt。
- 分别验证 plain prompt 和 Qwen chat prompt。
- 重点记录：
  - 第一次 `cached_tokens=0` 的输出
  - 第二次 `cached_tokens=device only` 时的输出
  - 两次输出是否仍漂移

2. 如果污染消失，再回头测性能
- 固定端口、日志文件、cache 根目录。
- 对比 no-fusion / raw-only / preprocess 三组样例。
- 输出 `compute_tokens`、`prefill_tokens`、`e2e latency`、P50/P95 对比。

3. 如果污染仍在，继续查 prefix cache 插入/复用
- 重点看 `radix_cache.py`、`hiradix_cache.py`：
  - `cache_unfinished_req`
  - `cache_finished_req`
  - `page_size > 1` 时 page-aligned prefix 与 tail token 的处理
- 核心问题是：树里 page 对齐前缀、`req.prefix_indices` 尾巴、下一次命中后的 fresh tail token 三者是否保持一致。

4. 再审计 scheduler 预算与 prefill 热路径
- 重点看是否还存在把“原始输入长度”误当成“真实 compute 长度”的路径。
- 特别检查 `schedule_policy.py`、`scheduler.py`、prefill batch 组装逻辑。

5. 清理与归档远端变更
- 如果 sa1 验证通过，最好把远端工作树的直接覆盖变更收敛成正式 commit，避免后续继续漂移。

## 7. 常用脚本与路径
关键文档：
- `next-to-do/full.md`
- `next-to-do/develop.md`
- `next-to-do/dev_plan.md`
- `next-to-do/demo.md`
- `next-to-do/HANDOFF.md`

常用脚本：
- `next-to-do/run_demo_sa1.sh`
- `next-to-do/quarantine_incompatible_cache.sh`
- `qwen_sglang.sh`

运行环境：
- sa1 服务仓库：`/mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline`
- sa1 评测仓库：`/mnt/data/shm/jybigdata`
- sa1 Python：`/mnt/data/shm/sglang-fusionrag/venv-kt/bin/python3.10`

## 8. 历史文档
旧的 handoff/status/todo 已归档到：
- `next-to-do/archive/2026-04-30/`
- `next-to-do/archive/2026-05-01-doc-cleanup/`
