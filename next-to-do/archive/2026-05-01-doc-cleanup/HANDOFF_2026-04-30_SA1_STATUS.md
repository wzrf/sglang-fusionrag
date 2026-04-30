# FusionRAG 接手交接（2026-04-30）

## 1) 当前完成进度

- 本地主仓库 `sglang-fusionrag` 已有关键提交（本地分支 `fusionrag_unified_pipeline`）：
  - `82a26c515`：load 时按 `chunk_plan.start_token` 做 RoPE 位置重映射
  - `d909382d9`：DeepSeek kv_gen 保存时做 RoPE local-0 校准
  - `f51557ad0`：Qwen2 kv_gen 保存时做 RoPE local-0 校准
  - `0399107a6`：修复并发写 cache 的 tmp 文件竞争（唯一 tmp 名 + FileNotFound 容忍）
- `sa1` 的 `sglang-fusionrag-fusionrag_unified_pipeline` 之前落后于本地提交，运行时出现新崩溃点。
- 新增修复（已在本地代码补齐）：
  - `scheduler.py`：chunked round 调用 `init_next_round_input` 时补传 `tree_cache_hicache/tree_cache_fusionrag`
  - `schedule_batch.py`：`is_kv_gen` 分支增加 `tree_cache_fusionrag is None` 防御兜底
- `jybigdata` 评测端分支：
  - `feat/adapt-fusionrag-v2-interface-20260430`
  - 已适配 `/generate + fusionrag_params(v2)`，并补了 `run_gen` 异常返回，避免 `NoneType unpack`。

## 2) 当前主要问题（可复现）

### A. 服务端已修崩溃（但需确认远端代码同步）

- 历史崩溃：
  - `AttributeError: 'NoneType' object has no attribute 'match_prefix'`
  - 触发链：`chunked_req.init_next_round_input()` 漏传 cache 对象
- 本地已修，远端需确认一致版本后再跑。

### B. 输出“乱码/重复串/空回答”仍存在

- 现象：结果中出现连续数字（如大量 `195555...`）、机械重复片段或空输出。
- 关键结论：不仅 FusionRAG 路径，`30004` 上纯 `/generate` 也可复现异常风格输出（非正常 QA）。
- 说明问题至少包含两层：
  1. 评测输入构造（超长拼接前缀 + completion 形式）对 `Qwen2.5-0.5B-Instruct` 非稳态；
  2. FusionRAG chunk/preprocess 路径可能进一步放大偏差（尤其 chunk 与 prefix 语义不一致时）。

### C. cache 兼容警告大量出现

- 日志有大量：
  - `[FusionRAG] skip incompatible chunk cache ... layout_or_load_error ... shape invalid`
- 这是历史缓存格式/布局与当前服务不兼容造成，会导致回退到重算，影响稳定性和性能判断。

## 3) 潜在高风险点（建议下个 agent 优先排查）

1. `jybigdata/sources/iter_rag/rag/agent/sglang_kvcache.py`
   - `run_one_question_sglang_preprocess` 的 preprocess 构建与最终 `chunk_plan` 语义是否严格一致。
   - `prompt/prefix_prompt/prompt_list/prefix_prompt_list` 参数语义存在重叠和隐式约定，易错。
2. Qwen 模板/停止条件
   - 当前使用 completion 形式（手拼 `<|im_start|>...`），需确认是否与服务端 tokenizer/chat template 行为一致。
   - 需做 A/B：`use_fusion_rag=False`、`raw-only`、`preprocess` 分层验证输出退化点。
3. 线上实例混跑
   - `30003` 与 `30004` 存在不同实验实例，日志与结果容易串线。
   - 必须固定端口、固定日志文件、固定 cache 根目录后再做回归。

## 4) 建议接手执行顺序（主线）

1. 先清场并固定单实例（端口、日志、cache 路径）。
2. 在同一实例上跑三组最小样本：
   - no-fusion
   - fusion + raw-only
   - fusion + preprocess
3. 若 no-fusion 仍异常，先修评测 prompt 组装与生成参数；若 no-fusion 正常再追 FusionRAG 路径。
4. 清理或隔离旧 cache，再观察是否还出现 layout 不兼容。

## 5) 关键路径

- 本地主仓库：
  - `/Users/hming/code/sglang-fusionrag`
- 本地交接文档：
  - `next-to-do/HANDOFF_2026-04-30_SA1_STATUS.md`（本文件）
- sa1 服务仓库：
  - `/mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline`
- sa1 评测仓库：
  - `/mnt/data/shm/jybigdata`
- sa1 服务日志：
  - `/mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline/sg_e2e_30004.log`

