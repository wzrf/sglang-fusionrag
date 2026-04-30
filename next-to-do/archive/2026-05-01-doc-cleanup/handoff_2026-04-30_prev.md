# Handoff 给下一位 Agent（对齐版）

## 1. 你要做什么
在复用现有代码前提下，完成 FusionRAG 的统一编排重构：
- `generate` 与 `kv_gen` 共用一条执行管线
- `radix + chunk + recompute + prefill + save` 行为符合 `full.md`

## 2. 必读顺序
1. `full.md`（唯一权威）
2. `dev_plan.md`（阶段计划）
3. `demo.md`（测试标准）

## 3. 强约束
1. 不要新增与 `full.md` 冲突的字段名或语义。
2. 一律使用：`chunk_plan`、`recompute_token_idx`、`recompute_token_idx_local`。
3. chunk cache 只读；recompute 只改 request-local。
4. 默认保存策略必须与 `full.md` 一致。

## 4. 交付要求
1. 代码改动 PR（按阶段提交）
2. 测试结果覆盖 `demo.md` 的 P1~P5 与 A~D
3. 若需改接口，必须先同步更新 `full.md`

## 5. 文件位置
- 权威需求：`/mnt/data/shm/sglang-fusionrag/next-to-do/full.md`
- 开发计划：`/mnt/data/shm/sglang-fusionrag/next-to-do/dev_plan.md`
- 测试标准：`/mnt/data/shm/sglang-fusionrag/next-to-do/demo.md`

## 6. 当前代码状态（2026-04-30）
分支与同步
- 本地：`/Users/hming/code/sglang-fusionrag` 分支 `fusionrag_unified_pipeline`
- sa1 正式 worktree：`/mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline` 分支 `fusionrag_unified_pipeline`
- 两边 HEAD 当前一致：`e5c8242cb7518df7c089ad6050133ad12cdc5f5f`

已实现的关键能力（对齐 full.md 的方向，但未宣称完全完成）
- 新 schema 解析/校验与 request-local planner：`python/sglang/srt/fusionrag_params.py`、`python/sglang/srt/fusionrag_plan.py`
- ingress 归一化接入：`python/sglang/srt/managers/tokenizer_manager.py`
- 执行链桥接：`python/sglang/srt/managers/schedule_batch.py`（携带 plan，chunk 命中结果、recompute 映射、fallback reason、统计字段）
- chunk 命中改为优先按 `chunk_plan` descriptor（`doc_id/chunk_id/doc_hash/cache_variant`）查找：`python/sglang/srt/mem_cache/fusionrag_cache.py`
- 稀疏写回与 compute_positions：`python/sglang/srt/mem_cache/common.py`
- logprob 元数据语义修正（compute len vs input len 分离）：`python/sglang/srt/layers/logits_processor.py` 及 forward batch 相关文件
- 运行级 smoke：`python/sglang/test/test_fusionrag_runtime_smoke.py`

已知限制/降级策略
- `input logprob + cache hit` 语义当前不成立（cache-hit token 没有 logits），会触发 scheduler 断言；现阶段做法是对需要 input logprob 的请求跳过 FusionRAG chunk 复用，并返回 fallback reason：`input_logprob_cache_hit_unsupported`。

## 7. 下一步开发与测试目标（优先级顺序）
1. 在 sa1 上跑通运行级 smoke，形成可验收闭环
- 环境：必须用 `venv-kt`（和 `qwen_sglang.sh` 一致风格）
- 命令（正式 worktree 上跑）：
  `cd /mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline && CUDA_VISIBLE_DEVICES=4 FUSIONRAG_SMOKE_HICACHE_SIZE=32 PYTHONPATH=/mnt/data/shm/sglang-fusionrag-fusionrag_unified_pipeline/python /mnt/data/shm/sglang-fusionrag/venv-kt/bin/python3.10 python/sglang/test/test_fusionrag_runtime_smoke.py`
- 期望：output-only logprob 请求允许 chunk hit；input-logprob 请求必须稳定降级且不崩溃。

2. 把“哪些组合不支持”固化为稳定的 fallback reason，并在 `demo.md` 对应 case 上补验证
- 重点：`input logprob + chunk hit`、`input logprob + recompute`、`non_contiguous_chunk_span`

3. 清理/隔离旧 cache artifact 的噪音（非阻塞，但影响可读性）
- 当前可能会看到旧格式 cache 的 layout mismatch 警告；需要决定跳过策略或版本化 key/path。
