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
6. single-pass prefill 关键口径修复：
   - `fusionrag_prefill_tokens` 改为 `total - radix_hit - chunk_hit` 统计
   - `kv_gen` 路径补齐 chunk hit 统计字段（`fusionrag_chunk_hit_tokens`）
   - C/D 强断言恢复并在 sa1 实跑通过
7. single-pass 稀疏写路径（`compute_positions != None`）接入 Triton：
   - 新增 `write_req_to_token_pool_sparse_triton`
   - 保留 fallback：不支持 Triton 时走 torch 写入
   - sa1（30004）回归 A/B/C/D/E1/E2 全通过
8. 验收脚本指标补强：
   - `run_demo_sa1.sh` 已增加 C/D/E2 的 `fusionrag_compute_tokens` 断言
   - sa1（30004）实跑通过

## PARTIAL（需要补稳定性/指标口径）
1. 性能基线对比尚未固化
   - 需要固定压测样例，输出 single-pass 路径开/关的 `compute_tokens/prefill_tokens/e2e_latency` 对比表（均值/P50/P95）

## TODO（下一位重点）
1. preprocess cache `unsupport layout detected` 告警
   - 已改：v2 新 cache 隔离到 `raw_kv_cache/v2` 与 `preprocess_kv_cache/v2`；metadata 写入 `fusionrag_cache_format_version/kv_shape/layer_num`
   - 已改：加载不兼容 tensor 时结构化 warning 并 skip，不再打印泛化的 `unsupport layout detected`
   - 已改：落盘采用临时文件 + `os.replace` 原子提交，并增加 `.ready` 标记；启动只加载 ready 条目，规避构建/读取竞态下的半成品读取
   - 待确认：需要在 sa1 重启服务观察启动日志
2. demo E 并发一致性（E1/E2）
   - 已改：`run_demo_sa1.sh` 增加 E1 并发读取同一 chunk、E2 构建/读取竞争
   - 已改：E2 新增 `E2_READ_DELAY_SEC`（默认 0.15s）用于稳定制造“构建中并发读取”窗口，必要时可按机器负载调大
   - 判据：E1 两个请求均成功且 chunk hit；E2 构建和读取均成功，读取端 fallback reason 只允许空、`chunk_miss` 或 `metadata_mismatch`
   - 现状：sa1 已实跑通过（E1 双请求都 `chunk_lookup_hits>=1` 且 fallback 为空；E2 构建/读取均成功）
3. single-pass prefill（一次性前向）收口与性能验证
   - 实现状态：主路径已打通并通过 sa1 回归（含 sparse Triton 写入）
   - 待完成：补基线数据与稳定性报告，作为合并门禁
4. `input logprob + cache hit` 是否需要支持 （已确定暂时不需要）
   - 现状：稳定降级 `input_logprob_cache_hit_unsupported`
   - 需产品确认：接受降级 or 需要实现“cache hit 也返回 input logprob”
