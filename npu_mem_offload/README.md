# NPU 显存 offload — storage-resize 方案（运行时注入，不改 vime 核心）

> **已收编，此目录仅作历史参考。** storage-resize 的 offload/onload 逻辑已合入
> `vime/utils/npu_weight_offloader.py`（连同假卸载死代码的清理），不再需要
> PYTHONPATH 注入，`sitecustomize.py`/`storage_resize_hook.py` 均已废弃。
> A/B 模式语义（`VIME_OFFLOAD_PARAM_BUFFER`）与下文一致。

修复 `NPUWeightOffloader` 的「假卸载」问题：原实现把 `param.data` 换成 CPU 指针，但
Megatron DDP 的 `_ParamAndGradBuffer` flat buffer（`param_data` bf16 + `grad_data` fp32）
被 bucket view 多重引用、**没有真正释放**，rollout 阶段 NPU HBM 一点没省。

本工具用 verl 式 **`storage().resize_(0) + empty_cache()`** 原地缩容那块共享 storage
（所有 bucket view 一次失效、保留张量 identity），真正把物理 HBM 还给驱动。

## 用法（PYTHONPATH 注入，不改 vime/Megatron 源码）

```bash
export PYTHONPATH="/workspace/vime/scripts/npu_mem_offload:$PYTHONPATH"
```

每个 ray actor 进程启动时自动 import `sitecustomize.py`，后台轮询到
`vime.utils.npu_weight_offloader.NPUWeightOffloader` 后 patch 其 `offload/onload`。

## A/B 模式（环境变量 `VIME_OFFLOAD_PARAM_BUFFER`）

| 值 | 模式 | 行为 | rollout 残留 |
|---|---|---|---|
| `0`（默认） | A | 只卸 `grad_data`（内容可丢，onload `resize` 回 + `zero`，backward 重填）；`param_data` 常驻保 IPC | 只剩 param |
| `1` | B | `param_data`+`grad_data` 全卸（param 备份 pinned CPU，onload copy 回）；rollout actor 让出全部权重显存 | ≈0（纯 framework） |

B 依赖 vLLM `load_weights` 是 copy（rollout 用自己的权重副本，不依赖 actor param 物理页）——
已实测成立。

## 实测（Qwen3.5-0.8B, 4×910B3, colocate）

| | 原 offloader | A（仅 grad） | B（全卸） |
|---|---|---|---|
| rollout `allocated` | 4.5 GB（假卸载） | 1.4 GB | **0.0 GB** |
| 物理释放 | 0 | grad 3.0 GB | param+grad 4.53 GB |
| 训练正确 | — | ✓ | ✓（reward 0.046，截断率与 A 一致） |

9B（TP2，per-rank）预期：rollout 残留 26.86GB → A 9GB / B ≈1.4GB。
