# vime 共卡模式（colocate）在 Ascend NPU 上的改造设计

以 2026-06-30 跑通的 `qwen35_9b_8card_20260630_061951` 这次训练为参照（脚本
`scripts/run_qwen35_9b_mathrl_npu_8card.sh`，提交 e19530af / d778d829）。这份文档回答四个问题：
为什么要共卡、显存怎么在训练和推理之间腾挪、权重/梯度/激活各自走什么路径、NPU 上相对 CUDA
补了哪些东西以及还欠什么。

配置速记：单机 8×910B3（64GB HBM），Qwen3.5-9B（GDN-hybrid，35% full attention + 65% 线性注意力），
训练侧 Megatron TP4 × PP1 × CP2（纯 DP=1），推理侧 vLLM 4 个 engine × TP2，32k response，
GRPO，vLLM `gpu_memory_utilization=0.7`。

---

## 1. 为什么共卡

RL 训练里训练和推理是严格串行的：rollout 产完数据才能训练，训练完权重同步才能产下一轮。
分卡部署（比如 4 卡 Megatron + 4 卡 vLLM）在这台 8 卡机器上两头都不够：

- 训练侧：9B 参数 bf16 + fp32 梯度在 TP4 下每卡约 13.5GB（见第 4 节的账），再叠 32k 序列的激活，
  4 卡塞不下像样的 batch；
- 推理侧：32k response × 64 并发的 KV cache，4 卡 TP2×2 engine 的 KV 预算直接砍半，
  长序列 decode 吞吐掉得厉害；
- 更根本的：串行流程里分卡意味着任意时刻总有一半卡在看戏。

共卡的思路就是时间片轮转：rollout 阶段 8 卡显存几乎全给 vLLM，训练阶段几乎全给 Megatron，
靠"睡眠/唤醒"在两个状态间切换。切换本身有搬运开销，赌的是它远小于分卡浪费的一半算力。

`--colocate` 打开后（`vime/utils/arguments.py` 的参数后处理）`offload_train` 和
`offload_rollout` 自动置 true，后面的一切围绕这两个开关展开。

## 2. 资源分配：两套进程压在同一批卡上

`create_placement_groups`（`vime/ray/placement_group.py:88`）在 colocate 下只建
8 个 bundle 的 placement group，rollout 不再追加卡，直接复用同一个 PG（`rollout_offset=0`）。
每张卡上住着两个进程：

- **训练 actor**：`MegatronTrainRayActor`，每个按 0.4 张卡向 ray 申报
  （`allocate_train_group`，`num_gpus_per_actor=0.4`）——纯粹是让 ray 允许同卡再塞别人，
  不代表真实显存配比；
- **vLLM engine**：独立的 `vllm serve` 子进程（HTTP server，`VLLM_SERVER_DEV_MODE=1`），
  由 `build_vllm_subprocess_env` 按卡槽设置 `ASCEND_RT_VISIBLE_DEVICES` 拉起，
  sleep/wake/权重更新都走 HTTP endpoint。

两者互不知晓对方的分配器状态，唯一的契约是：**任一时刻只有一方持有大额 HBM**。
谁违约谁 OOM——这次调试踩的最大的坑（假卸载）就是训练侧违约了 13GB，见第 6 节。

## 3. 一轮 rollout 的完整时序

`train.py:70-99` 的主循环，标注每步之后 HBM 里躺着谁：

```
状态: [vLLM 满显存: 权重9GB/卡 + KV~31GB + graph]  [Megatron: 已睡眠, 参数在CPU]

 1. rollout_manager.generate(rollout_id)
      → vLLM 8 卡推理, 产出 rollout 数据 (存盘 + 返回 ref)
 2. rollout_manager.offload()
      → 每个 engine: flush_cache; POST /sleep?level=1
        vLLM 权重卸到 host 内存, KV cache/graph 释放     [HBM 几乎清空]
 3. actor.train(rollout_id, data):
      wake_up():  参数 CPU→NPU, 重建通信组                [Megatron 满显存]
      ref/old logprob 前向 → advantages → 逐 micro-batch fwd/bwd → optimizer step
      weights_backuper.backup("actor")  → 最新权重备份到 CPU pinned
      sleep():    销毁通信组, 参数卸 CPU, flat buffer 物理释放  [HBM 几乎清空]
 4. rollout_manager.onload_weights()
      → POST /wake_up?tags=weights      vLLM 旧权重 host→NPU   [vLLM 权重回卡]
        (--rollout-sleep-level 2 时第 2 步不做 host 备份、这里只分配不拷贝, 见 7.2)
 5. actor.update_weights()
      → 训练侧临时重建 gloo 组, 从 CPU 备份分桶转 HF 格式 → NPU 建 IPC handle
        → vLLM worker 经 IPC 原地覆盖旧权重 (零磁盘零网络)
 6. rollout_manager.onload_kv()
      → POST /wake_up?tags=kv_cache,cuda_graph            [回到起点, 下一轮]
```

第 4、5 步看起来绕（先恢复旧权重再覆盖），这是当前实现为了复用 vLLM 的
sleep level=1 语义付出的代价，第 7 节讨论。

步骤 3 内部还有一层嵌套：`save_model` 和 `update_weights` 各自会按需 wake/sleep 或
重建/销毁通信组（`save_model` / `update_weights` 内部），保证"训练态资产只在需要时上卡"。

## 4. 权重、梯度、激活的路径

这是共卡改造的核心，三类张量的待遇完全不同：**权重来回搬、梯度就地毁、激活压根不过夜**。

### 4.1 权重

一份 9B 权重在系统里同时存在四个化身，各有各的生命周期：

| 化身 | 位置 | 精度/布局 | 生命周期 |
|---|---|---|---|
| 训练主体 | NPU，Megatron DDP flat `param_data` | bf16，TP4 切分，~4.5GB/卡 | 训练阶段在卡上；睡眠时 B 模式下物理释放 |
| fp32 master + adam 状态 | CPU | fp32，跟随 optimizer | 常驻 CPU（`--optimizer-cpu-offload` + `--use-precision-aware-optimizer`），从不占 HBM |
| CPU 备份（"actor" tag） | CPU pinned | bf16 分片 | 每轮训练结束 `backup("actor")` 刷新（`train_actor` 尾部），是权重同步的数据源 |
| 推理副本 | vLLM 各 engine | bf16，TP2 切分，~9GB/卡 | sleep 时卸 host，wake 后被 IPC 覆盖为新版本 |

**训练→睡眠**：`MegatronTrainRayActor.sleep()` 先 `destroy_process_groups()`
（HCCL 通信器本身占 HBM），然后 `NPUWeightOffloader.offload()`
（`vime/utils/npu_weight_offloader.py`，storage-resize 已收编进本体）对 DDP flat buffer
做 `untyped_storage().resize_(0)`：grad buffer 直接释放（内容可丢，backward 重填），
param buffer 在 B 模式（`VIME_OFFLOAD_PARAM_BUFFER=1`）下先备份 pinned CPU 再释放。
所有 `param.data`/`main_grad` 都是这两块 flat storage 的 view，缩掉共享 storage 就
一次性释放全部、且保留张量对象 identity——唤醒时 resize 回来 view 自动复活，
不需要任何指针手术。反过来，只换 `param.data` 指针释放不掉被 view 别名的
flat storage，这正是第 6 节坑 1 的假卸载。

**睡眠→权重同步**：主循环在训练 actor 睡着的状态下调 `update_weights()`（`actor.py`）。
colocate 分支不唤醒整个模型，只 `reload_process_groups()` 建出 gloo 组，然后
`UpdateWeightFromTensor.update_weights()`（`update_weight/update_weight_from_tensor.py:283`）：

1. 通知所有 engine `pause_generation` + `flush_cache`；
2. `weights_getter` 拿到 CPU 备份（"actor" tag，已是 HF 全局命名——`--megatron-to-hf-mode raw`）；
3. 按 bucket 流式处理：CPU→NPU、`reduce_tensor()` 生成 NPU IPC handle、pickle 后
   gloo `all_gather_object` 聚到每个 engine 槽位的 leader rank（IPC handle 是 host 对象，
   HCCL 传不了）、由 leader 经 ray 调 engine 的 collective_rpc，vLLM worker 反序列化 handle
   直接读训练进程显存、`weight_loader` 拷入自己的参数；
4. 每个 bucket 传完立刻释放 NPU 临时张量 + `ipc_collect()`，峰值显存只有一个 bucket 大小。

同机同卡进程间 IPC 直传，不落盘不走网卡，这是 colocate 相对分卡（NCCL/HCCL broadcast，
`UpdateWeightFromDistributed`）在权重同步上白赚的收益。

### 4.2 梯度

梯度从不离开 NPU，也从不被保存——它的路径是"生成、聚合、消费、销毁"四步：

1. **生成**：backward 中逐参数写入 `main_grad`，落在 Megatron DDP 的 fp32 flat
   `grad_data`（TP4 下 9e9×4B/4 ≈ 9GB/卡；加上 bf16 param 4.5GB 就是代码注释里
   "~13GB/rank @ 9B/TP4" 的来源，见 `npu_weight_offloader.py` docstring）；
2. **聚合**：bucket 写满即在 DP×CP 组内 reduce-scatter（CP2 下就是配对的两张卡，
   序列切半导致的梯度差在这里对齐）；
3. **消费**：optimizer step 在 CPU 上做（梯度 d2h、更新后的 fp32 master 转 bf16 h2d，
   `--overlap-cpu-optimizer-d2h-h2d` 让搬运和计算流水）；
4. **销毁**：本轮训练结束 `sleep()` 里 `grad_data.untyped_storage().resize_(0)`
   （`NPUWeightOffloader.offload`，grad 不备份），9GB/卡即刻归还。下一轮 wake_up 时
   storage resize 回来并清零——`main_grad` 这些 view 因对象 identity 未变自动复活。

敢直接销毁的前提是 RL 的训练节奏：每轮 rollout 的梯度在 optimizer step 后没有任何残余价值，
不存在跨轮梯度累积。

### 4.3 激活

激活是三者中最"短命"的，生命周期被三重压缩，从不参与 offload：

- **重计算**：`--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`，
  前向只留每层入口张量，backward 逐层重算。32 层 × 32k token 的完整激活图在 64GB 卡上
  存不下，这是拿 ~30% 算力换显存的标准交易；
- **序列切分**：CP2 把每条序列 zigzag 切两半（`cp_utils.py` 的 `slice_with_cp`），
  单卡激活减半。full attention 层靠 MindSpeed ring attention 分块交换 KV
  （send-recv-overlap 通信组），GDN 线性注意力层按独立的 `cu_seqlens_gdn` 处理
  各自的循环状态——两类层对序列切分的约定不同，是 e19530af 里 `data.py`
  三套 cu_seqlens 并存的原因；
- **分块 logits**：`--log-probs-chunk-size 1024`。词表 248320 的 lm_head 输出若整条序列
  一次算，40960 token × 248320 × 4B ≈ 40GB，必炸；分 1024 token 一块流式算 log_prob。

micro-batch 的组织交给 `--use-dynamic-batch-size --max-tokens-per-gpu 40960`：按 token
总量装箱而不是按样本数，长短混杂的 RL 样本不会让某张卡激活爆掉。ref/old log_prob
的前向（`compute_log_prob`）走同样的激活纪律，只是没有 backward。

训练进程的分配器开 `expandable_segments:True`（脚本 env），减少反复 wake/sleep
带来的碎片——注意 vLLM 子进程必须把这个 env 剥掉，见下一节。

## 5. NPU 相对 CUDA 路线补了什么

上游（CUDA）的共卡是两个现成组件拼出来的，NPU 上这两个组件一个没有、一个有约束：

| 环节 | CUDA 路线 | NPU 现状 | vime 的替代方案 |
|---|---|---|---|
| 训练侧睡眠 | `torch_memory_saver`：LD_PRELOAD 拦截 cudaMalloc 走 cuMem VMM，pause/resume 只解除/恢复物理页映射，虚拟地址不变，近零拷贝 | 无 LD_PRELOAD 等价物，torch_npu 无 VMM hook | `NPUWeightOffloader`（`vime/utils/npu_weight_offloader.py`）：flat buffer `storage().resize_(0)` 真释放，param 备份 pinned CPU（B 模式）；每轮全量 CPU↔NPU 拷贝，靠带宽硬扛 |
| 推理侧睡眠 | vLLM CuMemAllocator sleep mode | vllm-ascend CaMemAllocator，可用但有两个坑 | sleep 默认 level=1（`--rollout-sleep-level 2` 可选，见 7.2）；vLLM 子进程环境剥离 `PYTORCH_NPU_ALLOC_CONF`（`build_vllm_subprocess_env`） |
| 权重 IPC | CUDA IPC handle | NPU IPC 可用，但要求新驱动 | 驱动 26.0.rc1 起 IPC 权重同步正常（24.x 上此路不通，是当初迁移到 mynpu2 的原因） |

vllm-ascend 的两个坑具体是：

- **level=2 丢 `weight_loader`**：level=2 睡眠丢弃权重对象，wake 后重建的是裸
  tensor，丢了 `weight_loader` 属性，下一次 IPC 权重更新直接
  `'Parameter' object has no attribute 'weight_loader'`。level=1 保留参数对象、
  权重卸 host，HBM 释放效果一样，所以历史上 sleep 写死 level=1。现在 worker
  extension 在 `start_weight_update` 里做 VERL 式属性快照/恢复
  （`_capture_vllm_param_attrs` / `_restore_vllm_param_attrs`），把这个坑兜住了，
  level=2 经 `--rollout-sleep-level 2` 开放（默认仍 1，见 7.2）；
- **expandable_segments 冲突**：CaMemAllocator 的内存池断言容不下 expandable_segments。
  训练进程要开（抗碎片）、vLLM 子进程必须关，所以在拉子进程时清掉继承的 env，
  让 vllm-ascend 按自己的 sleep-mode 感知逻辑决定。

另一个 CUDA 上不存在的环节是**通信组的销毁与重建**（`vime/utils/reloadable_process_group.py`）：
HCCL 通信器占用 HBM，训练睡眠期间不能留。`monkey_patch_torch_dist()` 在启动时包住
`dist.new_group`，记录每一次建组的参数；`destroy_process_groups()` 全部拆掉，
`reload_process_groups()` 按记录重放。MindSpeed CP ring 的三个窗口组也走 `new_group`，
所以补建一次之后（见坑 3）每轮重建都会自动带上。

## 6. 踩过的坑（按发现顺序）

这些坑构成了本次改造大部分的实际工作量，记下来防止重蹈：

1. **假卸载**。最初的 `_release_ddp_buffers` 把 `param_data`/`grad_data` 换成空 tensor
   自以为释放了，但 Megatron 每个 bucket view 都别名着同一块 flat storage，Python 层
   换引用根本释放不了物理页。表现为：offload 后 `npu-smi` 占用纹丝不动，vLLM wake_up
   分配 31GB KV cache 时 `aclrtMallocPhysical` OOM。修法是 verl 风格的
   `untyped_storage().resize_(0)`——绕过所有别名直接把 storage 缩零
   （方案史见 `vime/utils/npu_weight_offloader.py` 模块 docstring）。
2. **sitecustomize 注入时机**。storage-resize 补丁最初想通过 `PYTHONPATH` 里的
   sitecustomize 在解释器启动时打，结果 import 时 torch 还没初始化完，报
   `operator prims::sum does not exist`；调试期改到 actor `init()` 里、torch/torch_npu
   完全就绪后再 patch。这套运行时注入本就是"不改源码"的过渡形态，现已整体收编进
   `NPUWeightOffloader` 本体（见 7.1），patch 机制随之退役。
3. **CP ring 通信组缺失**。vime 直接驱动 Megatron core 的 `GPTModel.forward`，不走
   MindSpeed 的 forward wrapper；而 `mpu.initialize_model_parallel` 在 `init(args)` 里
   先跑，MindSpeed 的 `repatch(args)` 后跑——它包在 initialize_model_parallel 上的
   ring 窗口组建组逻辑永远没机会执行。CP>1 时 full attention 层一进 ring 路径就崩
   `Context parallel ranks for ring intra window not initialized`。修法：repatch 之后
   手动补建 send_recv_overlap / hybrid_cp / double_ring 三组（`actor.py:81-96`）。
   这个 crash 起初被误判为"storage-resize 补丁导致多 step 死锁"，浪费了不少排查时间——
   两个改动同时上车时，先各自单独回归。
4. **三套 cu_seqlens 约定打架**。ring attention 要 CP-local、不带前导 0 的
   `cu_seqlens_q`（带 0 会出零长度段，触发 161001）；RoPE 的 THD 位置编码要原始长度、
   带前导 0 的 `cu_seqlens_q_padded`；GDN 的负载均衡还原要原始长度带 0 的
   `cu_seqlens_gdn`。Megatron core 一个都不填，全部在 `data.py` 里补齐并分流到
   三个消费者（e19530af 提交信息有完整对照）。
5. **异步下发队列导致 backward NaN**。Ascend 的 `TASK_QUEUE_ENABLE`（默认开）异步下发
   算子，GDN/ring 这类自定义算子链路下数值不稳定，训练在 micro-batch ~7 稳定崩
   `found NaN in local grad norm`。同一份 rollout 数据关掉队列（`TASK_QUEUE_ENABLE=0`）
   后 16+ micro-batch 无 NaN。已作为 NPU 自定义算子训练的必带 env 写进脚本（d778d829）。

## 7. 短板处理进展与后续工作

### 已处理（2026-07-16，待 NPU 环境恢复后回归）

1. **`storage_resize_hook` 收编进仓库——完成**。曾经的状态：`actor.py` 裸
   `import storage_resize_hook`，靠训练脚本把 mynpu2 上的 `scripts/npu_mem_offload`
   塞进 PYTHONPATH；换机器 import 失败会静默 fallback 到假卸载版 offloader（wake_up OOM
   复活），且 fallback 自己还埋着一段引用未定义变量的死代码（`_release_ddp_buffers`
   残留，触发即 NameError）。现在 hook 在 mynpu2 上验证过的行为（纯 flat-buffer
   resize + param pinned 备份，A/B 模式）就是 `NPUWeightOffloader` 本体
   （`vime/utils/npu_weight_offloader.py` 整体重写），假卸载路径、死代码、actor 里的
   patch 段、脚本里的 PYTHONPATH 注入全部移除；旧 hook 目录（`npu_mem_offload/`）
   保留 README 并标注弃用。与运行时 patch 的一个行为差异是有意的：hook 出错时
   try/except 全吞（patch 场景的防御），收编版不吞——出错就该炸，静默降级正是
   假卸载复活的温床。
2. **onload_weights 的多余全量搬运——已落地为可选开关**。时序第 4 步把 vLLM 的
   **旧**权重从 host 搬回 NPU（每卡 ~9GB h2d），第 5 步立刻整份覆盖。现加
   `--rollout-sleep-level`（默认 1，行为与之前完全一致）：设为 2 时睡眠直接丢弃权重
   （连 host 备份也省掉），wake 只分配不拷贝，靠随后的 IPC 全量覆盖填充。曾经堵死
   level=2 的 weight_loader 丢失问题，worker extension 的属性快照/恢复机制
   （`start_weight_update` 里的 VERL 式 re-patch，`update_weight_from_tensor.py`）
   已经兜住。两条保护：非 updatable server（固定 teacher/RM 引擎，没有 IPC 覆盖）
   无视该参数强制 level 1；引擎故障恢复路径同样强制 level 1（刚从盘加载的权重是
   唯一正确副本）。**未经 NPU 端到端回归，默认值保持 1**；mynpu2 恢复后跑一轮
   level=2 的 smoke 再考虑默认启用。
3. **两处小额清理——完成**。脚本头注释 "TP2 × DP4" 已改为实际的 "TP4 × CP2"；
   NPU 路径无效的 `--train-memory-margin-bytes`（只喂 torch_memory_saver）已从
   NPU 脚本删除。

### 仍欠着的

4. **全量拷贝式睡眠的开销上限**。CUDA 的 VMM pause/resume 近零成本，NPU 现在每轮
   sleep+wake 要搬 ~2×13.5GB/卡。当前 9B 规模下占比可接受，模型再大或 rollout 变短后
   这项会成为主要轮转开销。值得探的方向：基于 `aclrtMallocPhysical`/camem 做 NPU 版
   torch_memory_saver（vllm-ascend 的 CaMemAllocator 已证明这条路在 NPU 上通）。
5. **host 内存没有预算管理**。CPU 上同时躺着：fp32 master + adam 状态、每 rank 的
   bf16 pinned 备份（"actor"，开 ref/old_actor 还要翻倍）、训练睡眠期 B 模式的
   param flat 备份、vLLM level=1 睡眠期的权重副本（`--rollout-sleep-level 2` 验证
   通过后这份可省）。9B 规模粗算 150GB+，都是隐式分配，逼近物理内存时的失败模式是
   pin 不上或 OOM killer，没有任何预警。至少应该启动时把账算出来打条日志。

## 附：命令参数 → 机制对照

那条命令里与共卡直接相关的参数，和它们各自钩住的代码：

| 参数 | 作用点 |
|---|---|
| `--colocate` | PG 复用（`placement_group.py` 的 `create_placement_groups`）、自动置 offload_train/rollout、权重同步选 `UpdateWeightFromTensor` |
| `--rollout-num-gpus-per-engine 2` | 4 个 vLLM engine 各 TP2，engine 槽位决定 IPC gather 组的划分 |
| `--vllm-enable-sleep-mode` + `--vllm-gpu-memory-utilization 0.7` | vLLM 可睡眠 + 醒时占 70% HBM（权重 9GB + KV ~31GB/卡） |
| `--optimizer-cpu-offload --use-precision-aware-optimizer --overlap-cpu-optimizer-d2h-h2d` | optimizer 状态常驻 CPU，step 时 d2h/h2d 与计算重叠 |
| `--megatron-to-hf-mode raw` | CPU 备份直接以 HF 全局名存，权重同步免 rename |
| `--recompute-granularity full` 等三件套 | 激活重计算（4.3 节） |
| `--context-parallel-size 2` | 序列切半 + ring attention（依赖 actor.py 的补组） |
| `--rollout-sleep-level` | vLLM 睡眠级别：1（默认）host 备份+拷回；2 丢弃权重、wake 空壳等 IPC 填充（7.2 节，NPU 待回归） |
| env `VIME_OFFLOAD_PARAM_BUFFER=1` | 睡眠时连 param flat buffer 一起 resize_(0)（B 模式） |
| env `TASK_QUEUE_ENABLE=0` | 关闭 Ascend 异步下发队列，GDN/ring backward NaN 的修复 |
