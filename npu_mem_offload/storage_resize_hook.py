"""storage-resize offload hook(verl 式,A/B 可切换,不改 vime/Megatron 源码):
patch NPUWeightOffloader.offload/onload,用 `storage().resize_(0)+empty_cache` 真正释放
Megatron DDP flat buffer —— 绕开 vime offloader「换属性(param_data=empty)」释放不掉的 bug
(bucket view 仍 alias 旧 storage)。resize 缩的是所有 view 共享的那块 storage,一次释放、保留对象 identity。

环境变量 VIME_OFFLOAD_PARAM_BUFFER:
  "1" → B:param_data + grad_data 都释放(param 备份 pinned CPU,onload copy 回);rollout 残留≈0。
  其他 → A(默认):只释放 grad_data(可丢,onload resize 回 + zero,backward 重填);param 常驻保 IPC。
"""
import functools
import os

import torch

_OFFLOAD_PARAM = os.environ.get("VIME_OFFLOAD_PARAM_BUFFER", "0") == "1"
_LOG = "/root/sresize_{}.log".format(os.getpid())
_saved = {}  # (id(buf), attr) -> (orig_storage_size_bytes, cpu_backup_or_None)


def _w(t):
    try:
        with open(_LOG, "a") as f:
            f.write(t + "\n")
            f.flush()
    except Exception:
        pass
    try:
        print(t, flush=True)
    except Exception:
        pass


def _get_buffers(model):
    """收集 Megatron DDP 的 _ParamAndGradBuffer(ddp.buffers + expert_parallel_buffers)。"""
    models = model if isinstance(model, (list, tuple)) else [model]
    bufs = []
    seen = set()
    for m in models:
        if m is None:
            continue
        for la in ("buffers", "expert_parallel_buffers"):
            v = getattr(m, la, None)
            if isinstance(v, (list, tuple)):
                for b in v:
                    if b is not None and id(b) not in seen:
                        seen.add(id(b))
                        bufs.append(b)
    return bufs


def patch_offloader(cls):
    if getattr(cls, "_sresize_patched", False):
        return
    cls._sresize_patched = True

    @functools.wraps(cls.offload)
    def offload(self, model, *a, **k):
        try:
            free0, _ = torch.npu.mem_get_info()
            attrs = (["grad_data", "param_data"] if _OFFLOAD_PARAM else ["grad_data"])
            n = 0
            for buf in _get_buffers(model):
                for attr in attrs:
                    t = getattr(buf, attr, None)
                    if not (isinstance(t, torch.Tensor) and t.numel() > 0):
                        continue
                    st = t.untyped_storage()
                    key = (id(buf), attr)
                    if attr == "param_data":  # 权重不能丢 → 备份 pinned CPU
                        _saved[key] = (st.size(), t.detach().to("cpu", copy=True).pin_memory())
                    else:  # grad 可丢 → 不备份
                        _saved[key] = (st.size(), None)
                    st.resize_(0)
                    n += 1
            torch.npu.empty_cache()
            free1, _ = torch.npu.mem_get_info()
            _w("[sresize] OFFLOAD ({}): 释放 {} buffer, 物理 +{:.3f}GB free".format(
                "param+grad" if _OFFLOAD_PARAM else "grad-only", n, (free1 - free0) / 1e9))
        except Exception as e:
            _w("[sresize] offload EXC: {}".format(e))
        return 0

    @functools.wraps(cls.onload)
    def onload(self, model, *a, **k):
        try:
            for buf in _get_buffers(model):
                for attr in ("grad_data", "param_data"):
                    t = getattr(buf, attr, None)
                    if t is None:
                        continue
                    key = (id(buf), attr)
                    if key not in _saved:
                        continue
                    size, cpu = _saved.pop(key)
                    st = t.untyped_storage()
                    st.resize_(size)
                    if cpu is not None:  # param:恢复权重内容
                        t.copy_(cpu)
                    else:  # grad:清零(下次 backward 重填前)
                        t.zero_()
            _w("[sresize] ONLOAD done")
        except Exception as e:
            _w("[sresize] onload EXC: {}".format(e))
        return 0

    cls.offload, cls.onload = offload, onload
    _w("[sresize] patched NPUWeightOffloader.offload/onload pid={} OFFLOAD_PARAM={}".format(
        os.getpid(), _OFFLOAD_PARAM))
