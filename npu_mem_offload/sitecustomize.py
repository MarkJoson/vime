"""storage-resize 注入器(放 /root/probe_sresize,PYTHONPATH 最前):每个 ray actor 进程自动 import,
后台线程轮询,等 vime.utils.npu_weight_offloader 被 import 后 patch NPUWeightOffloader.offload/onload
为 verl 式 storage().resize_(0) 释放。只 patch offloader,不碰 torch.zeros/buffer/CaMemAllocator,
所以不污染 dynamo、不需去 expandable(但 storage-resize 物理释放需要 caching allocator 配合,见脚本)。"""
import sys
import threading
import time


def _loop():
    try:
        import storage_resize_hook
    except Exception as e:
        print("[sitecustomize_sresize] import storage_resize_hook err:", e, flush=True)
        return
    for _ in range(12000):  # ~20min @ 0.1s
        m = sys.modules.get("vime.utils.npu_weight_offloader")
        if m is not None and hasattr(m, "NPUWeightOffloader"):
            try:
                storage_resize_hook.patch_offloader(m.NPUWeightOffloader)
            except Exception as e:
                print("[sitecustomize_sresize] patch err:", e, flush=True)
            return
        time.sleep(0.1)


threading.Thread(target=_loop, daemon=True).start()
