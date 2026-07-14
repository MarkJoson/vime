import logging
import os
import time
import traceback
from pathlib import Path

import torch

from vime.utils.common import is_npu
from vime.utils.memory_utils import print_memory

logger = logging.getLogger(__name__)


class TrainProfiler:
    def __init__(self, args):
        self.args = args
        self._torch_profiler_overall = None
        self._memory_profiler_overall = None

        if args.use_pytorch_profiler and ("train_overall" in args.profile_target):
            self._torch_profiler_overall = _create_torch_profiler(args, name="train_overall")

        if args.record_memory_history and ("train_overall" in args.profile_target):
            self._memory_profiler_overall = _BaseMemoryProfiler.create(args)
            self._memory_profiler_overall.start()

    def on_init_end(self):
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.start()

    def step(self, rollout_id: int):
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.step()

        if (
            self._memory_profiler_overall is not None
            and ((s := self.args.memory_snapshot_num_steps) is not None)
            and (rollout_id == s - 1)
        ):
            self._memory_profiler_overall.stop()

    def capture_oom(self, **context):
        if self._memory_profiler_overall is not None:
            self._memory_profiler_overall.capture_oom(context=context)
            return
        print_memory(_format_oom_message("when oom", context))

    def iterate_train_actor(self, iterator):
        return _profile_simple_loop(iterator, self.args, name="train_actor")

    def iterate_train_log_probs(self, iterator):
        return _profile_simple_loop(iterator, self.args, name="train_log_probs")


def _profile_simple_loop(iterator, args, name):
    if not (args.use_pytorch_profiler and (name in args.profile_target)):
        yield from iterator
        return

    torch_profiler = _create_torch_profiler(args, name=name)
    torch_profiler.start()
    for item in iterator:
        yield item
        torch_profiler.step()


def _create_torch_profiler(args, name):
    return torch.profiler.profile(
        schedule=torch.profiler.schedule(
            # TODO the train_actor and train_log_probs ones may need to have different args to control step
            wait=max(args.profile_step_start - 1, 0),
            warmup=1 if args.profile_step_start > 0 else 0,
            active=args.profile_step_end - args.profile_step_start,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            args.tensorboard_dir,
            worker_name=f"{name}_rank_{torch.distributed.get_rank()}",
            use_gzip=True,
        ),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True,
    )


def _safe_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return "na"


def _sanitize_path_part(value):
    return str(value).replace(os.sep, "-").replace(" ", "_").replace(":", "-")


def _format_oom_message(prefix, context):
    if not context:
        return prefix
    items = [f"{k}={v}" for k, v in context.items() if v is not None]
    return f"{prefix} ({', '.join(items)})" if items else prefix


class _BaseMemoryProfiler:
    @staticmethod
    def create(args):
        c = {
            "torch": _TorchMemoryProfiler,
            "memray": _MemrayMemoryProfiler,
        }[args.memory_recorder]
        return c(args)

    def __init__(self, args):
        self.args = args
        self._path_dir = Path(args.memory_snapshot_dir)
        self._path_dir.mkdir(parents=True, exist_ok=True)
        self._path_name = getattr(args, "memory_snapshot_path", None) or "memory_snapshot.pickle"
        self._path_dump = self._build_path("final")
        self._oom_dump_path = None

    def _build_path(self, tag):
        return self._path_dir / (
            f"memory_snapshot_{tag}_time{time.time()}_pid{os.getpid()}_rank{_safe_rank()}_{_sanitize_path_part(self._path_name)}"
        )

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def _dump_snapshot(self, path):
        raise NotImplementedError

    def capture_oom(self, context=None, observer_payload=None):
        if self._oom_dump_path is not None:
            logger.info(f"OOM diagnostics already dumped to {self._oom_dump_path}")
            return self._oom_dump_path

        self._oom_dump_path = self._build_path("oom")
        logger.error(
            _format_oom_message(
                f"Observe OOM, will dump snapshot to {self._oom_dump_path}.",
                context,
            )
        )
        if observer_payload is not None:
            logger.error(f"OOM observer payload: {observer_payload!r}")
        traceback.print_stack()
        print_memory(_format_oom_message("when oom", context))
        try:
            self._dump_snapshot(self._oom_dump_path)
        except Exception:
            logger.exception(f"Failed to dump memory snapshot to {self._oom_dump_path}")
        return self._oom_dump_path


class _TorchMemoryProfiler(_BaseMemoryProfiler):
    def start(self):
        logger.info("Attach OOM dump memory history.")

        if is_npu():
            torch.npu.memory._record_memory_history(
                max_entries=1000000,
                stacks="all",
            )

            try:
                import torch_npu

                def oom_observer(*observer_payload):
                    self.capture_oom(
                        context={"backend": "npu", "stage": "oom_observer"},
                        observer_payload=observer_payload,
                    )

                torch_npu._C._npu_attach_out_of_memory_observer(oom_observer)
            except Exception:
                logger.exception("Failed to attach NPU OOM observer")
            return

        torch.cuda.memory._record_memory_history(
            max_entries=1000000,
            stacks="all",
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            self.capture_oom(
                context={"backend": "cuda", "device": device},
                observer_payload=(alloc, device_alloc, device_free),
            )

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    def _dump_snapshot(self, path):
        if is_npu():
            torch.npu.memory._dump_snapshot(str(path))
            return
        torch.cuda.memory._dump_snapshot(str(path))

    def stop(self):
        logger.info(f"Dump memory snapshot to: {self._path_dump}")
        self._dump_snapshot(self._path_dump)
        if is_npu():
            torch.npu.memory._record_memory_history(enabled=None)
        else:
            torch.cuda.memory._record_memory_history(enabled=None)


class _MemrayMemoryProfiler(_BaseMemoryProfiler):
    def __init__(self, args):
        super().__init__(args)
        assert args.memory_snapshot_num_steps is not None, "In memray, must provide --memory-snapshot-num-steps"

    def start(self):
        logger.info("Memray tracker started.")
        import memray

        self._tracker = memray.Tracker(
            file_name=self._path_dump,
            native_traces=True,
        )
        self._tracker.__enter__()

    def stop(self):
        logger.info(f"Memray tracker stopped and dump snapshot to: {self._path_dump}")
        self._tracker.__exit__(None, None, None)
