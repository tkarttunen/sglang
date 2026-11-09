"""INT4 QuickReduce, bundled and through AITER's dispatcher, on two GPUs.

Two ranks build the communicator with create_quick_allreduce, as SGLang
startup does. Without AITER_FLY_AR the bundled HIP codec serves INT4. With
AITER_FLY_AR=1 AITER QuickAllReduce serves it and picks FlyDSL or its HIP
kernel per payload. Each mode reduces bf16 and fp16 payloads, compares them
with an exact reduction, and checks that close() returns the IPC memory.
Driver free memory is read with torch.cuda.mem_get_info, which counts HIP
allocations outside the PyTorch caching allocator. No model is launched.
"""

import importlib.util
import multiprocessing
import os
import socket
import time
import traceback
import unittest

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.quick_all_reduce import (
    create_quick_allreduce,
    qr_rocm_arch_available,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=300, suite="stage-c-test-4-gpu-amd")
register_amd_ci(est_time=300, suite="stage-c-test-large-8-gpu-amd-mi35x")

_MIB = 1024 * 1024
# 16 MiB is above every INT4 minimum at two ranks and inside the FlyDSL
# mesh window on gfx950.
_PAYLOAD_ELEMENTS = 8 * _MIB


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _aiter_quickreduce_importable() -> bool:
    try:
        return (
            importlib.util.find_spec("aiter.dist.device_communicators.quick_all_reduce")
            is not None
        )
    except ImportError:
        return False


def _vram_used(device: int) -> int:
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return int(total_bytes - free_bytes)


def _relative_error(dtype: torch.dtype, adapter, rank: int, device) -> float:
    generator = torch.Generator(device="cpu").manual_seed(1234 + rank)
    local = torch.randn(_PAYLOAD_ELEMENTS, generator=generator).to(dtype)
    exact = local.float()
    dist.all_reduce(exact)
    inp = local.to(device)
    assert adapter.should_quick_allreduce(inp), dtype
    out = adapter.quick_all_reduce(inp).float().cpu()
    return float((out - exact).norm() / exact.norm())


def _one_mode(rank: int, device: torch.device, fly_ar: bool) -> dict:
    if fly_ar:
        os.environ["AITER_FLY_AR"] = "1"
    else:
        os.environ.pop("AITER_FLY_AR", None)
    before = _vram_used(rank)
    adapter = create_quick_allreduce(group=dist.group.WORLD, device=device)
    after = _vram_used(rank)
    probe = torch.empty(_PAYLOAD_ELEMENTS, dtype=torch.bfloat16, device=device)
    report = {
        "backend": type(adapter._communicator).__module__,
        "disabled": bool(adapter.disabled),
        "uses_flydsl_int4": adapter.uses_flydsl_int4(probe),
        "delta": after - before,
        "bf16_error": _relative_error(torch.bfloat16, adapter, rank, device),
        "fp16_error": _relative_error(torch.float16, adapter, rank, device),
        "before": before,
    }
    del probe
    # Both ranks must drop their peer mappings before the local free shows up.
    dist.barrier()
    adapter.close()
    dist.barrier()
    report["released_to"] = _vram_used(rank)
    return report


def _run_modes(rank: int, world_size: int, port: int, queue) -> None:
    os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = "INT4"
    os.environ["AITER_QUICK_REDUCE_CAST_BF16_TO_FP16"] = "1"
    os.environ.pop("ROCM_QUICK_REDUCE_QUANTIZATION", None)
    os.environ.pop("ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", None)
    os.environ["SGLANG_USE_AITER"] = "1"

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.empty(1024, device=device)
        reports = {"rank": rank, "bundled": _one_mode(rank, device, fly_ar=False)}
        if _aiter_quickreduce_importable():
            reports["aiter"] = _one_mode(rank, device, fly_ar=True)
        queue.put(reports)
    except Exception:
        queue.put({"rank": rank, "error": traceback.format_exc()})
    finally:
        dist.destroy_process_group()


class TestInt4QuickReduceBackends(CustomTestCase):
    @unittest.skipUnless(
        qr_rocm_arch_available() and torch.cuda.device_count() >= 2,
        "needs two ROCm GPUs that support QuickReduce",
    )
    def test_int4_reduces_correctly_on_each_backend(self):
        world_size = 2
        port = _get_open_port()
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        processes = [
            context.Process(target=_run_modes, args=(rank, world_size, port, queue))
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()

        deadline = time.monotonic() + 600
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        if any(process.is_alive() for process in processes):
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
            self.fail("INT4 QuickReduce test timed out")

        reports = []
        while not queue.empty():
            reports.append(queue.get())
        for rank, process in enumerate(processes):
            self.assertEqual(
                process.exitcode,
                0,
                f"rank {rank} exited {process.exitcode}; reports={reports}",
            )

        self.assertEqual(len(reports), world_size, reports)
        for item in reports:
            self.assertNotIn("error", item, item.get("error"))
            print(item, flush=True)
            self._assert_mode(item["bundled"], from_aiter=False)
            if "aiter" in item:
                self._assert_mode(item["aiter"], from_aiter=True)

    def _assert_mode(self, report: dict, from_aiter: bool) -> None:
        self.assertFalse(report["disabled"], report)
        self.assertEqual(report["backend"].startswith("aiter."), from_aiter, report)
        self.assertEqual(report["uses_flydsl_int4"], from_aiter, report)
        self.assertGreater(report["delta"], 0, report)
        # INT4 with a group scale keeps the relative error near 10%.
        self.assertLess(report["bf16_error"], 0.2, report)
        self.assertLess(report["fp16_error"], 0.2, report)
        self.assertLess(
            abs(report["released_to"] - report["before"]), 512 * _MIB, report
        )


if __name__ == "__main__":
    unittest.main()
