"""INT4 QuickReduce reaches FlyDSL only through AITER's QuickAllReduce.

Without AITER_FLY_AR=1, INT4 runs on the bundled HIP codec. With it, INT4 is
served by AITER QuickAllReduce, which picks FlyDSL or its HIP kernel per
payload. The adapter maps SGLang's fused all-reduce + RMSNorm hooks onto that
dispatcher's public methods. No model is launched and no GPU is needed.
"""

import sys
from types import ModuleType

import pytest
import torch

from sglang.srt.distributed.device_communicators import quick_all_reduce as qr_mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

QR_MAX_SIZE = 2 * 1024**3


@pytest.fixture(autouse=True)
def _clear_quick_reduce_env(monkeypatch):
    # The ROCm image sets ROCM_QUICK_REDUCE_QUANTIZATION.
    for names in qr_mod._QUICK_REDUCE_ENV_ALIASES.items():
        for name in names:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("AITER_FLY_AR", raising=False)


class _Bundled:
    def __init__(self, **kwargs) -> None:
        self.disabled = False


class _AiterDispatcher:
    """AITER main QuickAllReduce: no FlyDSL fused all-reduce + RMSNorm."""

    def __init__(self, **kwargs) -> None:
        self.disabled = False
        self.qr_max_size = QR_MAX_SIZE

    def should_quick_allreduce(self, inp: torch.Tensor) -> bool:
        return inp.dtype == torch.bfloat16

    def quick_all_reduce(self, inp, *, out=None):
        return inp if out is None else out.copy_(inp)


class _AiterFusedDispatcher(_AiterDispatcher):
    """AITER QuickAllReduce with aiter#5670's FlyDSL fused path."""

    fuses = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fused_calls = []

    def should_fly_allreduce_rmsnorm(self, inp, residual_inp, weight) -> bool:
        return type(self).fuses

    def fly_allreduce_rmsnorm(self, inp, residual_inp, weight, eps):
        self.fused_calls.append((inp, residual_inp, weight, eps))
        return inp + 1, residual_inp + 1


def _install(monkeypatch, aiter_cls):
    fake = ModuleType("aiter.dist.device_communicators.quick_all_reduce")
    fake.QuickAllReduce = aiter_cls
    monkeypatch.setitem(
        sys.modules, "aiter.dist.device_communicators.quick_all_reduce", fake
    )
    monkeypatch.setattr(qr_mod, "QuickAllReduce", _Bundled)
    monkeypatch.setattr(qr_mod, "qr_rocm_arch_available", lambda: True)
    monkeypatch.setenv("SGLANG_USE_AITER", "1")


def _create():
    return qr_mod.create_quick_allreduce(group=object(), device="cpu")


def _operands(dtype=torch.bfloat16, hidden=8192):
    inp = torch.zeros(4, hidden, dtype=dtype)
    return inp, torch.zeros_like(inp), torch.ones(hidden, dtype=dtype)


@pytest.mark.parametrize(
    "codec_env", ["AITER_QUICK_REDUCE_QUANTIZATION", "ROCM_QUICK_REDUCE_QUANTIZATION"]
)
def test_int4_with_fly_ar_uses_the_aiter_dispatcher(monkeypatch, codec_env):
    _install(monkeypatch, _AiterDispatcher)
    monkeypatch.setenv(codec_env, "INT4")
    monkeypatch.setenv("AITER_FLY_AR", "1")

    adapter = _create()

    assert isinstance(adapter._communicator, _AiterDispatcher)
    assert adapter._flydsl
    assert qr_mod.os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] == "INT4"


@pytest.mark.parametrize("fly_ar", [None, "0"])
def test_int4_without_fly_ar_keeps_the_bundled_codec(monkeypatch, fly_ar):
    _install(monkeypatch, _AiterDispatcher)
    monkeypatch.setenv("AITER_QUICK_REDUCE_QUANTIZATION", "INT4")
    if fly_ar is not None:
        monkeypatch.setenv("AITER_FLY_AR", fly_ar)

    adapter = _create()

    assert isinstance(adapter._communicator, _Bundled)
    assert not adapter._flydsl
    assert qr_mod.os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] == "INT4"
    assert not adapter.uses_flydsl_int4(_operands()[0])


def test_int4_with_fly_ar_falls_back_to_bundled_without_aiter(monkeypatch):
    _install(monkeypatch, _AiterDispatcher)
    monkeypatch.setitem(
        sys.modules, "aiter.dist.device_communicators.quick_all_reduce", None
    )
    monkeypatch.setenv("AITER_QUICK_REDUCE_QUANTIZATION", "INT4")
    monkeypatch.setenv("AITER_FLY_AR", "1")

    adapter = _create()

    assert isinstance(adapter._communicator, _Bundled)
    assert not adapter._flydsl
    assert qr_mod.os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] == "INT4"


def test_int8_ignores_fly_ar(monkeypatch):
    _install(monkeypatch, _AiterDispatcher)
    monkeypatch.setenv("ROCM_QUICK_REDUCE_QUANTIZATION", "INT8")
    monkeypatch.setenv("AITER_FLY_AR", "1")

    adapter = _create()

    assert isinstance(adapter._communicator, _Bundled)
    assert not adapter._flydsl


def test_dispatcher_owns_the_payloads_it_accepts(monkeypatch):
    _install(monkeypatch, _AiterDispatcher)
    monkeypatch.setenv("AITER_QUICK_REDUCE_QUANTIZATION", "INT4")
    monkeypatch.setenv("AITER_FLY_AR", "1")

    adapter = _create()

    assert adapter.uses_flydsl_int4(_operands(torch.bfloat16)[0])
    assert not adapter.uses_flydsl_int4(_operands(torch.float16)[0])


def test_aiter_without_the_fused_path_declines_fusion():
    adapter = qr_mod._QuickAllReduceAdapter(_AiterDispatcher(), flydsl=True)
    inp, residual, weight = _operands()

    assert not adapter.should_fused_allreduce_rmsnorm(inp, residual, weight)
    assert adapter.fused_rmsnorm_limit_bytes(inp) is None


def test_fused_path_maps_to_the_aiter_dispatcher():
    backend = _AiterFusedDispatcher()
    adapter = qr_mod._QuickAllReduceAdapter(backend, flydsl=True)
    inp, residual, weight = _operands()

    assert adapter.should_fused_allreduce_rmsnorm(inp, residual, weight)
    out, residual_out = adapter.fused_allreduce_rmsnorm(inp, residual, weight, 1e-6)

    assert backend.fused_calls == [(inp, residual, weight, 1e-6)]
    torch.testing.assert_close(out, inp + 1)
    torch.testing.assert_close(residual_out, residual + 1)
    assert adapter.fused_rmsnorm_limit_bytes(inp) == QR_MAX_SIZE


def test_fused_path_declines_where_the_dispatcher_declines(monkeypatch):
    monkeypatch.setattr(_AiterFusedDispatcher, "fuses", False)
    adapter = qr_mod._QuickAllReduceAdapter(_AiterFusedDispatcher(), flydsl=True)

    assert not adapter.should_fused_allreduce_rmsnorm(*_operands())


def test_fused_path_needs_fly_ar():
    adapter = qr_mod._QuickAllReduceAdapter(_AiterFusedDispatcher())
    inp, residual, weight = _operands()

    assert not adapter.should_fused_allreduce_rmsnorm(inp, residual, weight)
    assert not adapter.uses_flydsl_int4(inp)
    assert adapter.fused_rmsnorm_limit_bytes(inp) is None


def test_fused_limit_needs_a_payload_the_dispatcher_accepts():
    adapter = qr_mod._QuickAllReduceAdapter(_AiterFusedDispatcher(), flydsl=True)

    assert adapter.fused_rmsnorm_limit_bytes(_operands(torch.float16)[0]) is None


def test_bundled_int4_has_no_flydsl_state():
    assert not hasattr(qr_mod.QuickAllReduce, "_flydsl_int4")
    assert not hasattr(qr_mod, "_flydsl_int4_available")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
