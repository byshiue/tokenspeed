# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Compatibility and isolation checks for the private routing initializer."""

import functools
import inspect

import pytest
from tokenspeed_kernel.thirdparty.flashinfer._routing_padding import (
    _patch_producer,
    patch_routing_sources,
)
from tokenspeed_kernel.thirdparty.flashinfer.trtllm_moe import (
    _clone,
    _entrypoints,
    _prepare_routing_sources,
    _register_private,
    _relocate_header,
)

_PRODUCER = """
  params.mPtrCtaIdxXyToMnLimit[ctaOffset[e] + cta] = min(mnLimit1, mnLimit2);
  if (threadIdx.x == 0) {
    int32_t permutedIdxSize;
    if (params.mIsPow2) {
      permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);
    } else {
      permutedIdxSize = mulTileN<int32_t>(numNonExitingCtas, params.mTileTokensDim);
    }
    params.mPtrPermutedIdxSize[0] = permutedIdxSize;
    params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
  }
  params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
  cudaTriggerProgrammaticLaunchCompletion();
"""

_CLONE_VALUE = object()


def test_private_headers_resolve_parent_includes_without_copying_the_package():
    source = '#include "../../exception.h"\n#include "RoutingKernel.h"\n'
    assert _relocate_header(source) == (
        '#include "flashinfer/exception.h"\n#include "RoutingKernel.h"\n'
    )


@pytest.mark.parametrize("changed_body", [False, True])
@pytest.mark.parametrize("relocated_header", [False, True])
def test_prepared_sources_retain_upstream_notices(
    monkeypatch, changed_body, relocated_header
):
    license_header = (
        "/* Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.\n"
        ' * Licensed under the Apache License, Version 2.0 (the "License"). */\n'
    )
    original = license_header + '#include "../common.h"\nint original;\n'
    transformed = (
        original.replace("int original;", "int patched;") if changed_body else original
    )
    monkeypatch.setattr(
        "tokenspeed_kernel.thirdparty.flashinfer.trtllm_moe.patch_routing_sources",
        lambda sources: {**sources, "example.h": transformed},
    )
    sources = {"example.h": original, "unchanged.cu": license_header}
    actual = _prepare_routing_sources(
        sources, {"example.h"} if relocated_header else set()
    )
    expected = _relocate_header(transformed) if relocated_header else transformed
    if changed_body or relocated_header:
        assert actual["example.h"].startswith("// Modified by TokenSpeed")
        assert actual["example.h"].count("// Modified by TokenSpeed") == 1
        assert actual["example.h"].endswith(expected)
    else:
        assert actual["example.h"] == original
    assert license_header in actual["example.h"]
    assert actual["unchanged.cu"] == license_header
    assert sources["example.h"] == original


def test_padding_is_written_by_producer_before_pdl():
    actual = _patch_producer(_PRODUCER, 1)
    assert (
        "initializeRouteTilePadding(params, min(mnLimit1, mnLimit2), mnLimit1)"
        in actual
    )
    assert (
        "  initializeRouteMapSlack(params, numNonExitingCtas);\n  if (threadIdx.x == 0)"
        in actual
    )
    assert actual.index("initializeRouteMapSlack") < actual.index(
        "cudaTriggerProgrammatic"
    )
    assert "params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;" in actual
    assert "cudaMemset" not in actual


@pytest.mark.parametrize(
    "source",
    ["", _PRODUCER * 2, _PRODUCER.replace("min(mnLimit1, mnLimit2)", "mnLimit1")],
)
def test_unrecognized_producer_fails_closed(source):
    with pytest.raises(RuntimeError, match="Unsupported FlashInfer routing"):
        _patch_producer(source, 1)


def test_changed_publication_fails_closed():
    with pytest.raises(RuntimeError, match="count publishers"):
        _patch_producer(_PRODUCER.replace("mPtrPermutedIdxSize", "anotherSize"), 1)


def test_missing_native_sources_fail_closed():
    with pytest.raises(RuntimeError, match="missing"):
        patch_routing_sources({})


@pytest.mark.parametrize("guard", [" + 1", ""])
def test_native_routing_transforms_cover_every_producer(guard):
    env = pytest.importorskip("flashinfer.jit.env")
    paths = list((env.FLASHINFER_INCLUDE_DIR / "flashinfer/trtllm/fused_moe").iterdir())
    paths += list(env.FLASHINFER_CSRC_DIR.glob("trtllm_fused_moe_*.cu"))
    paths += list(
        (env.FLASHINFER_CSRC_DIR / "fused_moe/trtllm_backend").glob("*routing*.cu")
    )
    sources = {path.name: path.read_text() for path in paths if path.is_file()}
    launcher = "trtllm_fused_moe_kernel_launcher.cu"
    sources[launcher] = sources[launcher].replace(
        "max_num_padded_tokens + 1", "max_num_padded_tokens" + guard
    )
    before = dict(sources)
    actual = patch_routing_sources(sources)
    assert sources == before
    prepared = _prepare_routing_sources(
        sources, {path.name for path in paths if path.suffix in {".h", ".cuh"}}
    )
    for name, source in prepared.items():
        if source != sources[name]:
            assert source.startswith("// Modified by TokenSpeed")
            upstream_header = sources[name].split("*/", maxsplit=1)[0] + "*/"
            assert upstream_header in source
    assert "permuted_idx_to_token_idx.numel()" in actual[launcher]
    assert actual[launcher].count("cudaMemsetAsync") == before[launcher].count(
        "cudaMemsetAsync"
    )
    for name, count in (
        ("RoutingKernel.cuh", 3),
        ("trtllm_fused_moe_routing_custom.cu", 2),
        ("trtllm_fused_moe_routing_llama4.cu", 1),
    ):
        assert actual[name].count("initializeRouteTilePadding(params,") == count
        assert actual[name].count("initializeRouteMapSlack(params,") == count
    # No API change to Runner::run, and unrelated callers remain opted out.
    assert "int32_t mRouteMapCapacity{0}" in actual["runner.h"]
    assert "row < params.mRouteMapCapacity" in actual["RoutingKernel.cuh"]
    with pytest.raises(RuntimeError, match="route-map allocation"):
        patch_routing_sources({**sources, launcher: ""})
    with pytest.raises(RuntimeError, match="map writers"):
        patch_routing_sources({**sources, "new_routing.cu": _PRODUCER})


def test_function_rebinding_does_not_mutate_upstream():
    sentinel = object()

    def original(x, *, value):
        return x, value, _CLONE_VALUE

    clone = _clone(original, {**original.__globals__, "_CLONE_VALUE": sentinel})
    assert clone(3, value=4) == (3, 4, sentinel)
    assert inspect.signature(clone) == inspect.signature(original)
    assert original(3, value=4) == (3, 4, _CLONE_VALUE)
    assert original.__globals__["_CLONE_VALUE"] is not sentinel


def test_operator_names_are_private():
    def register(name, *, mutates_args):
        return name, mutates_args

    assert _register_private(register, "flashinfer::moe", mutates_args=("out",)) == (
        "tokenspeed_flashinfer_route_init::moe",
        ("out",),
    )
    with pytest.raises(RuntimeError, match="Unexpected FlashInfer operator"):
        _register_private(register, "another::moe", mutates_args=())


def test_upstream_dispatch_and_caches_are_unchanged():
    core = pytest.importorskip("flashinfer.fused_moe.core")
    before = dict(vars(core))
    private = _entrypoints()
    assert vars(core) == before
    assert (
        private["get_trtllm_moe_sm100_module"] is not core.get_trtllm_moe_sm100_module
    )
    for name in ("trtllm_fp4_block_scale_moe", "trtllm_fp4_block_scale_routed_moe"):
        assert private[name].__globals__ is private
        assert inspect.signature(private[name]) == inspect.signature(
            getattr(core, name)
        )
    factory = private.get("_get_trtllm_moe_sm100_module_impl")
    if factory is not None:
        assert isinstance(factory, functools._lru_cache_wrapper)
        assert factory is not core._get_trtllm_moe_sm100_module_impl
