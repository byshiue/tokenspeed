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

"""Immutable FP8 weight shards pulled into reusable full-GEMM storage."""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.gemm import (
    fp8_linear,
    fp8_linear_accepts_prepacked_input,
    fp8_linear_prepacked,
    prepare_fp8_linear,
)
from tokenspeed_kernel.ops.gemm.flashinfer import has_flashinfer_fp8_blockscale
from tokenspeed_kernel.ops.gemm.fp8_utils import (
    flashinfer_fp8_blockscale_quantize_prepacked,
)
from tokenspeed_kernel.registry import register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.cuda.async_copy import CudaAsyncCopy


@triton.jit
def _gather_prefetch_scales(
    pointers,
    canonical,
    prepared,
    N_BLOCKS: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    P: tl.constexpr,
    WEIGHT_WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    peer = tl.program_id(0)
    shard_k = K_BLOCKS // P
    offsets = tl.arange(0, BLOCK)
    source = tl.load(pointers + peer).to(tl.pointer_type(tl.int32))
    mask = offsets < N_BLOCKS * shard_k
    bits = tl.load(source + WEIGHT_WORDS + offsets, mask, other=0)
    row = offsets // shard_k
    col = peer * shard_k + offsets % shard_k
    tl.store(canonical + row * K_BLOCKS + col, bits, mask)
    tl.store(prepared + col * N_BLOCKS + row, bits, mask)


@triton.jit
def _restore_prefetch_weight(
    packed,
    weight,
    N: tl.constexpr,
    K_WORDS: tl.constexpr,
    P: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Copies arrive as [peer,N,K/P]. GEMM needs ordinary contiguous [N,K].
    dst = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = dst // K_WORDS, dst % K_WORDS
    shard_k = K_WORDS // P
    peer = col // shard_k
    src = peer * N * shard_k + row * shard_k + col % shard_k
    bits = tl.load(packed + src, dst < N * K_WORDS, other=0)
    tl.store(weight + dst, bits, dst < N * K_WORDS)


class CudaWeightPrefetchSource:
    """Publish one immutable layer shard; retain mappings until collective teardown.

    Args:
        group: Ordered TP process group, created before loading/capture.
        weight: FP8 input-channel shard, [N,K/TP].
        scales: Canonical FP32 scales, [N/128,K/TP/128].
        input_size: Full input width K, including every peer's channel shard.

    The returned weight/scales views replace the Linear parameters at setup,
    so no duplicate persistent GPU weight remains. Do not update these views
    while any rank can execute a graph or read peer mappings.
    """

    def __init__(self, group, weight, scales, input_size):
        self.group = group
        self.size, self.rank = group.size(), group.rank()
        self.n, self.k = weight.shape[0], input_size
        if (
            weight.device.type != "cuda"
            or weight.dtype != torch.float8_e4m3fn
            or scales.dtype != torch.float32
            or self.size < 2
            or self.n % 128
            or self.k % (128 * self.size)
            or weight.shape != (self.n, self.k // self.size)
            or scales.shape != (self.n // 128, self.k // self.size // 128)
        ):
            raise ValueError(
                "Weight prefetch requires CUDA FP8 weights and FP32 128x128 scales"
            )
        self.weight_bytes = weight.numel()
        self.payload = symm_mem.empty(
            (self.weight_bytes + scales.numel() * 4,),
            dtype=torch.uint8,
            device=weight.device,
        )
        self.weight = (
            self.payload[: self.weight_bytes].view(weight.dtype).view_as(weight)
        )
        self.scales = (
            self.payload[self.weight_bytes :].view(torch.float32).view_as(scales)
        )
        self.weight.copy_(weight)
        self.scales.copy_(scales)
        self.handle = symm_mem.rendezvous(self.payload, group)
        self.peers = [
            self.handle.get_buffer(
                peer, self.payload.shape, self.payload.dtype, storage_offset=0
            )
            for peer in range(self.size)
        ]
        self.pointers = torch.tensor(
            [peer.data_ptr() for peer in self.peers],
            dtype=torch.int64,
            device=weight.device,
        )
        self.copy = CudaAsyncCopy()
        self.closed = False
        # Publish once, before any peer may skip a forward or start capture.
        torch.cuda.synchronize(weight.device)
        dist.barrier(group=group)

    def close(self):
        """Collectively retire immutable mappings after every graph is released."""
        if self.closed:
            return
        torch.cuda.synchronize(self.weight.device)
        dist.barrier(group=self.group)
        self.peers.clear()
        self.handle = None
        self.closed = True


@register_kernel(
    "communication",
    "weight_prefetch",
    name="cuda_prefetch_fp8_weights",
    solution="cuda",
    signatures=format_signatures(("weight",), "dense", {torch.float8_e4m3fn}),
)
def cuda_prefetch_fp8_weights(
    source, packed, weight, canonical_scales, prepared_scales
):
    """Refresh full weight/scales on the current stream from immutable peer shards.

    Args:
        source: Published CudaWeightPrefetchSource for this layer.
        packed: Borrowed contiguous rank-major FP8 transfer scratch, N*K bytes.
        weight: Borrowed full [N,K] FP8 destination.
        canonical_scales: Borrowed [N/128,K/128] FP32 destination.
        prepared_scales: Borrowed contiguous [K/128,N/128] FP32 destination.

    Returns:
        None. The caller must order this stream before GEMM and after the
        preceding GEMM that consumed these destinations.
    """
    if source.closed:
        raise RuntimeError("Weight prefetch source has been closed")
    _gather_prefetch_scales[(source.size,)](
        source.pointers,
        canonical_scales.view(torch.int32),
        prepared_scales.view(torch.int32),
        source.n // 128,
        source.k // 128,
        source.size,
        source.weight_bytes // 4,
        triton.next_power_of_2(source.scales.numel()),
        num_warps=4,
    )
    stream = torch.cuda.current_stream(weight.device).cuda_stream
    # Remote first, local last: this order gave the best complete KDA latency.
    for step in range(1, source.size + 1):
        peer = (source.rank - step) % source.size
        source.copy.device_to_device(
            packed.data_ptr() + peer * source.weight_bytes,
            source.peers[peer].data_ptr(),
            source.weight_bytes,
            stream,
        )
    _restore_prefetch_weight[(triton.cdiv(source.n * source.k // 4, 1024),)](
        packed.view(torch.int32),
        weight.view(torch.int32),
        source.n,
        source.k // 4,
        source.size,
        1024,
    )


class CudaWeightPrefetchWorkspace:
    """Transfer and full-weight scratch reused by sequential projection layers.

    Args:
        shapes: All logical (N,K) shapes, fixed before capture.
        device: CUDA device hosting the local full-width GEMM.

    The plan follows the ordinary FlashInfer FP8 path, including its canonical
    scale fallback for large unaligned token counts. No token-dependent
    communication allocations or weight caches are created during forward.
    """

    def __init__(self, shapes, device):
        if not has_flashinfer_fp8_blockscale():
            raise ValueError(
                "Weight prefetch currently requires Blackwell FlashInfer block-FP8 GEMM"
            )
        self.shapes = tuple(sorted(set(shapes)))
        capacity = max(n * k for n, k in self.shapes)
        self.weight_storage = torch.empty(
            capacity, dtype=torch.float8_e4m3fn, device=device
        )
        self.packed_storage = torch.empty_like(self.weight_storage)
        self.scale_storage = torch.ones(
            capacity // (128 * 128), dtype=torch.float32, device=device
        )
        self.stream = torch.cuda.Stream(device=device, priority=-1)
        self.views = {}
        for n, k in self.shapes:
            weight = self.weight_storage[: n * k].view(n, k)
            scales = self.scale_storage[: n * k // (128 * 128)].view(n // 128, k // 128)
            plan = prepare_fp8_linear(weight, scales, (128, 128), scale_format=None)
            if not fp8_linear_accepts_prepacked_input(plan, 128):
                raise ValueError(
                    "Weight prefetch requires the prepared FlashInfer FP8 plan"
                )
            self.views[(n, k)] = weight, scales, plan

    def gather(self, source):
        """Enqueue a complete refresh into this shape's borrowed destinations."""
        weight, scales, plan = self.views[(source.n, source.k)]
        cuda_prefetch_fp8_weights(
            source, self.packed_storage, weight, scales, plan.prepared_weight_scales
        )

    def prepare_input(self, x, shape):
        """Quantize independently of weight arrival; return None for canonical input."""
        _, _, plan = self.views[shape]
        if fp8_linear_accepts_prepacked_input(plan, x.shape[0]):
            return flashinfer_fp8_blockscale_quantize_prepacked(x, 128)
        return None

    def project(self, x, prepared_input, shape):
        """Return owned full-width outputs after the caller joins the prefetch stream."""
        weight, scales, plan = self.views[shape]
        if prepared_input is not None:
            values, input_scales = prepared_input
            return fp8_linear_prepacked(
                plan, values, weight, input_scales, x.shape[0], x.dtype, None
            )
        return fp8_linear(
            plan,
            x,
            weight,
            scales,
            input_scales=None,
            bias=None,
            out_dtype=x.dtype,
            out=None,
        )
