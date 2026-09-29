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
from tokenspeed_kernel._triton import triton
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


class CudaWeightPrefetchSource:
    """Publish one immutable layer shard; retain mappings until collective teardown.

    Args:
        group: Ordered TP process group, created before loading/capture.
        weight: FP8 output-channel shard, [ceil(N/128/TP)*128,K].
        scales: Canonical FP32 scales, [shard_rows/128,K/128].
        output_size: Full logical output width N, excluding shard padding.

    The returned weight/scales views replace the Linear parameters at setup,
    so no duplicate persistent GPU weight remains. Do not update these views
    while any rank can execute a graph or read peer mappings.
    """

    def __init__(self, group, weight, scales, output_size):
        self.group = group
        self.size, self.rank = group.size(), group.rank()
        self.n, self.k = output_size, weight.shape[1]
        self.shard_rows = triton.cdiv(self.n // 128, self.size) * 128
        if (
            weight.device.type != "cuda"
            or weight.dtype != torch.float8_e4m3fn
            or scales.dtype != torch.float32
            or self.size < 2
            or self.n % 128
            or self.k % 128
            or weight.shape != (self.shard_rows, self.k)
            or scales.shape != (self.shard_rows // 128, self.k // 128)
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
def cuda_prefetch_fp8_weights(source, weight):
    """Refresh full weights on the current stream using only copy engines.

    Args:
        source: Published CudaWeightPrefetchSource for this layer.
        weight: Borrowed full [N,K] FP8 destination.

    Returns:
        None. The caller must order this stream before GEMM and after the
        preceding GEMM that consumed this destination. Scales are cached at setup.
    """
    if source.closed:
        raise RuntimeError("Weight prefetch source has been closed")
    stream = torch.cuda.current_stream(weight.device).cuda_stream
    # Remote first, local last: this order gave the best complete KDA latency.
    for step in range(1, source.size + 1):
        peer = (source.rank - step) % source.size
        start = peer * source.shard_rows
        valid = max(0, min(source.shard_rows, source.n - start))
        if not valid:
            continue
        # N shards land directly in GEMM's contiguous [N,K] layout. Ignore
        # padded trailing blocks, including peers with no valid output rows.
        source.copy.device_to_device(
            weight.data_ptr() + start * source.k,
            source.peers[peer].data_ptr(),
            valid * source.k,
            stream,
        )


class CudaWeightPrefetchWorkspace:
    """Shared full-weight scratch and immutable per-layer scale plans.

    Args:
        shapes: All logical (N,K) shapes, fixed before capture.
        device: CUDA device hosting the local full-width GEMM.

    The plan follows the ordinary FlashInfer FP8 path, including its canonical
    scale fallback for large unaligned token counts. Both scale layouts are
    cached before capture; only the large weight buffer is reused across layers.
    No token-dependent communication allocations are created during forward.
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
        self.stream = torch.cuda.Stream(device=device, priority=-1)
        self.views = {}

    def prepare(self, source):
        """Cache a published layer's full FP32 scales and GEMM layout once.

        Args:
            source: Immutable layer shard, published on every peer before this call.

        Returns:
            None. Call before memory profiling or capture on the setup stream.
        """
        if source in self.views:
            return
        n, k = source.n, source.k
        if (n, k) not in self.shapes:
            raise ValueError("Weight shape is not reserved in this workspace")
        weight = self.weight_storage[: n * k].view(n, k)
        scales = torch.empty(
            (n // 128, k // 128), dtype=torch.float32, device=weight.device
        )
        stream = torch.cuda.current_stream(weight.device).cuda_stream
        for peer in range(source.size):
            start = peer * source.shard_rows // 128
            valid = max(0, min(source.shard_rows // 128, n // 128 - start))
            if valid:
                source.copy.device_to_device(
                    scales.data_ptr() + start * (k // 128) * 4,
                    source.peers[peer].data_ptr() + source.weight_bytes,
                    valid * (k // 128) * 4,
                    stream,
                )
        plan = prepare_fp8_linear(weight, scales, (128, 128), scale_format=None)
        if not fp8_linear_accepts_prepacked_input(plan, 128):
            raise ValueError(
                "Weight prefetch requires the prepared FlashInfer FP8 plan"
            )
        # Same-shaped layers share weights' scratch address, never their scales.
        self.views[source] = weight, scales, plan

    def gather(self, source):
        """Refresh shared weight scratch; this layer's scales stay unchanged."""
        weight, _, _ = self.views[source]
        cuda_prefetch_fp8_weights(source, weight)

    def prepare_input(self, x, source):
        """Quantize independently of weight arrival; return None for canonical input."""
        _, _, plan = self.views[source]
        if fp8_linear_accepts_prepacked_input(plan, x.shape[0]):
            return flashinfer_fp8_blockscale_quantize_prepacked(x, 128)
        return None

    def project(self, x, prepared_input, source):
        """Return owned full-width outputs after the caller joins the prefetch stream."""
        weight, scales, plan = self.views[source]
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
