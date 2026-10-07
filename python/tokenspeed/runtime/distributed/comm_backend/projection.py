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

"""Prepared, backend-neutral contracts for DP projection collectives."""

import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.mapping import Group
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.distributed.comm_backend.base import CommBackend


@dataclass(frozen=True)
class ProjectionSpec:
    """Static layout and allocation bounds, identical on every subgroup peer.

    Sequential layers may share a prepared object with the same spec. Separate
    streams or concurrently executing models must prepare separate objects.
    Sizes are full (unsharded) widths; max_tokens is physical rows per owner.
    """

    group: Group
    kind: Literal["column", "row"]
    input_size: int
    output_size: int
    max_tokens: int
    dtype: torch.dtype
    device: torch.device


class PreparedProjection(Protocol):
    """Collectives with persistent storage prepared before memory profiling.

    Intermediate results, including FP8 values/scales and GEMM destinations,
    are borrowed until the next call to that operation on this object. Final
    inverse-A2A/RS results are caller-owned. Finish consumers on the same stream
    before reusing scratch;
    destroy referencing CUDA graphs before close().
    """

    spec: ProjectionSpec

    def all_gather(
        self, inputs: torch.Tensor, rows: int, quantize: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Gather padded owner rows, optionally into FP8 with MN-major scales.

        quantize permits 1x128 FP8 fusion. An unsupported backend returns
        ordinary activations and None, letting the Linear use its usual GEMM.
        """

    def all_to_all(
        self,
        inputs: torch.Tensor,
        rows: int,
        inverse: bool,
        quantize: bool,
        out: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Exchange local full channels for gathered channel shards, or inverse.

        Forward: [local_rows,K] -> [TP*rows,K/TP], with zero owner padding;
        out must be None and the result is borrowed, with optional quantization.
        Inverse: [TP*rows,N/TP] -> [rows,N], writing into required caller-owned
        out without quantization.
        """

    def acquire_output(self, rows: int) -> torch.Tensor:
        """Borrow [TP*rows,N] GEMM output storage for a following RS."""

    def reduce_scatter(self, partial: torch.Tensor, rows: int) -> torch.Tensor:
        """Sum [TP*rows,N] partials into owned [rows,N] owner outputs."""

    def close(self) -> None:
        """Collectively release persistent resources after graphs are gone."""


class ProjectionCollectives:
    """Backend-owned layout conversion and optional node-local Lamport kernels.

    Construct through comm_ops, not in model/Linear code. Fallbacks use the
    supplied backend, including its deterministic reduction policy. Capacities
    and physical rows, never this owner's valid rows, select the collective on
    every peer. Initialization errors are fatal; forward never retries setup.
    """

    def __init__(
        self,
        spec: ProjectionSpec,
        fallback: "CommBackend",
        use_lamport: bool,
        use_lamport_reduction: bool,
    ):
        if (
            spec.kind not in ("column", "row")
            or len(spec.group) < 2
            or min(spec.input_size, spec.output_size, spec.max_tokens) <= 0
        ):
            raise ValueError("Invalid prepared projection dimensions")
        width = spec.input_size if spec.kind == "row" else spec.output_size
        if width % len(spec.group):
            raise ValueError("Projection channel width must be divisible by TP")
        self.spec, self._fallback = spec, fallback
        self._closed = False
        self._send = torch.empty(
            spec.max_tokens * spec.input_size, dtype=spec.dtype, device=spec.device
        )
        self._received = torch.empty(
            spec.max_tokens * width, dtype=spec.dtype, device=spec.device
        )
        self._gathered = (
            torch.empty(
                len(spec.group) * spec.max_tokens,
                spec.input_size,
                dtype=spec.dtype,
                device=spec.device,
            )
            if spec.kind == "column"
            else None
        )
        self._partial = (
            torch.empty(
                len(spec.group) * spec.max_tokens,
                spec.output_size,
                dtype=spec.dtype,
                device=spec.device,
            )
            if spec.kind == "row"
            else None
        )
        self._gather = None
        self._gather_quant = None
        self._a2a = None
        self._reduction = None
        if use_lamport:
            # CUDA IPC requires a node-local group. Agree on topology before
            # any rank attempts collective native allocation.
            hosts = [None] * len(spec.group)
            dist.all_gather_object(
                hosts,
                socket.gethostname(),
                group=pg_manager.get_process_group("gloo", spec.group),
            )
            if len(set(hosts)) == 1:
                self._prepare_lamport(use_lamport_reduction)

        # Initialize fallback communicators and all selected native kernels
        # before capture. No persistent allocation is deferred to forward.
        probe = self._send[: spec.input_size].view(1, spec.input_size)
        probe.zero_()
        if spec.kind == "column":
            self.all_gather(probe, 1, False)
            self.all_gather(probe, 1, True)
            local = probe.new_zeros(
                len(spec.group), spec.output_size // len(spec.group)
            )
            self.all_to_all(local, 1, True, False, probe.new_empty(1, spec.output_size))
        else:
            self.all_to_all(probe, 1, False, False, None)
            self.all_to_all(probe, 1, False, True, None)
            partial = self.acquire_output(1)
            partial.zero_()
            self.reduce_scatter(partial, 1)
        # Warm NCCL even when all probes above selected a small-message kernel.
        small = probe.new_zeros(len(spec.group), 1)
        self._fallback.all_to_all_single(
            torch.empty_like(small),
            small,
            spec.group,
            output_split_sizes=None,
            input_split_sizes=None,
        )
        self._fallback.all_gather_single(small, probe.new_zeros(1, 1), spec.group)
        self._fallback.reduce_scatter(small, spec.group)

    def _prepare_lamport(self, use_reduction: bool) -> None:
        from tokenspeed_kernel.ops.communication.cuda import TokenSpeedA2ALamportState
        from tokenspeed_kernel.ops.communication.trtllm import (
            TrtllmAllGatherQuantState,
            TrtllmAllGatherState,
            TrtllmReduceScatterState,
        )
        from tokenspeed_kernel.ops.gemm.flashinfer import has_flashinfer_fp8_blockscale

        spec = self.spec
        size = len(spec.group)
        group = pg_manager.get_device_process_group(spec.group)
        sms = torch.cuda.get_device_properties(spec.device).multi_processor_count
        if (
            spec.kind == "column"
            and size in (2, 4, 8, 16)
            and spec.input_size % 128 == 0
        ):
            if size in (2, 4) and has_flashinfer_fp8_blockscale():
                self._gather_quant = TrtllmAllGatherQuantState(
                    group, min(spec.max_tokens, 128), spec.input_size, spec.device, sms
                )
                self._gather = self._gather_quant
            else:
                self._gather = TrtllmAllGatherState(
                    group, min(spec.max_tokens, 128), spec.input_size, spec.device, True
                )
        if (
            spec.kind == "row"
            and use_reduction
            and size in (2, 4, 8, 16)
            and spec.output_size % 8 == 0
        ):
            self._reduction = TrtllmReduceScatterState(
                group, min(spec.max_tokens, 128), spec.output_size, spec.device
            )
        channels = spec.input_size if spec.kind == "row" else spec.output_size
        if size == 4 and channels % 8 == 0:
            self._a2a = TokenSpeedA2ALamportState(
                group, min(spec.max_tokens, 512), channels, spec.device, min(128, sms)
            )
            if channels % 32 == 0:
                self._a2a.prepare_chunk_exchange(threshold_bytes=8 * 2**20 + 1)
            if spec.kind == "row" and channels % 512 == 0:
                self._a2a.prepare_fp8_quantization()

    def _check(self, rows: int) -> None:
        if self._closed or not 0 < rows <= self.spec.max_tokens:
            raise RuntimeError("Projection communication is closed or exceeds capacity")

    def _pad(self, inputs: torch.Tensor, rows: int) -> torch.Tensor:
        if (
            inputs.shape[0] == rows
            and inputs.is_contiguous()
            and inputs.data_ptr() % 16 == 0
        ):
            return inputs
        padded = self._send[: rows * self.spec.input_size].view(
            rows, self.spec.input_size
        )
        padded.zero_()
        padded[: inputs.shape[0]].copy_(inputs)
        return padded

    def all_gather(
        self, inputs: torch.Tensor, rows: int, quantize: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._check(rows)
        send = self._pad(inputs, rows)
        if self._gather is not None and rows <= self._gather.max_rows:
            from tokenspeed_kernel.ops.communication.trtllm import (
                trtllm_allgather,
                trtllm_allgather_fp8_quantize,
            )

            if quantize and self._gather_quant is not None:
                return trtllm_allgather_fp8_quantize(self._gather_quant, send)
            return trtllm_allgather(self._gather, send), None
        gathered = self._gathered[: len(self.spec.group) * rows]
        self._fallback.all_gather_single(gathered, send, self.spec.group)
        return gathered, None

    def all_to_all(
        self,
        inputs: torch.Tensor,
        rows: int,
        inverse: bool,
        quantize: bool,
        out: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from tokenspeed_kernel.ops.communication.triton import (
            triton_pack_channel_shards_for_a2a,
        )

        self._check(rows)
        if inverse and (quantize or out is None):
            raise ValueError("Inverse A2A requires owned output and no quantization")
        if not inverse and out is not None:
            # Forward may return BF16 or FP8 borrowed scratch. Do not silently
            # discard a destination when fusion or a fallback changes the path.
            raise ValueError("Forward A2A returns borrowed output; out must be None")
        if self._a2a is not None and rows <= self._a2a.max_rows:
            from tokenspeed_kernel.ops.communication.cuda import (
                tokenspeed_a2a_lamport,
                tokenspeed_a2a_lamport_fp8_quantize,
            )

            sent = inputs.contiguous() if inverse else self._pad(inputs, rows)
            if quantize and self._a2a.fp8_output is not None:
                return tokenspeed_a2a_lamport_fp8_quantize(self._a2a, sent)
            return (
                tokenspeed_a2a_lamport(self._a2a, sent, inverse=inverse, out=out),
                None,
            )
        size = len(self.spec.group)
        width = self.spec.output_size if inverse else self.spec.input_size
        shard = width // size
        received = self._received[: rows * width].view(size * rows, shard)
        if inverse:
            sent = inputs.contiguous()
        else:
            scratch = self._send[: rows * width].view(size, rows, shard)
            sent = triton_pack_channel_shards_for_a2a(inputs, scratch)
        self._fallback.all_to_all_single(
            received,
            sent,
            self.spec.group,
            output_split_sizes=None,
            input_split_sizes=None,
        )
        if inverse:
            out.view(rows, size, shard).copy_(
                received.view(size, rows, shard).transpose(0, 1)
            )
            return out, None
        return received, None

    def acquire_output(self, rows: int) -> torch.Tensor:
        self._check(rows)
        if self._reduction is not None and rows <= self._reduction.max_rows:
            return self._reduction.input_buffer(rows)
        return self._partial[: len(self.spec.group) * rows]

    def reduce_scatter(self, partial: torch.Tensor, rows: int) -> torch.Tensor:
        self._check(rows)
        if self._reduction is not None and rows <= self._reduction.max_rows:
            from tokenspeed_kernel.ops.communication.trtllm import trtllm_reduce_scatter

            return trtllm_reduce_scatter(self._reduction, partial, rows)
        return self._fallback.reduce_scatter(partial, self.spec.group)

    def close(self) -> None:
        if self._closed:
            return
        if self.spec.device.type == "cuda":
            torch.cuda.synchronize(self.spec.device)
        dist.barrier(group=pg_manager.get_device_process_group(self.spec.group))
        self._a2a = None
        if self._gather is not None:
            self._gather.close()
            self._gather = self._gather_quant = None
        if self._reduction is not None:
            self._reduction.close()
            self._reduction = None
        self._closed = True
