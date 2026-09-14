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

"""Opt-in capacity-based KDA graph cache for breakable-prefill experiments."""

import logging
from dataclasses import dataclass, fields, is_dataclass, replace

import torch
from tokenspeed_kernel.ops.attention.gdn.triton import (
    CAUSAL_CONV1D_BLOCK_M,
    build_causal_conv1d_prefill_metadata,
    refresh_causal_conv1d_capacity_metadata,
)
from tokenspeed_kernel.ops.attention.kda import KdaPrefillCapacity
from tokenspeed_kernel.platform import pdl_enabled

from tokenspeed.runtime.layers.attention.backends.state.mamba import (
    MambaForwardMetadata,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class KdaPrefillGraphMetadata(MambaForwardMetadata):
    capacity: KdaPrefillCapacity

    @property
    def prefill_token_extent(self) -> int:
        return self.capacity.token_capacity


def _capacity_metadata(source, bucket):
    capacity = KdaPrefillCapacity(bucket, source.extend_seq_lens_cpu.numel())
    capacity.validate(source.cu_extend_seq_lens_cpu, bucket)
    cloned = _clone_metadata(source)
    result = KdaPrefillGraphMetadata(
        **{
            field.name: getattr(cloned, field.name)
            for field in fields(MambaForwardMetadata)
        },
        capacity=capacity,
    )
    # Build immutable per-request capacity maps once, independently of live
    # packed boundaries. Overscheduled conv programs honor the live bounds.
    host_bounds = capacity.boundaries_cpu()
    result.conv_prefill_metadata = build_causal_conv1d_prefill_metadata(
        host_bounds.to(device=source.query_start_loc.device, dtype=torch.int32),
        host_bounds[1:] - host_bounds[:-1],
        CAUSAL_CONV1D_BLOCK_M,
    )
    refresh_causal_conv1d_capacity_metadata(
        result.query_start_loc, result.conv_prefill_metadata, bucket
    )
    return result


def _clone_metadata(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_metadata(item) for key, item in value.items()}
    if is_dataclass(value):
        return replace(
            value,
            **{
                field.name: _clone_metadata(getattr(value, field.name))
                for field in fields(value)
            },
        )
    return value


def _argument_key(value):
    if isinstance(value, torch.Tensor):
        return (
            value.data_ptr(),
            tuple(value.shape),
            value.stride(),
            value.dtype,
            value.device,
        )
    return value


class KdaPrefillGraphCache:
    """Capture extend; retain max_shapes schedules per live sequence count.

    Caller supplies only pure EXTEND work inside breakable graph replay.
    Outputs have the usual break-output lifetime: consume before the next call.
    The cache owns execution metadata, not scheduler state or cache pages.
    """

    def __init__(self, max_shapes):
        self.max_shapes = max_shapes
        self.schedules = {}
        self.last_source = None
        self.last_key = None
        self.captures = 0
        self.replays = 0
        self._capture_resources = {}

    def run(self, backend, layer_id, bucket, arguments, forward):
        source = backend.forward_metadata
        stream = torch.cuda.current_stream()
        if source is not self.last_source:
            self.last_key = (
                source.extend_seq_lens_cpu.numel(),
                bucket,
                stream.cuda_stream,
                pdl_enabled(),
            )
            self.last_source = source
        # A single metadata object may also be used with another padded bucket.
        key = (self.last_key[0], bucket, stream.cuda_stream, pdl_enabled())
        schedule = self.schedules.get(key)
        if schedule is None:
            if sum(item[0] == key[0] for item in self.schedules) >= self.max_shapes:
                return forward()
            schedule = {
                "metadata": _capacity_metadata(source, bucket),
                "source": source,
                "layers": {},
            }
            self.schedules[key] = schedule
        elif schedule["source"] is not source:
            target = schedule["metadata"]
            target.capacity.validate(source.cu_extend_seq_lens_cpu, bucket)
            for name in (
                "query_start_loc",
                "query_start_loc_int64",
                "extend_seq_lens_cpu",
                "cu_extend_seq_lens_cpu",
            ):
                getattr(target, name).copy_(getattr(source, name))
            refresh_causal_conv1d_capacity_metadata(
                target.query_start_loc, target.conv_prefill_metadata, bucket
            )
            # Refresh once on the consumer stream; all layers reuse the maps.
            for name in ("state_in_blocks_by_group", "state_out_blocks_by_group"):
                old = getattr(schedule["metadata"], name)
                new = getattr(source, name)
                if old.keys() != new.keys():
                    raise RuntimeError(
                        "KDA graph state groups changed without pool rebind"
                    )
                for group, indices in old.items():
                    if (
                        indices.shape != new[group].shape
                        or indices.dtype != new[group].dtype
                    ):
                        raise RuntimeError("KDA graph state index geometry changed")
                    indices.copy_(new[group])
            schedule["source"] = source

        signature = tuple(
            (name, _argument_key(value)) for name, value in sorted(arguments.items())
        )
        entry = schedule["layers"].get(layer_id)
        if entry is not None and entry["signature"] != signature:
            # Keep the original tensor objects alive: an allocator-recycled
            # pointer must never turn a different argument into a cache hit.
            return forward()
        original = backend.forward_metadata
        backend.forward_metadata = schedule["metadata"]
        try:
            if entry is None:
                result = forward()  # Ordinary first execution warms native plans.
                schedule["layers"][layer_id] = {
                    "signature": signature,
                    "arguments": dict(arguments),
                    "graph": None,
                    "output": None,
                }
                return result
            if entry["graph"] is None:
                graph = torch.cuda.CUDAGraph()
                # Only serial consumers share scratch. Keep separate pools for
                # different replay streams, and never borrow the outer graph's
                # pool: its live intermediates span this attention break.
                resource_key = (stream.device, stream.cuda_stream)
                resources = self._capture_resources.get(resource_key)
                if resources is None:
                    resources = (
                        torch.cuda.graph_pool_handle(),
                        torch.cuda.Stream(device=stream.device),
                    )
                    self._capture_resources[resource_key] = resources
                pool, capture_stream = resources
                capture_stream.wait_stream(stream)
                # Capture records work but does not execute cache writes. Replay
                # exactly once; never warm up again against an in-place state.
                with torch.cuda.graph(
                    graph,
                    pool=pool,
                    stream=capture_stream,
                    capture_error_mode="thread_local",
                ):
                    output = forward()
                stream.wait_stream(capture_stream)
                entry["graph"], entry["output"] = graph, output
                entry["capture_stream"] = capture_stream
                self.captures += 1
                logger.info(
                    "KDA prefill subgraph captured: layer=%s bucket=%s sequences=%s",
                    layer_id,
                    bucket,
                    key[0],
                )
            entry["graph"].replay()
            self.replays += 1
            return entry["output"]
        finally:
            backend.forward_metadata = original
