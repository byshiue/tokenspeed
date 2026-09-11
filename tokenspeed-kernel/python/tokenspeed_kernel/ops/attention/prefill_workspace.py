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

"""Forward-owned scratch and opaque plans for sequential KDA prefill layers."""

from collections.abc import Callable

import torch


class KdaPrefillWorkspace:
    """Reuse scratch, never recurrent state, within one immutable forward.

    Args:
        cu_seqlens: The forward's device int64 sequence boundaries.
        cu_seqlens_cpu: Equal host int64 boundaries. Both tensors must remain
            unchanged for the entire forward, including queued GPU consumers.

    Create a fresh owner for every forward, even when reusing a boundary
    buffer. Scratch is separate per head count and CUDA stream; ordered layers
    on the same stream may overwrite it after the preceding scan consumes it.
    No result tensors or request cache pages are retained here.
    """

    def __init__(self, cu_seqlens: torch.Tensor, cu_seqlens_cpu: torch.Tensor):
        if (
            cu_seqlens.ndim != 1
            or cu_seqlens.dtype != torch.int64
            or cu_seqlens_cpu.device.type != "cpu"
            or cu_seqlens_cpu.dtype != torch.int64
            or cu_seqlens_cpu.shape != cu_seqlens.shape
        ):
            raise ValueError(
                "KDA workspace requires matching device/host int64 boundaries"
            )
        self.boundaries = cu_seqlens
        self._host_boundaries = tuple(cu_seqlens_cpu.tolist())
        self._scratch: dict[tuple[int, int], torch.Tensor | None] = {}
        self._plans: dict[tuple[int, int], object] = {}

    def _cache_key(
        self, boundaries: torch.Tensor, heads: int
    ) -> tuple[int, int] | None:
        if boundaries is not self.boundaries:
            raise ValueError(
                "KDA workspace belongs to a different forward's boundaries"
            )
        if boundaries.device.type != "cuda":
            return heads, 0
        if torch.cuda.is_current_stream_capturing():
            return None
        return heads, torch.cuda.current_stream(boundaries.device).cuda_stream

    def get_plan(
        self,
        boundaries: torch.Tensor,
        heads: int,
        prepare: Callable[..., object],
    ) -> object | None:
        """Return a stream-local native plan, or None inside CUDA capture.

        Args:
            boundaries: This owner's immutable device boundaries.
            heads: Native head count.
            prepare: Native plan builder, receiving the snapshotted host hint.

        Native plans own ordinary-pool views. A capture uses the ordinary
        per-call preparation path instead; no graph-private plan can escape
        through this owner. The native wrapper owns all plan internals.
        """
        key = self._cache_key(boundaries, heads)
        if key is None:
            return None
        if key not in self._plans:
            self._plans[key] = prepare(
                boundaries, heads, cu_seqlens_cpu=self._host_boundaries
            )
        return self._plans[key]

    def get(
        self,
        boundaries: torch.Tensor,
        heads: int,
        workspace_size: Callable[..., int],
    ) -> torch.Tensor | None:
        """Return this stream's scratch, or None for a zero-byte engine route.

        Args:
            boundaries: Must be this owner's immutable boundary tensor.
            heads: Native scan head count.
            workspace_size: Native size query; remains the sole route and
                capacity authority, receiving the snapshotted host hint.

        Capture allocations are deliberately not cached: graph-private pool
        storage must never escape into a later eager call through this owner.
        """
        key = self._cache_key(boundaries, heads)
        device = boundaries.device
        if key is not None and key in self._scratch:
            return self._scratch[key]
        size = workspace_size(boundaries, heads, cu_seqlens_cpu=self._host_boundaries)
        scratch = torch.empty(size, dtype=torch.uint8, device=device) if size else None
        if key is not None:
            self._scratch[key] = scratch
        return scratch
