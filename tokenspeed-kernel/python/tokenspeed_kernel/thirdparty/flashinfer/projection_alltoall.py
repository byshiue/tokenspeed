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


"""Optional, startup-only FlashInfer Ulysses communicator construction."""

import torch
import torch.distributed as dist


def create_projection_a2a(
    group: dist.ProcessGroup,
    max_elems: int,
    dtype: torch.dtype,
    device: torch.device,
    backend: str,
):
    """Return an NVLink communicator, or None for a collectively agreed fallback.

    All subgroup ranks must call before graph capture. Missing optional packages
    are agreed before entering FlashInfer's collective constructor. FlashInfer
    itself coordinates topology checks and initialization failure cleanup.
    """
    communicator = None
    reason = None
    try:
        from flashinfer.comm.ulysses import UlyssesCommunicator

        communicator = UlyssesCommunicator
    except ImportError as exc:
        reason = str(exc)
    reasons = [None] * group.size()
    dist.all_gather_object(reasons, reason, group=group)
    if any(item is not None for item in reasons):
        if backend == "flashinfer":
            raise RuntimeError(f"FlashInfer projection A2A unavailable: {reasons}")
        return None, f"FlashInfer projection A2A unavailable: {reasons}"
    comm = communicator(
        group=group,
        max_elems=max_elems,
        dtype=dtype,
        device=device,
        backend="nvlink" if backend == "flashinfer" else "auto",
    )
    if comm.backend != "nvlink":
        reason = comm.fallback_reason
        comm.close()
        return None, reason
    return comm, None


def flashinfer_projection_a2a(comm, inputs: torch.Tensor) -> torch.Tensor:
    """Gather tokens and shard channels, returning rank-major [P*N, K/P].

    inputs is contiguous [N,K] with equal positive N across the subgroup.
    Use 128-element heads where possible; eight-element lanes also meet the
    NVLink kernel's 16-byte BF16 alignment for smaller projection dimensions.
    The caller owns communicator lifetime and serializes its stream use.
    """
    rows, channels = inputs.shape
    head_dim = 128 if channels % (128 * comm.world_size) == 0 else 8
    return comm.scatter_heads(
        inputs.view(1, rows, channels // head_dim, head_dim)
    ).view(rows * comm.world_size, channels // comm.world_size)
