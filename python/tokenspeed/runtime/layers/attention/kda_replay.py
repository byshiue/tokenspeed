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


"""Storage geometry for the experimental cache-owned KDA replay history."""

from dataclasses import dataclass

# Independent of capacity and prefix identity. Kernels consume explicit strides
# and raw block tables; a block contains this many per-token history entries.
KDA_REPLAY_BLOCK_TOKENS = 8


def kda_buffered_workspace_bytes(
    *,
    layers: int,
    max_bs: int,
    max_context_len: int,
    max_window: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    groups: int,
    state_grain: int,
    history_block_tokens: int,
) -> int:
    """Tensor-storage budget for buffered decode, outside the LCM arena.

    Count BF16 raw candidates/output plus shared conv/gate scratch (FP32 for
    multi-token windows, BF16 for width one),
    int64 conv pointers and int32 group indices, and the metadata owner's raw
    table stacks/position buffers. Geometry is per rank and startup-fixed;
    max_bs is runtime capacity, not the capture ladder. CUDA stream/event
    implementation overhead is not tensor storage and is not included here.
    """
    if (
        min(
            layers,
            max_bs,
            max_context_len,
            max_window,
            heads,
            key_dim,
            value_dim,
            groups,
            state_grain,
            history_block_tokens,
        )
        < 1
    ):
        raise ValueError("positive buffered workspace geometry required")
    channels = heads * (2 * key_dim + value_dim)
    producer_itemsize = 4 if max_window > 1 else 2
    activations = (
        max_bs
        * max_window
        * (
            (layers * channels + heads * value_dim) * 2
            + (channels + heads * key_dim) * producer_itemsize
        )
    )
    descriptors = layers * (8 + 4 + 6 * 8)
    raw_groups = 2 * groups
    table_grain = min(state_grain, history_block_tokens)
    columns = (max_context_len + max_window + table_grain - 1) // table_grain
    # GroupTableStacks also owns decode_locs and the page_sizes vector, even
    # though this consumer uses only its ratio-one raw-table fill.
    tables = raw_groups * (max_bs * columns + max_bs * max_window + 1) * 4
    positions = max_bs * (9 + groups * (19 + 4 * max_window))
    return activations + descriptors + tables + positions


@dataclass(frozen=True, kw_only=True)
class KDAReplayLayout:
    """Startup-fixed logical capacity and maximum target execution width.

    Physical storage also includes block rounding and scheduling protection;
    capacity counts recurrence entries, not allocator blocks.
    """

    capacity: int
    max_window: int
    block_tokens: int

    def __post_init__(self) -> None:
        for name in ("capacity", "max_window", "block_tokens"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 < value <= 2**31 - 1
            ):
                raise ValueError(f"{name} must be a positive int32 integer")
        if self.capacity < 2 * self.max_window:
            raise ValueError("replay capacity must cover two maximum execution windows")

    @property
    def max_state_lag_tokens(self) -> int:
        return self.capacity - self.max_window
