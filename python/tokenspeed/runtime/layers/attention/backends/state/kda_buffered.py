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

"""Fixed-address, per-group metadata for cache-owned KDA buffered replay.

This is execution scratch, not a request-state pool: stamps and history remain
in LCM fields. The serving backend stays gated until conv commit and exact
endpoint handoff are integrated. No ordinary/speculative or eager/graph fork.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.attention.kda._triton.buffered import (
    validate_recurrent_blocks,
)
from tokenspeed_kernel.ops.attention.kda._triton.buffered_metadata import (
    commit_positions,
    prepare_positions,
    refresh_decode_inputs,
)

from tokenspeed.runtime.layers.attention.backends.paged.group_tables import (
    GroupTableSpec,
    GroupTableStacks,
)
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
    HybridKDATokenToKVPool,
    KDAReplayLayer,
)


@dataclass(frozen=True, kw_only=True)
class KDAReplayGroupMetadata:
    """One group's per-batch views, shared by every local layer in that group.

    Tables retain scheduler block IDs (no page expansion). ``end``/``width``
    are shared across groups; remaining vectors are group-specific. A false
    ``ok`` suppresses recurrence/commit stores, but the owner must still reject
    the forward before publishing an endpoint. These flags are not provenance.
    """

    history_table: torch.Tensor
    state_table: torch.Tensor
    end: torch.Tensor
    width: torch.Tensor
    checkpoint: torch.Tensor
    length: torch.Tensor
    flushed: torch.Tensor
    ok: torch.Tensor


class KDAReplayMetadata:
    """Bind one pool's replay groups and allocate scratch at runtime capacity.

    ``max_bs`` is full decode capacity, not the graph capture ladder. Rebinding
    requires a new owner and recapturing graphs; no old cache views survive.
    Before any prepare, the caller must fence access to every local layer's
    transferred fields. A shared group commit follows *all* layer stores, so
    every layer's checkpoint stamp is equal at the next prepare. It is then
    sufficient to load one local layer's stamp per group (also for PP views).
    """

    def __init__(
        self, pool: HybridKDATokenToKVPool, *, max_bs: int, max_context_len: int
    ) -> None:
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0
            for v in (max_bs, max_context_len)
        ):
            raise ValueError("positive runtime batch and context capacities required")
        specs = {spec.group_id: spec for spec in pool.arena.cache_group_specs}
        layers = {
            layer: pool.get_replay_buffers(layer) for layer in pool.state_group_by_layer
        }
        if not layers:
            raise ValueError("buffered metadata requires local replay layers")
        self._groups: dict[str, list[KDAReplayLayer]] = {}
        self._state_counts: dict[str, int] = {}
        self._state_grains: dict[str, int] = {}
        self._group_by_layer = {
            layer: replay.group_id for layer, replay in layers.items()
        }
        layout = next(iter(layers.values())).layout
        for layer, replay in layers.items():
            if replay.layout != layout:
                raise ValueError(
                    "local replay groups must share capacity and maximum width"
                )
            gid = replay.group_id
            state_count = pool.get_component(layer, "recurrent_state").shape[0]
            group = self._groups.setdefault(gid, [])
            if group:
                first = group[0]
                if (
                    replay.checkpoint_group_id != first.checkpoint_group_id
                    or replay.key.shape != first.key.shape
                    or replay.checkpoint.shape != first.checkpoint.shape
                    or replay.checkpoint.stride() != first.checkpoint.stride()
                    or replay.checkpoint.device != first.checkpoint.device
                    or state_count != self._state_counts[gid]
                ):
                    raise ValueError(
                        "a replay group must have uniform local layer geometry"
                    )
            group.append(replay)
            self._state_counts[gid] = state_count
            self._state_grains[gid] = specs[
                replay.checkpoint_group_id
            ].block_granularity
        self._stamps = {
            gid: tuple(replay.checkpoint for replay in group)
            for gid, group in self._groups.items()
        }
        self.max_bs = max_bs
        self.layout = layout
        # Include the fixed input window: verify may extend past the last
        # accepted context. Ratio one reuses the common raw-table fill, without
        # inventing row geometry for a recurrent checkpoint.
        group_ids = tuple(
            dict.fromkeys(
                gid
                for group in self._groups.values()
                for gid in (group[0].group_id, group[0].checkpoint_group_id)
            )
        )
        device = next(iter(layers.values())).checkpoint.device
        self.tables = GroupTableStacks(
            tuple(
                GroupTableSpec(
                    group_id=gid,
                    block_granularity=specs[gid].block_granularity,
                    kernel_page_size=specs[gid].block_granularity,
                    max_num_pages=(
                        max_context_len
                        + layout.max_window
                        + specs[gid].block_granularity
                        - 1
                    )
                    // specs[gid].block_granularity,
                )
                for gid in group_ids
            ),
            max_bs=max_bs,
            max_tokens_per_req=layout.max_window,
            device=device,
        )
        self.end = torch.zeros(max_bs, dtype=torch.int32, device=device)
        self.width = torch.zeros_like(self.end)
        shape = (len(self._groups), max_bs)
        self.checkpoint = torch.zeros(shape, dtype=torch.int64, device=device)
        self.length = torch.zeros(shape, dtype=torch.int32, device=device)
        self.flushed = torch.zeros(shape, dtype=torch.bool, device=device)
        self.ok = torch.zeros_like(self.flushed)
        self._views: dict[int, dict[str, KDAReplayGroupMetadata]] = {}

    def _batch(self, bs: int) -> dict[str, KDAReplayGroupMetadata]:
        if not 0 <= bs <= self.max_bs:
            raise ValueError("batch exceeds allocated replay metadata capacity")
        if bs not in self._views:
            self._views[bs] = {
                gid: KDAReplayGroupMetadata(
                    history_table=self.tables.table(gid, bs),
                    state_table=self.tables.table(group[0].checkpoint_group_id, bs),
                    end=self.end[:bs],
                    width=self.width[:bs],
                    checkpoint=self.checkpoint[index, :bs],
                    length=self.length[index, :bs],
                    flushed=self.flushed[index, :bs],
                    ok=self.ok[index, :bs],
                )
                for index, (gid, group) in enumerate(self._groups.items())
            }
        return self._views[bs]

    def layer(self, layer_id: int, bs: int) -> KDAReplayGroupMetadata:
        """Return cached views; accessing another layer never launches preparation."""
        return self._batch(bs)[self._group_by_layer[layer_id]]

    def refresh(
        self,
        bs: int,
        actual_bs: int,
        seq_lens: torch.Tensor,
        block_tables: Mapping[str, torch.Tensor],
    ) -> None:
        """Refill raw tables and endpoint/width buffers for eager or graph use.

        Live delivery must include every consumed group, with unit column
        stride; reject oversized tables instead of silently truncating them.
        Idle refresh clears the requested rows. Position stamps are read later
        by ``prepare``, in the same stream order as the model forward.
        """
        self._batch(bs)
        if not 0 <= actual_bs <= bs:
            raise ValueError("need 0 <= actual_bs <= bs")
        if actual_bs:
            for gid in self.tables.group_ids:
                if gid not in block_tables:
                    raise ValueError(f"missing replay cache table {gid!r}")
                table = block_tables[gid]
                if (
                    table.ndim != 2
                    or table.shape[0] < actual_bs
                    or not 0 < table.shape[1] <= self.tables.table(gid, bs).shape[1]
                    or table.dtype != torch.int32
                    or table.stride(1) != 1
                    or table.device != self.end.device
                ):
                    raise ValueError(f"invalid replay cache table {gid!r}")
        self.tables.fill(bs, actual_bs, block_tables)
        refresh_decode_inputs(
            seq_lens,
            self.end[:bs],
            self.width[:bs],
            actual_bs=actual_bs,
            max_window=self.layout.max_window,
        )

    def prepare(self, bs: int) -> None:
        """Read and validate positions once per group before any layer consumes them."""
        for gid, view in self._batch(bs).items():
            prepare_positions(
                self._stamps[gid][0],
                view.history_table,
                view.end,
                view.width,
                view.checkpoint,
                view.length,
                view.flushed,
                view.ok,
                capacity=self.layout.capacity,
                max_window=self.layout.max_window,
            )
            validate_recurrent_blocks(
                view.history_table,
                view.state_table,
                view.end,
                view.checkpoint,
                view.length,
                view.width,
                view.flushed,
                view.ok,
                history_blocks=self._groups[gid][0].key.shape[0],
                state_blocks=self._state_counts[gid],
                history_block_tokens=self.layout.block_tokens,
                state_block_tokens=self._state_grains[gid],
                capacity=self.layout.capacity,
                max_window=self.layout.max_window,
            )

    def commit(self, bs: int, accepted: torch.Tensor) -> None:
        """Commit live acceptance after all recurrent/conv stores have completed.

        ``accepted`` contains only live requests, including the target input.
        The owner's endpoint/handoff protocol must consume ``ok`` before
        publication. This method alone neither materializes nor publishes state.
        """
        live = accepted.numel()
        if live > bs:
            raise ValueError("acceptance exceeds the prepared batch")
        for gid, view in self._batch(bs).items():
            commit_positions(
                self._stamps[gid],
                view.history_table[:live],
                view.end[:live],
                view.width[:live],
                accepted,
                view.checkpoint[:live],
                view.length[:live],
                view.flushed[:live],
                view.ok[:live],
            )
