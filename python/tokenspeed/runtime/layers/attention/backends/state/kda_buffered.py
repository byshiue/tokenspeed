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
in LCM fields. The serving backend stays gated until exact endpoint handoff
and failure feedback are integrated. No ordinary/speculative or eager/graph fork.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.attention.kda._triton.buffered import (
    buffered_recurrent,
    validate_recurrent_blocks,
)
from tokenspeed_kernel.ops.attention.kda._triton.buffered_conv import (
    buffered_conv,
    commit_conv_windows,
    prepare_conv_blocks,
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
from tokenspeed.runtime.utils.cuda_stream import StreamFork


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
    conv_read: torch.Tensor
    conv_writes: torch.Tensor


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
        self.conv_read = torch.full(shape, -1, dtype=torch.int32, device=device)
        self.conv_writes = torch.full(
            (*shape, layout.max_window), -1, dtype=torch.int32, device=device
        )
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
                    conv_read=self.conv_read[index, :bs],
                    conv_writes=self.conv_writes[index, :bs],
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

            # Convolution reads at e, not at the lagging recurrent checkpoint c.
            # Its possible acceptance destinations must be backed even when
            # this round does not flush recurrent state.
            prepare_conv_blocks(
                view.state_table,
                view.end,
                view.width,
                view.ok,
                view.conv_read,
                view.conv_writes,
                blocks=self._state_counts[gid],
                grain=self._state_grains[gid],
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


class KDAReplayWorkspace:
    """One width-parameterized conv/gate/recurrent forward and accepted commit.

    Persistent request data belongs to the bound pool. This owner keeps only
    per-round raw candidates, shared output scratch, descriptors and metadata.
    Call refresh/prepare once, forward for every local layer, then commit once.
    Outputs alias shared scratch and must be consumed before the next layer.
    Serving dispatch remains gated on endpoint materialization/publication.
    """

    def __init__(
        self, pool: HybridKDATokenToKVPool, *, max_bs: int, max_context_len: int
    ) -> None:
        self.metadata = KDAReplayMetadata(
            pool, max_bs=max_bs, max_context_len=max_context_len
        )
        self.layer_ids = tuple(pool.state_group_by_layer)
        self._row = {layer: row for row, layer in enumerate(self.layer_ids)}
        self._state = {layer: pool.get_state_buffers(layer) for layer in self.layer_ids}
        self._history = {
            layer: pool.get_replay_buffers(layer) for layer in self.layer_ids
        }
        conv, recurrent = self._state[self.layer_ids[0]]
        if not conv.is_cuda or conv.dtype != torch.bfloat16:
            raise ValueError("buffered workspace requires BF16 GPU convolution state")
        self.heads, self.value_dim, self.key_dim = recurrent.shape[1:]
        self.channels = self.heads * (2 * self.key_dim + self.value_dim)
        if conv.shape[1:] != (self.channels, 3):
            raise ValueError("buffered KDA requires a four-tap convolution")
        for current_conv, current_state in self._state.values():
            if (
                current_conv.shape[1:] != conv.shape[1:]
                or current_conv.stride() != conv.stride()
                or current_conv.dtype != conv.dtype
                or current_conv.device != conv.device
                or current_state.shape[1:] != recurrent.shape[1:]
                or current_state.dtype != torch.float32
            ):
                raise ValueError(
                    "buffered workspace requires uniform local layer geometry"
                )
        self._conv_strides = conv.stride()
        layout = self.metadata.layout
        common = (max_bs, layout.max_window)
        device = conv.device
        self.payload = torch.empty(
            (len(self.layer_ids), *common, self.channels),
            dtype=torch.bfloat16,
            device=device,
        )
        self.conv_output = torch.empty(
            (*common, self.channels), dtype=torch.bfloat16, device=device
        )
        self.gate_output = torch.empty(
            (*common, self.heads, self.key_dim), dtype=torch.bfloat16, device=device
        )
        self.output = torch.empty(
            (*common, self.heads, self.value_dim), dtype=torch.bfloat16, device=device
        )
        self.conv_ptrs = torch.tensor(
            [self._state[layer][0].data_ptr() for layer in self.layer_ids],
            dtype=torch.int64,
            device=device,
        )
        group_rows = {gid: row for row, gid in enumerate(self.metadata._groups)}
        self.group_indices = torch.tensor(
            [group_rows[self._history[layer].group_id] for layer in self.layer_ids],
            dtype=torch.int32,
            device=device,
        )
        self._producer_stream = torch.cuda.Stream(device=device, priority=-1)
        self._forks = {
            layer: StreamFork(self._producer_stream) for layer in self.layer_ids
        }
        self._views: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}

    @property
    def nbytes(self) -> int:
        """Allocated tensor bytes, excluding LCM fields and CUDA event overhead."""
        meta = self.metadata
        return sum(
            t.nbytes
            for t in (
                self.payload,
                self.conv_output,
                self.gate_output,
                self.output,
                self.conv_ptrs,
                self.group_indices,
                meta.tables.tables,
                meta.tables.decode_locs,
                meta.tables.page_sizes,
                meta.end,
                meta.width,
                meta.checkpoint,
                meta.length,
                meta.flushed,
                meta.ok,
                meta.conv_read,
                meta.conv_writes,
            )
        )

    def _layer_views(self, layer_id: int, bs: int) -> tuple[torch.Tensor, ...]:
        key = (layer_id, bs)
        self.metadata.layer(layer_id, bs)  # Validate layer and runtime batch capacity.
        if key not in self._views:
            conv = self.conv_output[:bs]
            q, k, v = conv.split(
                (
                    self.heads * self.key_dim,
                    self.heads * self.key_dim,
                    self.heads * self.value_dim,
                ),
                dim=-1,
            )
            window = self.metadata.layout.max_window
            self._views[key] = (
                conv,
                self.payload[self._row[layer_id], :bs],
                q.view(bs, window, self.heads, self.key_dim),
                k.view(bs, window, self.heads, self.key_dim),
                v.view(bs, window, self.heads, self.value_dim),
                self.gate_output[:bs],
                self.output[:bs],
            )
        return self._views[key]

    def forward(
        self,
        layer_id: int,
        bs: int,
        raw_qkv: torch.Tensor,
        conv_weight: torch.Tensor,
        f_a: torch.Tensor,
        f_b_weight: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        lower_bound: float | None,
    ) -> torch.Tensor:
        """Run one layer from BF16 raw QKV and low-rank gate producers.

        raw_qkv is [B,T,C], f_a [B*T,rank], f_b_weight [H*K,rank], beta logits
        [B,T,H], A_log FP32 [H], dt_bias FP32 [H*K]. Metadata preparation must
        have validated the group before this call. The conv producer captures
        raw candidates; gate GEMM overlaps it using the existing StreamFork
        protocol. All outputs have stable, preallocated storage on both paths.
        """
        conv_out, payload, q, k, v, gate, out = self._layer_views(layer_id, bs)
        if (
            raw_qkv.shape != conv_out.shape
            or f_a.ndim != 2
            or f_a.shape[0] != bs * self.metadata.layout.max_window
            or f_b_weight.shape != (self.heads * self.key_dim, f_a.shape[1])
            or f_a.dtype != torch.bfloat16
            or f_b_weight.dtype != torch.bfloat16
        ):
            raise ValueError("buffered raw/gate input geometry is inconsistent")
        meta = self.metadata.layer(layer_id, bs)
        conv, state = self._state[layer_id]
        with self._forks[layer_id].scope(enable=True, overlap=True) as fork:
            with fork.branch():
                torch.mm(
                    f_a, f_b_weight.t(), out=gate.view(-1, self.heads * self.key_dim)
                )
            buffered_conv(
                raw_qkv,
                conv_weight,
                conv,
                meta.conv_read,
                meta.width,
                meta.ok,
                conv_out,
                payload,
            )
        history = self._history[layer_id]
        buffered_recurrent(
            q,
            k,
            v,
            gate,
            beta,
            state,
            history.key,
            history.correction,
            history.decay,
            meta.history_table,
            meta.state_table,
            meta.end,
            meta.checkpoint,
            meta.length,
            meta.width,
            meta.flushed,
            meta.ok,
            out,
            capacity=history.layout.capacity,
            state_block_tokens=self.metadata._state_grains[history.group_id],
            transform_inputs=True,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
        )
        return out

    def commit(self, bs: int, accepted: torch.Tensor) -> None:
        """Commit accepted conv windows, then every local layer's replay stamp.

        Call only after all local layer forwards complete, with live accepted
        input counts (no added token). This does not materialize an endpoint
        for external consumers or grant snapshot publication provenance.
        """
        if accepted.numel() > bs or bs > self.metadata.max_bs:
            raise ValueError("acceptance or batch exceeds prepared capacity")
        commit_conv_windows(
            self.payload,
            self.conv_ptrs,
            self.group_indices,
            self.metadata.conv_read,
            self.metadata.conv_writes,
            self.metadata.width,
            self.metadata.ok,
            accepted,
            conv_strides=self._conv_strides,
        )
        self.metadata.commit(bs, accepted)
