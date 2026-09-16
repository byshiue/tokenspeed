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

"""Paged latent-KV and KDA-state cache."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import torch

from tokenspeed.runtime.layers.attention.kda_replay import KDAReplayLayout
from tokenspeed.runtime.layers.attention.kv_cache.base import (
    derive_state_groups_by_layer,
)
from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    cache_field_layer_id,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    STATE_LAYER_TYPES,
)


@dataclass(frozen=True, kw_only=True)
class KDAReplayLayer:
    """Zero-copy views of one layer's LCM-owned replay fields.

    Floating fields are [blocks, rows, heads, channels]. Checkpoint stamps
    are int64 [blocks, rows], encoding c+1 at the last committed input row;
    zero denotes an empty history seeded by prefill. Strides are explicit.
    """

    group_id: str
    checkpoint_group_id: str
    layout: KDAReplayLayout
    key: torch.Tensor
    correction: torch.Tensor
    decay: torch.Tensor
    checkpoint: torch.Tensor


class HybridKDATokenToKVPool(MLATokenToKVPool):
    """MLA compute interface whose latent KV and KDA state share one buffer."""

    #: Sanitizing latent write: under the prefill graph a padded row's NaN
    #: reaches a live row's softmax through the shared dummy slot, because the
    #: paged MLA decode computes ``q·k`` before the causal mask.
    latent_write_sanitizes: ClassVar[bool] = True

    def __init__(
        self,
        *,
        layer_types: tuple[str, ...],
        **kwargs,
    ):
        self._layer_types = tuple(layer_types)
        self._state_buffers_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.requires_page_zeroing = True

        if len(self._layer_types) != kwargs["layer_num"]:
            raise ValueError("cache layer types must cover every model layer")

        super().__init__(**kwargs)

    # A KDA layer has conv/recurrent planes planned and an MLA layer has a
    # latent plane; the plan's field list decides per layer. Latent-page
    # contiguity is a plan invariant (exact_page_stride).
    # State planes keep their planned shape: the KDA decode ABI reads them
    # as the plan lays them out.
    layer_plane_bindings: ClassVar[dict[str, str]] = {
        **MLATokenToKVPool.layer_plane_bindings,
        "conv_state": "_conv_state",
        "recurrent_state": "_recurrent_state",
        "replay_key": "_replay_key",
        "replay_correction": "_replay_correction",
        "replay_decay": "_replay_decay",
        "replay_checkpoint": "_replay_checkpoint",
    }

    def _bind_layer_planes(self) -> None:
        if self.quant_method == "per_token_head":
            raise ValueError("KDA cache does not support per-token-head KV")
        super()._bind_layer_planes()
        # A state layer with no planned planes belongs to another pipeline
        # stage (the PP-narrowed plan drops its fields); this view never
        # executes it, so it gets no entry.
        self._state_buffers_by_layer = {
            layer_id: (self._conv_state[layer_id], self._recurrent_state[layer_id])
            for layer_id, label in enumerate(self._layer_types)
            if label in STATE_LAYER_TYPES and self._conv_state[layer_id] is not None
        }
        self._replay_buffers_by_layer: dict[int, KDAReplayLayer] = {}
        specs = {spec.group_id: spec for spec in self.arena.cache_group_specs}
        replay_fields: dict[int, dict[str, str]] = {}
        for field in self.arena.plan.fields:
            suffix = field.field_id.rsplit(".", 1)[-1]
            if suffix not in (
                "replay_key",
                "replay_correction",
                "replay_decay",
                "replay_checkpoint",
            ):
                continue
            layer_id = cache_field_layer_id(field.field_id) - self._field_layer_offset
            if not 0 <= layer_id < self.layer_num:
                continue
            replay_fields.setdefault(layer_id, {})[suffix] = field.group_id
        for layer_id, fields in replay_fields.items():
            if len(fields) != 4 or len(set(fields.values())) != 1:
                raise ValueError(
                    "all four KDA replay fields must belong to the same history group"
                )
            spec = specs[fields["replay_key"]]
            state_group = self.state_group_by_layer.get(layer_id)
            if (
                state_group is None
                or spec.family != "history"
                or spec.replay_checkpoint_group != state_group
            ):
                raise ValueError(
                    "KDA replay fields must depend on their layer's state group"
                )
            layout = KDAReplayLayout(
                capacity=spec.sliding_window_tokens,
                max_window=spec.sliding_window_tokens
                - specs[state_group].max_state_lag_tokens,
                block_tokens=spec.rows_per_page,
            )
            key, correction, decay, checkpoint = (
                self._replay_key[layer_id],
                self._replay_correction[layer_id],
                self._replay_decay[layer_id],
                self._replay_checkpoint[layer_id],
            )
            heads, value_dim, key_dim = self._recurrent_state[layer_id].shape[1:]
            count = self.arena.cache_group_page_counts[spec.group_id]
            for tensor, shape, dtype in (
                (key, (count, layout.block_tokens, heads, key_dim), torch.float32),
                (
                    correction,
                    (count, layout.block_tokens, heads, value_dim),
                    torch.float32,
                ),
                (decay, (count, layout.block_tokens, heads, key_dim), torch.float32),
                (checkpoint, (count, layout.block_tokens), torch.int64),
            ):
                if tensor is None or tensor.shape != shape or tensor.dtype != dtype:
                    raise ValueError(
                        "KDA replay fields do not match the planned history geometry"
                    )
            self._replay_buffers_by_layer[layer_id] = KDAReplayLayer(
                group_id=spec.group_id,
                checkpoint_group_id=state_group,
                layout=layout,
                key=key,
                correction=correction,
                decay=decay,
                checkpoint=checkpoint,
            )

    @cached_property
    def state_group_by_layer(self) -> dict[int, str]:
        """View-local state layer id -> its state-family cache group id."""
        return derive_state_groups_by_layer(
            self.arena,
            first_layer=self._field_layer_offset,
            num_layers=self.layer_num,
            state_layer_ids=(
                layer_id
                for layer_id, label in enumerate(self._layer_types)
                if label in STATE_LAYER_TYPES
            ),
            state_field_suffixes=("conv_state", "recurrent_state"),
        )

    def get_component(self, layer_id: int, component_name: str) -> torch.Tensor:
        """Return one KDA state plane. Latent KV is read via ``kv_buffer``."""
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
        try:
            conv, recurrent = self._state_buffers_by_layer[layer_id]
        except KeyError as exc:
            raise ValueError(f"layer {layer_id} has no KDA state") from exc
        if component_name == "conv_state":
            return conv
        if component_name == "recurrent_state":
            return recurrent
        raise ValueError(f"unknown KDA component {component_name!r}")

    def get_state_buffers(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return self._state_buffers_by_layer[layer_id]
        except KeyError as exc:
            raise ValueError(f"layer {layer_id} has no KDA state") from exc

    def get_replay_buffers(self, layer_id: int) -> KDAReplayLayer:
        """Return bound history views, fencing this layer's first cache access.

        Raises ValueError for a layer without replay fields in this pool view.
        No history storage is allocated by this accessor or by the pool.
        """
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
        try:
            return self._replay_buffers_by_layer[layer_id]
        except KeyError as exc:
            raise ValueError(f"layer {layer_id} has no buffered KDA history") from exc

    def zero_new_blocks(self, new_page_ids: dict[str, list[int]]) -> None:
        if new_page_ids:
            self.arena.zero_blocks(new_page_ids)

    def get_kv_size_bytes(self):
        return self.arena.buffer.nbytes
