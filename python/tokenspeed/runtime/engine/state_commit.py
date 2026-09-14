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

"""CPU-only agreement on deferred cache-state commit before scheduler feedback."""

from __future__ import annotations

import torch
import torch.distributed as dist


class StateCommitValidator:
    """Reject an invalid round on every participating model rank.

    ``enabled`` is a global cache-contract fact, identical across TP/PP ranks;
    ``local_groups`` counts this rank's buffered consumers. ``groups`` contains
    the nontrivial CPU TP and PP process groups, in that order. All ranks enter
    the agreement even when a pipeline stage owns no recurrent layers.

    Only already-synced CPU results enter here. A cache invariant failure is
    fatal: unlike a bad sampled token, silently continuing could publish a
    lagging checkpoint or reuse storage from an invalid execution.
    """

    def __init__(self, *, enabled: bool, local_groups: int, groups: tuple) -> None:
        self.enabled = enabled
        self.local_groups = local_groups
        self.groups = groups
        self._failed = torch.zeros(1, dtype=torch.int32)

    def validate(
        self, validity: torch.Tensor | None, *, bs: int, requires_commit: bool
    ) -> None:
        """Agree on live ``[local_groups, bs]`` validity, then fail if needed.

        The caller must have joined the output copy event. Missing decode
        flags and malformed results fail through the same collective as false
        flags, never through an early rank-local exception. Prefill has no
        deferred commit and may return None. Padding must already be sliced off.
        """
        if not self.enabled:
            return
        if validity is None:
            failed = requires_commit and self.local_groups > 0
        elif (
            not isinstance(validity, torch.Tensor)
            or validity.device.type != "cpu"
            or validity.dtype != torch.bool
            or validity.shape != (self.local_groups, bs)
        ):
            failed = True
        else:
            failed = not bool(validity.all())
        self._failed.fill_(int(failed))
        for group in self.groups:
            dist.all_reduce(self._failed, op=dist.ReduceOp.MAX, group=group)
        if self._failed.item():
            raise RuntimeError(
                "Deferred cache-state commit failed on a model rank; "
                "the round must not be published to the scheduler"
            )
