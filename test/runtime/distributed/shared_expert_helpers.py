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

"""DEP mapping fixture for shared-expert sharding validation."""

from tokenspeed.runtime.distributed.mapping import Mapping


def dep_mapping(rank: int, world: int) -> Mapping:
    return Mapping(
        rank=rank,
        world_size=world,
        attn_tp_size=1,
        attn_cp_size=1,
        attn_dp_size=world,
        attn_dcp_size=1,
        dense_tp_size=1,
        dense_dp_size=world,
        moe_tp_size=1,
        moe_ep_size=world,
        moe_dp_size=1,
        vision_tp_size=1,
        vision_dp_size=1,
        linear_attn_tp_size=1,
        pp_size=1,
        pp_layer_partition=None,
        nprocs_per_node=None,
        nnodes=None,
        base_gpu_id=0,
        gpu_id_step=1,
    )
