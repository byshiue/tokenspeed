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

"""One-kernel quantize/exchange, FP8 blockwise GEMM, and inverse A2A experiment."""

import argparse
import json
import os
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass.cute.runtime import from_dlpack
from support import initialize, measure, module_from_path, reference
from tokenspeed_kernel.ops.quantization import quantize_fp8


class FullExperiment:
    def __init__(
        self,
        x,
        b,
        sb,
        output,
        rank,
        world,
        active_clusters,
        exchange,
        kernel_path,
        tile_m,
    ):
        self.rank = rank
        self.rows = x.shape[0]
        self.blocks = active_clusters
        self.world = world
        self.b = b
        self.sb = sb
        # Experimental addressing below assumes one 128-row tile per
        # owner and a resident grid whose size is divisible by TP4.
        assert world == 4 and x.shape[0] == 128 and active_clusters % world == 0
        assert tile_m in (64, 128) and active_clusters % (world * 128 // tile_m) == 0
        assert b.shape[0] % 128 == 0 and x.shape[1] % 128 == 0
        assert (
            0
            < active_clusters
            <= torch.cuda.get_device_properties(x.device).multi_processor_count
        )
        self.a = symm_mem.empty(
            (x.shape[0] * world, x.shape[1]), dtype=torch.float8_e4m3fn, device=x.device
        )
        self.sa = symm_mem.empty(
            (x.shape[0] * world, x.shape[1] // 128),
            dtype=torch.float32,
            device=x.device,
        )
        self.scratch = symm_mem.empty(
            (x.shape[0] * world, b.shape[0]), dtype=output.dtype, device=x.device
        )
        ready_size = 16 + active_clusters * (world + 1 if exchange == "push" else 1)
        self.ready = symm_mem.empty((ready_size,), dtype=torch.int64, device=x.device)
        self.ack = symm_mem.empty(
            (16 + active_clusters,), dtype=torch.int64, device=x.device
        )
        self.input_flags = symm_mem.empty(
            (16 + active_clusters,), dtype=torch.int64, device=x.device
        )
        for flag in (self.ready, self.ack, self.input_flags):
            flag.zero_()
        tensors = (
            self.a,
            self.sa,
            self.scratch,
            self.ready,
            self.ack,
            self.input_flags,
        )
        self.handles = [symm_mem.rendezvous(t, dist.group.WORLD) for t in tensors]
        self.peer_tensors = [
            tuple(
                h.get_buffer((rank + r) % world, t.shape, t.dtype) for r in range(world)
            )
            for h, t in zip(self.handles, tensors)
        ]
        self.input_peers, self.scale_peers, self.peers = [
            tuple(from_dlpack(t.unsqueeze(-1), assumed_align=16) for t in peers)
            for peers in self.peer_tensors[:3]
        ]
        self.peer_ready, self.peer_ack, self.peer_input_flags = [
            tuple(from_dlpack(t, assumed_align=16) for t in peers)
            for peers in self.peer_tensors[3:]
        ]
        self.receive = None
        self.receive_peers = None
        if exchange in ("push", "tma"):
            self.receive = symm_mem.empty(
                output.shape, dtype=output.dtype, device=x.device
            )
            handle = symm_mem.rendezvous(self.receive, dist.group.WORLD)
            self.handles.append(handle)
            self.receive_peers = tuple(
                handle.get_buffer((rank + r) % world, output.shape, output.dtype)
                for r in range(world)
            )
            if exchange == "tma":
                self.scratch = self.receive.view(x.shape[0] * world, b.shape[0])
                self.receive_peers = tuple(
                    t[:, rank * b.shape[0] : (rank + 1) * b.shape[0]]
                    for t in self.receive_peers
                )
            self.peers = (
                from_dlpack(self.scratch.unsqueeze(-1), assumed_align=16),
                from_dlpack(self.receive.unsqueeze(-1), assumed_align=16),
                *(
                    from_dlpack(t.unsqueeze(-1), assumed_align=16)
                    for t in self.receive_peers
                ),
            )
        self.args = [
            from_dlpack(t.unsqueeze(-1), assumed_align=16)
            for t in (self.a, b, self.scratch, self.sa, sb)
        ]
        self.ready_cute = from_dlpack(self.ready, assumed_align=16)
        self.ack_cute = from_dlpack(self.ack, assumed_align=16)
        self.input_flags_cute = from_dlpack(self.input_flags, assumed_align=16)
        torch.cuda.synchronize()
        dist.barrier()
        mod = module_from_path(kernel_path.stem, kernel_path)
        kernel = mod.FusedBlockwiseGemmKernel(
            cutlass.Float32, False, (tile_m, 128), (1, 1)
        )
        self.compiled = cute.compile(
            kernel,
            *self.args,
            active_clusters,
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            from_dlpack(output, assumed_align=16),
            self.peers,
            self.ready_cute,
            self.peer_ready,
            self.ack_cute,
            self.peer_ack,
            cutlass.Int32(rank),
            from_dlpack(x, assumed_align=16),
            self.input_peers,
            self.scale_peers,
            self.input_flags_cute,
            self.peer_input_flags,
            lambda v: v,
            options="--opt-level=2"
        )
        self.resources = {
            "tile_m": tile_m,
            "accumulator_stages": kernel.num_acc_stage,
            "ab_stages": kernel.num_ab_stage,
            "epilogue_stages": kernel.num_c_stage,
            "scale_stages": kernel.num_scale_stage,
            "tile_stages": kernel.num_tile_stage,
            "tmem_columns": kernel.num_tmem_alloc_cols,
            "register_budget_accumulator": kernel.num_regs_acc_update_warps,
            "register_budget_epilogue": kernel.num_regs_epilogue_warps,
            "register_budget_producers": kernel.num_regs_uniform_warps,
        }

    def __call__(self, x, output):
        self.compiled(
            *self.args,
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            from_dlpack(output, assumed_align=16),
            self.peers,
            self.ready_cute,
            self.peer_ready,
            self.ack_cute,
            self.peer_ack,
            cutlass.Int32(self.rank),
            from_dlpack(x, assumed_align=16),
            self.input_peers,
            self.scale_peers,
            self.input_flags_cute,
            self.peer_input_flags
        )
        return output
