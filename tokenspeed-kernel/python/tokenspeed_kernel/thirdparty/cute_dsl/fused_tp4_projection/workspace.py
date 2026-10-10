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

"""Model-owned symmetric storage and precompiled fixed-shape projection."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass.cute.runtime import from_dlpack
from tokenspeed_kernel.thirdparty.cute_dsl.fused_tp4_projection.optimized_kernel import (
    FusedBlockwiseGemmKernel,
)


class FusedTP4ProjectionState:
    """Collectively prepare the validated TP4 schedule before graph capture.

    Args:
        group: Four node-local GB300 ranks in output-shard order.
        device: This rank's CUDA device, with 152 SMs available to the grid.

    Sequential layers share this storage, but supply their own weights to run.
    Concurrent models/streams require separate states. No layer weight or final
    output pointer is retained. Call close collectively after destroying graphs.
    """

    def __init__(self, group: dist.ProcessGroup, device: torch.device) -> None:
        if (
            group.size() != 4
            or torch.cuda.get_device_capability(device) != (10, 3)
            or torch.cuda.get_device_properties(device).multi_processor_count != 152
        ):
            raise ValueError("Fused TP4 projection requires four 152-SM GB300 GPUs")
        self.group = group
        self.device = device
        self.rank = group.rank()
        self.closed = False
        self.a = symm_mem.empty((512, 7168), dtype=torch.float8_e4m3fn, device=device)
        self.sa = symm_mem.empty((512, 56), dtype=torch.float32, device=device)
        self.receive = symm_mem.empty((128, 49664), dtype=torch.bfloat16, device=device)
        self.ready = symm_mem.empty((168,), dtype=torch.int64, device=device)
        self.ack = symm_mem.empty((168,), dtype=torch.int64, device=device)
        self.input_flags = symm_mem.empty((168,), dtype=torch.int64, device=device)
        for flags in (self.ready, self.ack, self.input_flags):
            flags.zero_()
        tensors = (
            self.a,
            self.sa,
            self.receive,
            self.ready,
            self.ack,
            self.input_flags,
        )
        self.handles = [symm_mem.rendezvous(tensor, group) for tensor in tensors]
        self.peer_tensors = tuple(
            tuple(
                handle.get_buffer((self.rank + peer) % 4, tensor.shape, tensor.dtype)
                for peer in range(4)
            )
            for handle, tensor in zip(self.handles, tensors)
        )
        self.input_peers, self.scale_peers = (
            tuple(from_dlpack(t.unsqueeze(-1), assumed_align=16) for t in peers)
            for peers in self.peer_tensors[:2]
        )
        self.peers = (
            from_dlpack(self.receive.view(512, 12416, 1), assumed_align=16),
            from_dlpack(self.receive.unsqueeze(-1), assumed_align=16),
            *(
                from_dlpack(
                    t[:, self.rank * 12416 : (self.rank + 1) * 12416].unsqueeze(-1),
                    assumed_align=16,
                )
                for t in self.peer_tensors[2]
            ),
        )
        self.peer_ready, self.peer_ack, self.peer_input_flags = (
            tuple(from_dlpack(t, assumed_align=16) for t in peers)
            for peers in self.peer_tensors[3:]
        )
        self.a_cute = from_dlpack(self.a.unsqueeze(-1), assumed_align=16)
        self.sa_cute = from_dlpack(self.sa.unsqueeze(-1), assumed_align=16)
        self.ready_cute = from_dlpack(self.ready, assumed_align=16)
        self.ack_cute = from_dlpack(self.ack, assumed_align=16)
        self.input_flags_cute = from_dlpack(self.input_flags, assumed_align=16)
        inputs = torch.zeros((128, 7168), dtype=torch.bfloat16, device=device)
        weight = torch.zeros((12416, 7168), dtype=torch.float8_e4m3fn, device=device)
        scales = torch.ones((97, 56), dtype=torch.float32, device=device)
        out = torch.empty_like(self.receive)
        kernel = FusedBlockwiseGemmKernel(cutlass.Float32, False, (128, 128), (1, 1))
        args = self._arguments(inputs, weight, scales, out)
        self.compiled = cute.compile(
            kernel, *args[:5], 152, *args[5:], lambda v: v, options="--opt-level=2"
        )
        # A peer must finish zeroing its flags before another rank can publish.
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        # Exercise the compiled launch and mappings outside capture. Device epochs
        # continue across warmup, eager calls, graph capture and graph replay.
        self.run(inputs, weight, scales, out)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def _arguments(
        self,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        scales: torch.Tensor,
        out: torch.Tensor,
    ) -> tuple:
        return (
            self.a_cute,
            from_dlpack(weight.unsqueeze(-1), assumed_align=16),
            self.peers[0],
            self.sa_cute,
            from_dlpack(scales.unsqueeze(-1), assumed_align=16),
            cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream),
            from_dlpack(out, assumed_align=16),
            self.peers,
            self.ready_cute,
            self.peer_ready,
            self.ack_cute,
            self.peer_ack,
            cutlass.Int32(self.rank),
            from_dlpack(inputs, assumed_align=16),
            self.input_peers,
            self.scale_peers,
            self.input_flags_cute,
            self.peer_input_flags,
        )

    def run(
        self,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        weight_scales: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Launch with this layer's tensors and return its owned destination."""
        if self.closed:
            raise RuntimeError("Fused TP4 projection workspace is closed")
        for tensor, shape, dtype in (
            (inputs, (128, 7168), torch.bfloat16),
            (weight, (12416, 7168), torch.float8_e4m3fn),
            (weight_scales, (97, 56), torch.float32),
            (out, (128, 49664), torch.bfloat16),
        ):
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != self.device
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
            ):
                raise ValueError("Incompatible fused TP4 projection tensor")
        self.compiled(*self._arguments(inputs, weight, weight_scales, out))
        return out

    def close(self) -> None:
        """Release peer mappings collectively after all graph consumers finish."""
        if self.closed:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self.compiled = None
        self.input_peers = self.scale_peers = self.peers = ()
        self.peer_ready = self.peer_ack = self.peer_input_flags = ()
        self.a_cute = self.sa_cute = self.ready_cute = self.ack_cute = (
            self.input_flags_cute
        ) = None
        self.peer_tensors = ()
        self.handles = []
        self.a = self.sa = self.receive = self.ready = self.ack = self.input_flags = (
            None
        )
        self.closed = True
