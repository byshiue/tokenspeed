// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Fused BF16-to-FP8 quantization and TP4 Ulysses redistribution.
#include "flashinfer/comm/ulysses_all_to_all.cuh"
#include "tvm_ffi_utils.h"
#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace fi = flashinfer::comm::ulysses;

__global__ void quant_a2a_kernel(const nv_bfloat16 *input, fi::RankData peers,
                                 fi::RankSignals signals, fi::Signal *self,
                                 int rank, int rows, int channels,
                                 int64_t scale_offset) {
  fi::multi_gpu_barrier<4, true>(signals, self, rank);
  const int lane = threadIdx.x % 32;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int warps = gridDim.x * blockDim.x / 32;
  const int groups = channels / 128;
  const int shard_groups = groups / 4;
  const int shard = channels / 4;
  for (int group = warp; group < rows * groups; group += warps) {
    const int row = group / groups;
    const int channel_group = group % groups;
    float values[4];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      values[i] = __bfloat162float(
          input[row * channels + channel_group * 128 + lane * 4 + i]);
      amax = fmaxf(amax, fabsf(values[i]));
    }
#pragma unroll
    for (int d = 16; d > 0; d /= 2)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, d));
    // The runtime's amax floor is represented in the BF16 input type.
    const float floor = __bfloat162float(__float2bfloat16(1e-10f));
    const float inverse = __fdiv_rn(448.0f, fmaxf(amax, floor));
    const float scale = __fdiv_rn(1.0f, inverse);
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t q =
          __nv_cvt_float_to_fp8(values[i] * inverse, __NV_SATFINITE, __NV_E4M3);
      packed |= q << (8 * i);
    }
    const int peer = channel_group / shard_groups;
    const int local_group = channel_group % shard_groups;
    auto output = reinterpret_cast<uint8_t *>(peers.ptrs[peer]);
    const int destination =
        (rank * rows + row) * shard + local_group * 128 + lane * 4;
    *reinterpret_cast<uint32_t *>(output + destination) = packed;
    if (lane == 0)
      reinterpret_cast<float *>(
          output + scale_offset)[local_group * (4 * rows) + rank * rows + row] =
          scale;
  }
  fi::multi_gpu_barrier<4, false, true>(signals, self, rank);
}

void quant_a2a(int64_t handle, TensorView input, int64_t scale_offset,
               int64_t blocks) {
  auto comm = reinterpret_cast<fi::UlyssesA2A *>(handle);
  ffi::CUDADeviceGuard guard(input.device().device_id);
  CHECK_INPUT(input);
  TVM_FFI_ICHECK_EQ(comm->world_size_, 4);
  TVM_FFI_ICHECK_EQ(input.ndim(), 2);
  TVM_FFI_ICHECK_EQ(input.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(input.size(1) % 512, 0);
  if (blocks == 0)
    blocks = std::min<int64_t>(fi::kMaxBlocks, 4 * input.size(0));
  TVM_FFI_ICHECK(blocks > 0 && blocks <= fi::kMaxBlocks);
  auto stream = get_stream(input.device());
  quant_a2a_kernel<<<blocks, 512, 0, stream>>>(
      reinterpret_cast<const nv_bfloat16 *>(input.data_ptr()), comm->out_ptrs_,
      comm->sg_, comm->self_sg_, comm->rank_, input.size(0), input.size(1),
      scale_offset);
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quant_a2a, quant_a2a);
