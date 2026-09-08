#!/usr/bin/env bash

set -euo pipefail

: "${K3_NSYS_WORKTREE:?}"
: "${K3_NSYS_MODEL_CONFIG:?}"
: "${K3_NSYS_OUTPUT_DIR:?}"
: "${K3_NSYS_SOURCE_SHA:?}"
: "${K3_PACKAGE_CACHE_PATH:?}"
: "${NSYS_BIN:?}"
: "${K3_NSYS_ATTN_TP_SIZE:?}"
: "${K3_NSYS_MOE_TP_SIZE:?}"
: "${K3_NSYS_MAX_CUDAGRAPH_CAPTURE_SIZE:?}"
: "${K3_NSYS_CUDAGRAPH_CAPTURE_SIZES:?}"
: "${K3_NSYS_DIST_INIT_ADDR:?}"
: "${K3_NSYS_RUN_LABEL:?}"

node_name=${SLURMD_NODENAME:-$(hostname)}
report_prefix=${K3_NSYS_OUTPUT_DIR}/kimi_k3_agentic_incremental_${K3_NSYS_RUN_LABEL}_${node_name}
source_root=${K3_NSYS_SOURCE_ROOT:-${K3_NSYS_WORKTREE}}
venv_dir=${K3_NSYS_VENV_DIR:-${K3_NSYS_WORKTREE}/.venv}
install_marker=${venv_dir}/tokenspeed-source-sha
install_fingerprint=${K3_NSYS_SOURCE_SHA}-sm103-clean-objs-v1
install_lock=${venv_dir}.install.lock

# Both Slurm tasks share this worktree.  Serialize virtualenv and native-wheel
# creation so a second node never reads a partially written wheel.
exec 9>"${install_lock}"
flock 9

if [[ ! -x "${venv_dir}/bin/python3" ]] || \
   [[ ! -f "${install_marker}" ]] || \
   [[ "$(<"${install_marker}")" != "${install_fingerprint}" ]]; then
    python3 -m venv --system-site-packages "${venv_dir}"
    source "${venv_dir}/bin/activate"
    export CI_RUNNER_LABEL=slurm-gb300-4gpu
    export CI_CACHE_ROOT=${K3_PACKAGE_CACHE_PATH}/tokenspeed-ci
    export PIP_CACHE_DIR=${K3_PACKAGE_CACHE_PATH}/pip
    export UV_CACHE_DIR=${K3_PACKAGE_CACHE_PATH}/uv
    mkdir -p "${CI_CACHE_ROOT}" "${PIP_CACHE_DIR}" "${UV_CACHE_DIR}"
    export CUDA_VERSION=13.0.1
    export SM=sm103
    # K3_NSYS_SOURCE_ROOT may select a detached comparison worktree. Build
    # both Python and native packages from that source so the scheduler and
    # runtime always come from the same commit.
    export WORKSPACE=${source_root}

    # The shared source checkout can contain ignored CUDA objects built for a
    # different architecture.  setup.py treats newer .so files as reusable, so
    # move them aside before the SM103 wheel build rather than packaging SM100.
    kernel_objs=${source_root}/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/objs
    if [[ -d "${kernel_objs}" ]]; then
        mv "${kernel_objs}" "${kernel_objs}.pre-sm103-${SLURM_JOB_ID}"
    fi
    bash "${source_root}/test/ci_system/install_deps.sh"

    printf '%s\n' "${install_fingerprint}" > "${install_marker}"
else
    source "${venv_dir}/bin/activate"
fi
flock -u 9
exec 9>&-

# The current package cache supplies a newer distribution metadata layout but
# not its ``tokenspeed_triton`` module.  Reuse the pinned runtime module from
# the established TP4 profiling environment when it is absent.
baseline_triton_site=${K3_PACKAGE_CACHE_PATH}/tokenspeed-triton-pinned-3.8.10.post20260721
if [[ ! -d "${venv_dir}/lib/python3.12/site-packages/tokenspeed_triton" ]]; then
    cp -a "${baseline_triton_site}/tokenspeed_triton" \
        "${venv_dir}/lib/python3.12/site-packages/tokenspeed_triton"
fi
if [[ ! -d "${venv_dir}/lib/python3.12/site-packages/tokenspeed_triton-3.8.10.post20260906.dist-info" ]]; then
    cp -a "${baseline_triton_site}/tokenspeed_triton-3.8.10.post20260906.dist-info" \
        "${venv_dir}/lib/python3.12/site-packages/tokenspeed_triton-3.8.10.post20260906.dist-info"
fi

# Native packages must come from the architecture-specific wheels built above.
# Source-tree PYTHONPATH entries can shadow them with stale objects from a
# different GPU architecture.  TokenSpeed itself is installed editable.
unset PYTHONPATH
export PYTHONPATH=${source_root}/python
export TOKENSPEED_LOG_MM_TIMING=1

if [[ "${K3_TIMING_DISABLE_PROFILING:-0}" == "1" ]]; then
    unset TOKENSPEED_NVTX
    profiler_command=()
    enable_nvtx_flag=()
else
    export TOKENSPEED_NVTX=1
    profiler_command=(
        "${NSYS_BIN}" profile
        --capture-range=cudaProfilerApi
        --capture-range-end=stop
        --cuda-graph-trace=node
        --trace=cuda,nvtx,cublas,cudnn
        --sample=none
        --cpuctxsw=none
        --trace-fork-before-exec=true
        --force-overwrite=true
        --output "${report_prefix}"
    )
    enable_nvtx_flag=(--enable-nvtx)
fi

# The real Kimi checkpoint supplies its tokenizer through HuggingFace remote
# code.  Sharing the default module cache across the two nodes allows ranks to
# observe a partially materialized module during concurrent engine startup.
# Preload it once per node into a node-private cache before the launcher forks
# local ranks.
export HF_MODULES_CACHE="${K3_NSYS_OUTPUT_DIR}/hf_modules_${node_name}"
mkdir -p "${HF_MODULES_CACHE}"
python3 -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(__import__("os").environ["K3_NSYS_MODEL_CONFIG"], trust_remote_code=True)'

exec "${profiler_command[@]}" python3 -m tokenspeed.cli serve \
        --model "${K3_NSYS_MODEL_CONFIG}" \
        --served-model-name kimi-k3-nvfp4-20l \
        --skip-tokenizer-init \
        --language-model-only \
        --trust-remote-code \
        --dtype bfloat16 \
        --quantization nvfp4 \
        --kv-cache-dtype fp8 \
        --attn-tp-size "${K3_NSYS_ATTN_TP_SIZE}" \
        --moe-tp-size "${K3_NSYS_MOE_TP_SIZE}" \
        --dist-init-addr "${K3_NSYS_DIST_INIT_ADDR}" \
        --max-model-len 65536 \
        --max-num-seqs 4 \
        --chunked-prefill-size 8192 \
        --max-prefill-tokens 8192 \
        --attention-backend tokenspeed_mla \
        --kda-backend cutedsl_kda \
        --moe-backend flashinfer_trtllm \
        --sampling-backend greedy \
        --gpu-memory-utilization 0.80 \
        --disable-autotune \
        --weight-loader-prefetch-num-threads 1 \
        --max-cudagraph-capture-size "${K3_NSYS_MAX_CUDAGRAPH_CAPTURE_SIZE}" \
        --cudagraph-capture-sizes ${K3_NSYS_CUDAGRAPH_CAPTURE_SIZES} \
        --disable-kvstore \
        --enable-prefix-caching \
        --enable-cache-report \
        --enable-log-request-stats \
        "${enable_nvtx_flag[@]}" \
        --host 0.0.0.0 \
        --port 21000 \
        --engine-startup-timeout 2400 \
        --gateway-startup-timeout 600
