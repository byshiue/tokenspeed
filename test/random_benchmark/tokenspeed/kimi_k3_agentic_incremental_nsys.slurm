#!/usr/bin/env bash

# Run this inside an existing persistent two-node allocation:
#   salloc --no-shell --nodes=2 --gpus-per-node=4 ...
#   K3_NSYS_ALLOCATION_ID=<job-id> K3_NSYS_RUN_LABEL=<label> \
#     K3_NSYS_MODEL=<checkpoint> K3_NSYS_CACHE_SETTINGS=<cache-setting.json> \
#     K3_NSYS_ROOT=<nsight-systems-root> bash "$0"

set -euo pipefail

worktree=${K3_NSYS_WORKTREE:-$(git rev-parse --show-toplevel)}
allocation_id=${K3_NSYS_ALLOCATION_ID:-${SLURM_JOB_ID:?}}
run_label=${K3_NSYS_RUN_LABEL:?}
profile_concurrency=${K3_NSYS_PROFILE_CONCURRENCY:-1}
source_root=${K3_NSYS_SOURCE_ROOT:-${worktree}}
source_model=${K3_NSYS_MODEL:?}
output_root=${K3_NSYS_OUTPUT_ROOT:-${worktree}/outputs/kimi_k3_agentic_incremental_nsys}
output_dir=${output_root}/${allocation_id}-${run_label}
model_config=${output_dir}/model_config
cache_settings=${K3_NSYS_CACHE_SETTINGS:?}
export K3_NSYS_CACHE_SETTINGS=${cache_settings}
docker_image_cache_path=$(jq -er '.docker_image_cache_path' "${cache_settings}")
package_cache_path=$(jq -er '.package_cache_path' "${cache_settings}")
export TOKENSPEED_DOCKER_IMAGE_CACHE_PATH=${docker_image_cache_path}
remote_image=ghcr.io/lightseekorg/tokenspeed-runner:cu130-torch-2.13.0-flashinfer-0.6.18@sha256:d6067daeeb1fafecc531d45e282797076e1cd2e2c16eaa90712634dd76a709ca
cached_image=${docker_image_cache_path}/tokenspeed-runner-cu130-torch-2.13.0-flashinfer-0.6.18-d6067dae.sqsh
image=${remote_image}
saved_image_tmp=
container_save_args=()
mkdir -p "${docker_image_cache_path}" "${package_cache_path}"
if [[ -s "${cached_image}" ]]; then
    image=${cached_image}
else
    saved_image_tmp=${cached_image}.partial-${allocation_id}
    container_save_args=(--container-save="${saved_image_tmp}")
fi
nsys_root=${K3_NSYS_ROOT:?}

mkdir -p "${model_config}"
# GHCR occasionally resets this large image's HTTP/2 streams on fresh nodes.
# Pyxis/Enroot inherits PATH, so use a tiny curl wrapper that forces HTTP/1.1.
export PATH=${worktree}/test/random_benchmark/tokenspeed/curl-http1:${PATH}
export CURL_HOME=${worktree}/test/random_benchmark/tokenspeed/curl-http1
# Pyxis filters ENROOT_* task variables. BASH_ENV restores the setting inside
# /usr/bin/enroot before it reads the system config.
export BASH_ENV=${worktree}/test/random_benchmark/tokenspeed/curl-http1/enroot-env.sh
jq \
    '.text_config.num_hidden_layers = 20
     | .text_config.linear_attn_config.kda_layers |= map(select(. <= 20))
     | .text_config.linear_attn_config.full_attn_layers |= map(select(. <= 20))' \
    "${source_model}/config.json" > "${model_config}/config.json"
cp "${source_model}/generation_config.json" "${model_config}/generation_config.json"
cp "${source_model}/hf_quant_config.json" "${model_config}/hf_quant_config.json"
cp "${source_model}/tokenizer_config.json" "${model_config}/tokenizer_config.json"
cp "${source_model}/tiktoken.model" "${model_config}/tiktoken.model"
cp "${source_model}/tokenization_kimi.py" "${model_config}/tokenization_kimi.py"
cp "${source_model}/encoding_k3.py" "${model_config}/encoding_k3.py"
jq \
    '.weight_map |= with_entries(
        select(
            (.key | test("\\.layers\\.(?:[0-9]|1[0-9])\\."))
            or (.key | test("\\.layers\\.") | not)
        )
    )' \
    "${source_model}/model.safetensors.index.json" \
    > "${model_config}/model.safetensors.index.json"
for weight_shard in "${source_model}"/model-*.safetensors; do
    ln -sfn "${weight_shard}" "${model_config}/$(basename "${weight_shard}")"
done

export K3_NSYS_WORKTREE=${worktree}
export K3_NSYS_VENV_DIR=${worktree}/.venv
export K3_NSYS_SOURCE_ROOT=${source_root}
export K3_NSYS_RUN_LABEL=${run_label}
export K3_NSYS_MODEL_CONFIG=${model_config}
export K3_NSYS_OUTPUT_DIR=${output_dir}
export K3_NSYS_SOURCE_SHA=$(git -C "${source_root}" rev-parse HEAD)
export K3_PACKAGE_CACHE_PATH=${package_cache_path}
export NSYS_BIN=/opt/nsys/target-linux-sbsa-armv8/nsys
export K3_NSYS_ATTN_TP_SIZE=8
export K3_NSYS_MOE_TP_SIZE=8
export K3_NSYS_MAX_CUDAGRAPH_CAPTURE_SIZE=4
export K3_NSYS_CUDAGRAPH_CAPTURE_SIZES="1 2 4"
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
mkdir -p "${output_dir}"

mapfile -t nodes < <(scontrol show hostnames "$(squeue -j "${allocation_id}" -h -o '%N')")
if [[ ${#nodes[@]} -ne 2 ]]; then
    echo "TP8 validation requires exactly two allocated nodes, got ${#nodes[@]}" >&2
    exit 1
fi
head_node=${nodes[0]}
export K3_NSYS_DIST_INIT_ADDR=${head_node}:8000

mounts=${worktree}:${worktree},${source_model}:${source_model},${nsys_root}:/opt/nsys,${package_cache_path}:${package_cache_path}
if [[ "${source_root}" != "${worktree}" ]]; then
    mounts=${mounts},${source_root}:${source_root}
fi
server_step_pid=

cleanup() {
    if [[ -n "${server_step_pid}" ]] && kill -0 "${server_step_pid}" 2>/dev/null; then
        kill -TERM "${server_step_pid}" 2>/dev/null || true
        wait "${server_step_pid}" || true
    fi
}
trap cleanup EXIT

srun \
    --jobid="${allocation_id}" \
    --overlap \
    --nodes=2 \
    --ntasks=2 \
    --ntasks-per-node=1 \
    --gpus-per-task=4 \
    --container-image="${image}" \
    "${container_save_args[@]}" \
    --container-workdir="${worktree}" \
    --container-mounts="${mounts}" \
    --container-writable \
    --container-remap-root \
    --no-container-entrypoint \
    bash "${worktree}/test/random_benchmark/tokenspeed/kimi_k3_agentic_incremental_nsys_server.sh" \
    > "${output_dir}/server.log" 2>&1 &
server_step_pid=$!

ready_url=http://${head_node}:21000/readiness
deadline=$((SECONDS + 6000))
until curl --fail --silent --output /dev/null "${ready_url}"; do
    if ! kill -0 "${server_step_pid}" 2>/dev/null; then
        tail -200 "${output_dir}/server.log"
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "server readiness timed out"
        tail -200 "${output_dir}/server.log"
        exit 1
    fi
    sleep 10
done

python3 "${worktree}/test/random_benchmark/tokenspeed/kimi_k3_agentic_incremental_nsys_client.py" \
    --base-url http://${head_node}:21000 \
    --control-url http://${head_node}:21001 \
    --model kimi-k3-nvfp4-20l \
    --first-turn-length 50000 \
    --subsequent-turn-length 800 \
    --output-length 500 \
    --profile-output-length 1 \
    --seed 1 \
    --shape-warmup-seed 2 \
    --profile-concurrency "${profile_concurrency}" \
    > "${output_dir}/client.log" 2>&1

sleep 20
cleanup
server_step_pid=

if [[ -n "${saved_image_tmp}" ]]; then
    if [[ ! -s "${saved_image_tmp}" ]]; then
        echo "container save did not create ${saved_image_tmp}" >&2
        exit 1
    fi
    mv "${saved_image_tmp}" "${cached_image}"
    image=${cached_image}
fi

srun \
    --jobid="${allocation_id}" \
    --overlap \
    --nodes=1 \
    --ntasks=1 \
    --container-image="${image}" \
    --container-workdir="${worktree}" \
    --container-mounts="${mounts}" \
    --container-writable \
    --container-remap-root \
    --no-container-entrypoint \
    bash -lc \
    'for report in "${K3_NSYS_OUTPUT_DIR}"/*.nsys-rep; do
         /opt/nsys/target-linux-sbsa-armv8/nsys stats --force-export=true \
             --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum \
             "${report}" > "${report%.nsys-rep}.stats.txt"
     done'

find "${output_dir}" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
