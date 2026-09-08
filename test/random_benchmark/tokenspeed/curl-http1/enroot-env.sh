#!/usr/bin/env bash

# Pyxis strips ENROOT_* variables from the task environment. Bash sources this
# file before /usr/bin/enroot reads its configuration, which preserves the
# bounded-concurrency override through the layer downloader. Each transfer is
# resumable below, so independent layers can make progress in parallel without
# turning a transport reset into a full-layer restart.
export ENROOT_MAX_CONNECTIONS=3

: "${K3_NSYS_CACHE_SETTINGS:?}"
cache_settings=${K3_NSYS_CACHE_SETTINGS}
if [[ -z "${TOKENSPEED_DOCKER_IMAGE_CACHE_PATH:-}" ]]; then
    export TOKENSPEED_DOCKER_IMAGE_CACHE_PATH=$(
        jq -er '.docker_image_cache_path' "${cache_settings}"
    )
fi

# Enroot streams each compressed layer directly into its transform pipeline,
# so a transport reset restarts a multi-GB layer from byte zero. Buffer GHCR
# blobs in shared storage with curl resume, verify the registry digest, and
# only then replay the complete bytes to Enroot's normal checksum/decompress
# pipeline. The raw blob also survives a cancelled SLURM step.
curl() {
    local url=${!#}
    if [[ "${url}" != https://ghcr.io/v2/lightseekorg/tokenspeed-runner/blobs/sha256:* ]]; then
        /usr/bin/curl "$@"
        return
    fi

    local digest=${url##*:}
    local image_digest=d6067daeeb1fafecc531d45e282797076e1cd2e2c16eaa90712634dd76a709ca
    local blob_dir=${TOKENSPEED_DOCKER_IMAGE_CACHE_PATH}/enroot-blobs/${image_digest}
    local blob=${blob_dir}/${digest}.blob
    local marker=${blob}.sha256-ok
    mkdir -p "${blob_dir}"

    until [[ -f "${marker}" ]]; do
        if ! /usr/bin/curl --continue-at - --output "${blob}" "$@"; then
            sleep 2
            continue
        fi
        if [[ "$(sha256sum "${blob}" | cut -d ' ' -f 1)" == "${digest}" ]]; then
            touch "${marker}"
        else
            mv "${blob}" "${blob}.bad.$(date +%s)"
        fi
    done
    /usr/bin/cat "${blob}"
}
export -f curl
