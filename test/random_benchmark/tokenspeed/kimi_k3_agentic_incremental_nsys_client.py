#!/usr/bin/env python3
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

"""Capture an agentic-style incremental prefill after warming its prefix."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import threading
import time
import urllib.request


def post_json(url: str, payload: dict, *, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def generate(
    base_url: str, model: str, input_ids: list[int], output_length: int
) -> list[int]:
    started = time.perf_counter()
    response = post_json(
        f"{base_url}/generate",
        {
            "model": model,
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": output_length,
                "temperature": 0,
                "ignore_eos": True,
            },
            "stream": False,
        },
        timeout=3600,
    )
    if isinstance(response, list):
        if len(response) != 1:
            raise RuntimeError(f"expected one generation result, got {len(response)}")
        response = response[0]
    output_ids = response["output_ids"]
    output_digest = hashlib.sha256(
        json.dumps(output_ids, separators=(",", ":")).encode()
    ).hexdigest()
    print(
        f"generate prompt={len(input_ids)} output={len(output_ids)} "
        f"elapsed={time.perf_counter() - started:.3f}s sha256={output_digest}",
        flush=True,
    )
    return output_ids


def generate_batch(
    base_url: str,
    control_url: str,
    model: str,
    conversations: list[list[int]],
    output_length: int,
) -> list[list[int]]:
    """Submit one request per conversation at the same scheduler boundary."""
    if len(conversations) == 1:
        return [generate(base_url, model, conversations[0], output_length)]

    pause_response = post_json(
        f"{control_url}/pause_generation",
        {},
        timeout=60,
    )
    print(f"pause response: {pause_response}", flush=True)
    start_barrier = threading.Barrier(len(conversations))

    def generate_after_barrier(conversation: list[int]) -> list[int]:
        start_barrier.wait()
        return generate(base_url, model, conversation, output_length)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(conversations)
    ) as executor:
        futures = [
            executor.submit(generate_after_barrier, conversation)
            for conversation in conversations
        ]
        # Requests wait at the frontend admission gate while the scheduler is
        # paused. Resuming releases the whole group into one receive drain.
        time.sleep(1)
        continue_response = post_json(
            f"{control_url}/continue_generation",
            {},
            timeout=60,
        )
        print(f"continue response: {continue_response}", flush=True)
        return [future.result() for future in futures]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--control-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--first-turn-length", required=True, type=int)
    parser.add_argument("--subsequent-turn-length", required=True, type=int)
    parser.add_argument("--output-length", required=True, type=int)
    parser.add_argument("--profile-output-length", required=True, type=int)
    parser.add_argument("--profile-iterations", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--shape-warmup-seed", required=True, type=int)
    parser.add_argument("--profile-concurrency", required=True, type=int)
    args = parser.parse_args()
    if args.profile_concurrency < 1:
        parser.error("--profile-concurrency must be positive")
    if args.profile_iterations < 1:
        parser.error("--profile-iterations must be positive")

    request_rngs = [
        random.Random(args.seed + request_id)
        for request_id in range(args.profile_concurrency)
    ]
    conversations = [
        [rng.randrange(1000, 160000) for _ in range(args.first_turn_length)]
        for rng in request_rngs
    ]
    first_outputs = generate_batch(
        args.base_url,
        args.control_url,
        args.model,
        conversations,
        args.output_length,
    )
    for conversation, first_output, rng in zip(
        conversations, first_outputs, request_rngs, strict=True
    ):
        conversation.extend(first_output)

    profile_conversation_batches = []
    for _ in range(args.profile_iterations):
        profile_conversations = []
        for conversation, rng in zip(conversations, request_rngs, strict=True):
            profile_conversation = conversation.copy()
            profile_conversation.extend(
                rng.randrange(1000, 160000) for _ in range(args.subsequent_turn_length)
            )
            profile_conversations.append(profile_conversation)
        profile_conversation_batches.append(profile_conversations)

    warmup_rngs = [
        random.Random(args.shape_warmup_seed + request_id)
        for request_id in range(args.profile_concurrency)
    ]
    warmup_conversations = [
        [rng.randrange(1000, 160000) for _ in range(args.first_turn_length)]
        for rng in warmup_rngs
    ]
    warmup_outputs = generate_batch(
        args.base_url,
        args.control_url,
        args.model,
        warmup_conversations,
        args.output_length,
    )
    for warmup_conversation, warmup_output, rng in zip(
        warmup_conversations, warmup_outputs, warmup_rngs, strict=True
    ):
        warmup_conversation.extend(warmup_output)
        warmup_conversation.extend(
            rng.randrange(1000, 160000) for _ in range(args.subsequent_turn_length)
        )
    generate_batch(
        args.base_url,
        args.control_url,
        args.model,
        warmup_conversations,
        args.profile_output_length,
    )

    profile_response = post_json(
        f"{args.base_url}/start_profile",
        {
            "activities": ["CUDA_PROFILER"],
            "with_stack": False,
            "record_shapes": False,
        },
        timeout=60,
    )
    print(f"profile start response: {profile_response}", flush=True)
    for iteration, profile_conversations in enumerate(
        profile_conversation_batches, start=1
    ):
        print(f"profile iteration={iteration}/{args.profile_iterations}", flush=True)
        generate_batch(
            args.base_url,
            args.control_url,
            args.model,
            profile_conversations,
            args.profile_output_length,
        )
    stop_response = post_json(
        f"{args.base_url}/stop_profile",
        {},
        timeout=60,
    )
    print(f"profile stop response: {stop_response}", flush=True)


if __name__ == "__main__":
    main()
