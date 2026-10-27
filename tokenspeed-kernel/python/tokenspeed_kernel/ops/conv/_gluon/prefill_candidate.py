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

"""One prefill kernel with compile-time rolling and parallel token tactics."""

from tokenspeed_kernel._triton import gl, gluon, triton


@gluon.jit
def _gluon_load_input(X, start, rows, channels, sx_token, sx_channel, input_mask):
    # Keep strides dynamic, but expose unit strides inside uniform branches.
    # Otherwise the compiler can place adjacent lanes on distant token rows.
    if sx_channel == 1:
        values = gl.load(
            X + (start + rows) * sx_token + channels,
            input_mask,
            other=0,
        )
    elif sx_token == 1:
        values = gl.load(
            X + start + rows + channels * sx_channel,
            input_mask,
            other=0,
        )
    else:
        values = gl.load(
            X + (start + rows) * sx_token + channels * sx_channel,
            input_mask,
            other=0,
        )
    return values


@gluon.jit
def _gluon_load_values(
    X,
    State,
    start,
    rows,
    channels,
    slot,
    length,
    history,
    sx_token,
    sx_channel,
    ss_slot,
    ss_channel,
    ss_token,
    valid_rows,
    valid_channels,
    WIDTH: gl.constexpr,
    FIRST: gl.constexpr,
):
    input_mask = valid_rows & (rows >= 0) & (rows < length) & valid_channels
    values = _gluon_load_input(
        X,
        start,
        rows,
        channels,
        sx_token,
        sx_channel,
        input_mask,
    )
    if FIRST:
        old = gl.load(
            State
            + slot * ss_slot
            + channels * ss_channel
            + (rows + WIDTH - 1) * ss_token,
            valid_rows & (rows < 0) & (rows >= 1 - WIDTH) & history & valid_channels,
            other=0,
        )
        return gl.where(rows < 0, old, values)
    return values


@gluon.jit
def _gluon_load_window(
    X,
    State,
    start,
    rows,
    channels,
    slot,
    length,
    history,
    sx_token,
    sx_channel,
    ss_slot,
    ss_channel,
    ss_token,
    valid_rows,
    valid_channels,
    WIDTH: gl.constexpr,
    FIRST: gl.constexpr,
):
    return _gluon_load_values(
        X,
        State,
        start,
        rows[:, None],
        channels[None, :],
        slot,
        length,
        history,
        sx_token,
        sx_channel,
        ss_slot,
        ss_channel,
        ss_token,
        valid_rows[:, None],
        valid_channels[None, :],
        WIDTH,
        FIRST,
    )


@gluon.jit
def _gluon_accumulate(
    acc,
    values,
    weights,
    HAS_BIAS: gl.constexpr,
    SILU: gl.constexpr,
    WIDTH: gl.constexpr,
    TAP: gl.constexpr,
):
    # Preserve the frozen baseline's product type and ordered accumulation.
    # BF16 and FP16 products round before the FP32 accumulation.
    if (
        HAS_BIAS
        and SILU
        and WIDTH == 3
        and TAP == 0
        and values.dtype == gl.float32
        and weights.dtype == gl.float32
    ):
        # The SM103 baseline emits this multiply separately from the bias add.
        return acc + gl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [values, weights],
            dtype=gl.float32,
            is_pure=True,
            pack=1,
        )
    elif values.dtype == gl.float32 and weights.dtype == gl.float32:
        return gl.fma(values, weights, acc)
    else:
        return acc + values * weights


@gluon.jit
def _gluon_store_output(
    Out, rows, channels, acc, mask, so_token, so_channel, SILU: gl.constexpr
):
    if SILU:
        acc = acc / (1 + gl.exp(-acc))
    if so_channel == 1:
        gl.store(Out + rows * so_token + channels, acc, mask)
    elif so_token == 1:
        gl.store(Out + rows + channels * so_channel, acc, mask)
    else:
        gl.store(Out + rows * so_token + channels * so_channel, acc, mask)


@gluon.jit
def _gluon_update_state(
    X,
    State,
    start,
    length,
    slot,
    history,
    channels,
    valid_channels,
    sx_token,
    sx_channel,
    ss_slot: gl.constexpr,
    ss_channel: gl.constexpr,
    ss_token: gl.constexpr,
    WIDTH: gl.constexpr,
    TOKEN_LAYOUT: gl.constexpr,
):
    # Only this tile reads old state. Other tiles read immutable input.
    # A short request takes its final state from old history + new input.
    state_positions = gl.arange(
        0, triton.next_power_of_2(WIDTH - 1), layout=TOKEN_LAYOUT
    )
    tail = _gluon_load_window(
        X,
        State,
        start,
        length - (WIDTH - 1) + state_positions,
        channels,
        slot,
        length,
        history,
        sx_token,
        sx_channel,
        ss_slot,
        ss_channel,
        ss_token,
        state_positions < WIDTH - 1,
        valid_channels,
        WIDTH,
        True,
    )
    # Finish every old-state load before any thread overwrites that state.
    gl.barrier()
    gl.store(
        State
        + slot * ss_slot
        + channels[None, :] * ss_channel
        + state_positions[:, None] * ss_token,
        tail,
        (state_positions[:, None] < WIDTH - 1) & valid_channels[None, :],
    )


@gluon.jit
def _gluon_body(
    X,
    Weight,
    Bias,
    State,
    Out,
    channel_block,
    start,
    token,
    length,
    slot,
    history,
    sx_token,
    sx_channel,
    so_token,
    so_channel,
    sw_channel: gl.constexpr,
    sw_tap: gl.constexpr,
    ss_slot: gl.constexpr,
    ss_channel: gl.constexpr,
    ss_token: gl.constexpr,
    CHANNELS: gl.constexpr,
    WIDTH: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    SILU: gl.constexpr,
    WINDOW: gl.constexpr,
    TOKEN_PARALLEL: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    WINDOW_ROWS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    FIRST: gl.constexpr,
):
    VECTOR_N: gl.constexpr = max(1, min(8, BLOCK_N // (32 * NUM_WARPS)))
    WARPS_N: gl.constexpr = min(BLOCK_N // (32 * VECTOR_N), NUM_WARPS)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, VECTOR_N], [1, 32], [NUM_WARPS // WARPS_N, WARPS_N], [1, 0]
    )
    channel_layout: gl.constexpr = gl.SliceLayout(0, layout)
    token_layout: gl.constexpr = gl.SliceLayout(1, layout)
    channels = (
        channel_block * BLOCK_N + gl.arange(0, BLOCK_N, layout=channel_layout)
    ).to(gl.int64)
    valid_channels = channels < CHANNELS
    if TOKEN_PARALLEL:
        positions = gl.arange(0, BLOCK_M, layout=token_layout)
        valid_outputs = token + positions < length

        if WINDOW:
            # Padding to a power of two is an IR shape, not an extra global read.
            window_positions = gl.arange(0, WINDOW_ROWS, layout=token_layout)
            window = _gluon_load_window(
                X,
                State,
                start,
                token + window_positions - (WIDTH - 1),
                channels,
                slot,
                length,
                history,
                sx_token,
                sx_channel,
                ss_slot,
                ss_channel,
                ss_token,
                window_positions < BLOCK_M + WIDTH - 1,
                valid_channels,
                WIDTH,
                FIRST,
            )
        acc = gl.full((BLOCK_M, BLOCK_N), 0, gl.float32, layout)
        if HAS_BIAS:
            bias = gl.load(Bias + channels, valid_channels, other=0).to(gl.float32)
            acc, _ = gl.broadcast(bias[None, :], acc)
        for tap in gl.static_range(WIDTH):
            if WINDOW:
                indices = (positions + tap)[:, None] + gl.full(
                    (BLOCK_M, BLOCK_N), 0, gl.int32, layout
                )
                values = gl.gather(window, indices, axis=0)
            else:
                values = _gluon_load_window(
                    X,
                    State,
                    start,
                    token + positions + tap - (WIDTH - 1),
                    channels,
                    slot,
                    length,
                    history,
                    sx_token,
                    sx_channel,
                    ss_slot,
                    ss_channel,
                    ss_token,
                    valid_outputs,
                    valid_channels,
                    WIDTH,
                    FIRST,
                )
            weights = gl.load(
                Weight + channels * sw_channel + tap * sw_tap, valid_channels, other=0
            )
            acc = _gluon_accumulate(
                acc,
                values,
                weights[None, :],
                HAS_BIAS,
                SILU,
                WIDTH,
                tap,
            )

        _gluon_store_output(
            Out,
            start + token + positions[:, None],
            channels[None, :],
            acc,
            valid_outputs[:, None] & valid_channels[None, :],
            so_token,
            so_channel,
            SILU,
        )
    else:
        # Keep the rolling window in registers. Load one new row per token.
        # The loop visits only valid tokens, which matters for tiny requests.
        columns = ()
        for tap in gl.static_range(WIDTH - 1):
            if FIRST:
                previous = _gluon_load_values(
                    X,
                    State,
                    start,
                    tap - (WIDTH - 1),
                    channels,
                    slot,
                    length,
                    history,
                    sx_token,
                    sx_channel,
                    ss_slot,
                    ss_channel,
                    ss_token,
                    True,
                    valid_channels,
                    WIDTH,
                    True,
                )
            else:
                # A nonfirst tile starts at BLOCK_M >= WIDTH - 1.
                previous = _gluon_load_input(
                    X,
                    start,
                    token + tap - (WIDTH - 1),
                    channels,
                    sx_token,
                    sx_channel,
                    valid_channels,
                )
            columns += (previous,)
        if FIRST:
            # History is already in registers. Finish the state update before
            # the rolling loop so state addresses do not stay live across it.
            _gluon_update_state(
                X,
                State,
                start,
                length,
                slot,
                history,
                channels,
                valid_channels,
                sx_token,
                sx_channel,
                ss_slot,
                ss_channel,
                ss_token,
                WIDTH,
                token_layout,
            )
        weights = ()
        for tap in gl.static_range(WIDTH):
            weights += (
                gl.load(
                    Weight + channels * sw_channel + tap * sw_tap,
                    valid_channels,
                    other=0,
                ),
            )
        initial = gl.full((BLOCK_N,), 0, gl.float32, channel_layout)
        if HAS_BIAS:
            initial = gl.load(Bias + channels, valid_channels, other=0).to(gl.float32)
        # Only the bounded loop counter is int32. All tensor addresses stay int64.
        segment_length = gl.minimum(BLOCK_M, length - token).to(gl.int32)
        # The tile size is fixed. Unroll the rolling window so the compiler
        # can release per-token temporaries without a loop backedge. The guard
        # keeps short requests and partial tiles from reading invalid rows.
        for position in gl.static_range(BLOCK_M):
            if position < segment_length:
                # The loop bound already proves that this input row is valid.
                current = _gluon_load_input(
                    X,
                    start,
                    token + position,
                    channels,
                    sx_token,
                    sx_channel,
                    valid_channels,
                )
                window = columns + (current,)
                acc = initial
                for tap in gl.static_range(WIDTH):
                    acc = _gluon_accumulate(
                        acc,
                        window[tap],
                        weights[tap],
                        HAS_BIAS,
                        SILU,
                        WIDTH,
                        tap,
                    )
                _gluon_store_output(
                    Out,
                    start + token + position,
                    channels,
                    acc,
                    valid_channels,
                    so_token,
                    so_channel,
                    SILU,
                )
                columns = window[1:]

    if FIRST and TOKEN_PARALLEL:
        _gluon_update_state(
            X,
            State,
            start,
            length,
            slot,
            history,
            channels,
            valid_channels,
            sx_token,
            sx_channel,
            ss_slot,
            ss_channel,
            ss_token,
            WIDTH,
            token_layout,
        )


@gluon.jit(
    do_not_specialize=["sx_token", "sx_channel", "so_token", "so_channel"],
    do_not_specialize_on_alignment=["Batches", "Offsets"],
)
def gluon_causal_conv1d_prefill_kernel(
    X,
    Weight,
    Bias,
    State,
    Slots,
    History,
    Boundaries,
    Batches,
    Offsets,
    Out,
    sx_token,
    sx_channel,
    # Weight and state strides belong to fixed model layouts. Both input and
    # output strides can follow token counts for different tensor layouts.
    sw_channel: gl.constexpr,
    sw_tap: gl.constexpr,
    ss_slot: gl.constexpr,
    ss_channel: gl.constexpr,
    ss_token: gl.constexpr,
    so_token,
    so_channel,
    CHANNELS: gl.constexpr,
    WIDTH: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    SILU: gl.constexpr,
    WINDOW: gl.constexpr,
    TOKEN_PARALLEL: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    WINDOW_ROWS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    ENABLE_PDL: gl.constexpr,
):
    if ENABLE_PDL:
        gl.inline_asm_elementwise(
            "griddepcontrol.wait; mov.u32 $0, 0;",
            "=r,~{memory}",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        gl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; mov.u32 $0, 0;",
            "=r,~{memory}",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    if TOKEN_PARALLEL:
        channel_block = gl.program_id(0)
        token_block = gl.program_id(1)
    else:
        # Preserve the original rolling kernel's CTA traversal order.
        channel_block = gl.program_id(1)
        token_block = gl.program_id(0)
    request = gl.load(Batches + token_block)
    if request == -1:
        return
    start = gl.load(Boundaries + request).to(gl.int64)
    length = gl.load(Boundaries + request + 1).to(gl.int64) - start
    token = gl.load(Offsets + token_block).to(gl.int64) * BLOCK_M
    if token >= length:
        return
    slot = gl.load(Slots + request).to(gl.int64)
    if slot == -1:
        return
    if token == 0:
        history = gl.load(History + request)
        # Specialize the compact first tile too. The rolling loop must not
        # repeat runtime stride branches for every token.
        if (
            not TOKEN_PARALLEL
            and sx_token == CHANNELS
            and sx_channel == 1
            and so_token == CHANNELS
            and so_channel == 1
        ):
            _gluon_body(
                X,
                Weight,
                Bias,
                State,
                Out,
                channel_block,
                start,
                token,
                length,
                slot,
                history,
                CHANNELS,
                1,
                CHANNELS,
                1,
                sw_channel,
                sw_tap,
                ss_slot,
                ss_channel,
                ss_token,
                CHANNELS,
                WIDTH,
                HAS_BIAS,
                SILU,
                WINDOW,
                TOKEN_PARALLEL,
                BLOCK_M,
                BLOCK_N,
                WINDOW_ROWS,
                NUM_WARPS,
                True,
            )
        else:
            _gluon_body(
                X,
                Weight,
                Bias,
                State,
                Out,
                channel_block,
                start,
                token,
                length,
                slot,
                history,
                sx_token,
                sx_channel,
                so_token,
                so_channel,
                sw_channel,
                sw_tap,
                ss_slot,
                ss_channel,
                ss_token,
                CHANNELS,
                WIDTH,
                HAS_BIAS,
                SILU,
                WINDOW,
                TOKEN_PARALLEL,
                BLOCK_M,
                BLOCK_N,
                WINDOW_ROWS,
                NUM_WARPS,
                True,
            )

    elif (
        sx_token == CHANNELS
        and sx_channel == 1
        and so_token == CHANNELS
        and so_channel == 1
    ):
        _gluon_body(
            X,
            Weight,
            Bias,
            State,
            Out,
            channel_block,
            start,
            token,
            length,
            slot,
            False,
            CHANNELS,
            1,
            CHANNELS,
            1,
            sw_channel,
            sw_tap,
            ss_slot,
            ss_channel,
            ss_token,
            CHANNELS,
            WIDTH,
            HAS_BIAS,
            SILU,
            WINDOW,
            TOKEN_PARALLEL,
            BLOCK_M,
            BLOCK_N,
            WINDOW_ROWS,
            NUM_WARPS,
            False,
        )
    else:
        _gluon_body(
            X,
            Weight,
            Bias,
            State,
            Out,
            channel_block,
            start,
            token,
            length,
            slot,
            False,
            sx_token,
            sx_channel,
            so_token,
            so_channel,
            sw_channel,
            sw_tap,
            ss_slot,
            ss_channel,
            ss_token,
            CHANNELS,
            WIDTH,
            HAS_BIAS,
            SILU,
            WINDOW,
            TOKEN_PARALLEL,
            BLOCK_M,
            BLOCK_N,
            WINDOW_ROWS,
            NUM_WARPS,
            False,
        )
