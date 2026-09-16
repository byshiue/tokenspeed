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


"""Explicit NVIDIA verify arithmetic shared by Triton and Gluon callers.

The 128-key reduction pairs adjacent logical elements at every level, regardless
of the launch layout. Products and additions round separately; only the state
correction uses FMA. This is a numerical contract, not an emulation of a particular
compiler's value-row-dependent contractions. Accepted replay keeps its own
producer/state contract. No global compiler flags or additional GPU launch.
"""

from tokenspeed_kernel._triton import gl, tl, triton


@triton.jit
def _verify_mul(left, right, language: tl.constexpr):
    return language.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=(left, right),
        dtype=language.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _verify_add(left, right, language: tl.constexpr):
    return language.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=(left, right),
        dtype=language.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _verify_sum(value, language: tl.constexpr):
    # A source-level sum otherwise inherits its tree from the register layout.
    language.static_assert(value.shape[-1] == 128)
    for level in language.static_range(7):
        paired = language.reshape(value, value.shape[:-1] + [value.shape[-1] // 2, 2])
        if language == gl:
            # Gluon split needs each logical pair in one thread. Layout
            # conversion changes ownership, never the arithmetic pairing.
            if len(value.shape) == 1:
                paired = gl.convert_layout(
                    paired,
                    gl.BlockedLayout([1, 2], [32, 1], [gl.num_warps(), 1], [0, 1]),
                )
            else:
                paired = gl.convert_layout(
                    paired,
                    gl.BlockedLayout(
                        [1, 1, 2], [1, 32, 1], [gl.num_warps(), 1, 1], [1, 0, 2]
                    ),
                )
        left, right = language.split(paired)
        value = _verify_add(left, right, language)
    return language.sum(value, axis=-1)


@triton.jit
def verify_normalize(query, key, scale: tl.constexpr, language: tl.constexpr):
    """Normalize FP32 vectors and scale Q before the output projection."""
    q_norm = _verify_add(
        _verify_sum(_verify_mul(query, query, language), language), 1e-6, language
    )
    k_norm = _verify_add(
        _verify_sum(_verify_mul(key, key, language), language), 1e-6, language
    )
    query = language.inline_asm_elementwise(
        "{ .reg .f32 norm; sqrt.rn.f32 norm, $2; div.rn.f32 $0, $1, norm; }",
        constraints="=f,f,f",
        args=(query, q_norm),
        dtype=language.float32,
        is_pure=True,
        pack=1,
    )
    key = language.inline_asm_elementwise(
        "{ .reg .f32 norm; sqrt.rn.f32 norm, $2; div.rn.f32 $0, $1, norm; }",
        constraints="=f,f,f",
        args=(key, k_norm),
        dtype=language.float32,
        is_pure=True,
        pack=1,
    )
    return _verify_mul(query, scale, language), key


@triton.jit
def verify_recurrence(state, query, key, value, decay, beta, language: tl.constexpr):
    """Return (state, output, correction) with one value-row-independent rule."""
    state = _verify_mul(state, decay[None, :], language)
    projection = _verify_sum(_verify_mul(state, key[None, :], language), language)
    if language == gl:
        projection = gl.convert_layout(projection, value.type.layout)
    correction = _verify_mul(_verify_add(value, -projection, language), beta, language)
    state = language.fma(correction[:, None], key[None, :], state)
    output = _verify_sum(_verify_mul(state, query[None, :], language), language)
    if language == gl:
        output = gl.convert_layout(output, value.type.layout)
    return state, output, correction
