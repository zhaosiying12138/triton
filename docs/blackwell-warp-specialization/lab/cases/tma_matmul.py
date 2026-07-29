"""Compile-only TMA matmul used to expose Blackwell automatic WS.

This deliberately imports neither torch nor a CUDA runtime API.  Importing the
module constructs a JIT function; ``make_source`` returns an ASTSource for the
offline compiler and never launches it.
"""

import triton
import triton.language as tl


@triton.jit
def tma_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    WARP_SPECIALIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # B is represented as [N, K] so its TMA tile can be transposed immediately
    # before tl.dot, matching Triton's existing automatic-WS test case.
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load((0, k))
        b = b_desc.load((0, k))
        accumulator = tl.dot(a, b.T, accumulator)

    c_desc.store((0, 0), accumulator.to(tl.float16))


def make_source(warp_specialize: bool):
    """Return a fixed-shape ASTSource; this function performs no device work."""
    from triton.compiler import ASTSource

    signature = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp16",
        "c_ptr": "*fp16",
    }
    constexprs = {
        "WARP_SPECIALIZE": bool(warp_specialize),
        "NUM_STAGES": 3,
        "M": 1024,
        "N": 1024,
        "K": 1024,
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
    }
    attrs = {
        (0,): [["tt.divisibility", 16]],
        (1,): [["tt.divisibility", 16]],
        (2,): [["tt.divisibility", 16]],
    }
    return ASTSource(fn=tma_matmul_kernel, signature=signature, constexprs=constexprs, attrs=attrs)
