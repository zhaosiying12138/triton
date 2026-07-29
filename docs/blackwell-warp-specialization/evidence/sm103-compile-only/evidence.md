# Curated SM103 compile-only evidence

Locked target: `cuda:103:32`. Cubins were compiled; any disassembly treats them only as data, and none were launched.

| Case | WS | Result | Required families present |
|---|---:|---:|---|
| `tma_matmul` | `off` | PASS | `tma`, `tcgen05`, `tmem`, `mbarrier` |
| `tma_matmul` | `on` | PASS | `warp_specialize`, `tma`, `tcgen05`, `tmem`, `mbarrier` |
| `ws_skeleton` | `off` | PASS | — |
| `ws_skeleton` | `on` | PASS | `warp_specialize`, `named_barrier`, `setmaxnreg` |

## tma_matmul / WS off

| Family | Contract | Observed | First evidence |
|---|---|---:|---|
| `warp_specialize` | forbidden | no | — |
| `aref` | forbidden | no | — |
| `tma` | required | yes | `ptx:345` `@%p9 cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%r6], [%rd9, {%r426, %r426}], [%r12];` |
| `tcgen05` | required | yes | `ptx:411` `@%p25 tcgen05.mma.cta_group::1.kind::f16 [ %r205 + 0 ], %rd17, %rd18, %r34, %p24;` |
| `tmem` | required | yes | `ptx:38` `@%p1 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r6], 128;` |
| `mbarrier` | required | yes | `ptx:336` `@%p3 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r12], 32768;` |
| `named_barrier` | optional | no | — |
| `cluster_barrier` | forbidden | no | — |
| `setmaxnreg` | forbidden | no | — |

## tma_matmul / WS on

| Family | Contract | Observed | First evidence |
|---|---|---:|---|
| `warp_specialize` | required | yes | `ttgir:70` `ttg.warp_specialize(%accumulator_13, %accumulator_8, %a, %accumulator_6, %accumulator_9, %accumulator_4, %a_desc_0, %b_desc_1) attributes {requestedRegisters = array<i32: 24, 24>}` |
| `aref` | optional | yes | `trace:2356` `%a_4 = nvws.aref.create %a : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> loc(#loc21)` |
| `tma` | required | yes | `ptx:825` `@%p41 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd32, {%r188, %r50}], [%r189];` |
| `tcgen05` | required | yes | `ptx:907` `@%p14 tcgen05.mma.cta_group::1.kind::f16 [ %r2 + 0 ], %rd4, %rd5, %r24, %p13;` |
| `tmem` | required | yes | `ptx:33` `@%p1 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r7], 128;` |
| `mbarrier` | required | yes | `ptx:1067` `@%p4 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r11], 32768;` |
| `named_barrier` | optional | yes | `ptx:378` `barrier.sync 	1;` |
| `cluster_barrier` | optional | no | — |
| `setmaxnreg` | optional | yes | `ptx:54` `setmaxnreg.inc.sync.aligned.u32 	256;` |

## ws_skeleton / WS off

| Family | Contract | Observed | First evidence |
|---|---|---:|---|
| `warp_specialize` | forbidden | no | — |
| `aref` | forbidden | no | — |
| `tma` | optional | no | — |
| `tcgen05` | optional | no | — |
| `tmem` | optional | no | — |
| `mbarrier` | optional | no | — |
| `named_barrier` | forbidden | no | — |
| `cluster_barrier` | forbidden | no | — |
| `setmaxnreg` | forbidden | no | — |

## ws_skeleton / WS on

| Family | Contract | Observed | First evidence |
|---|---|---:|---|
| `warp_specialize` | required | yes | `ttgir:6` `ttg.warp_specialize(%arg0) attributes {requestedRegisters = array<i32: 24>}` |
| `aref` | optional | no | — |
| `tma` | optional | no | — |
| `tcgen05` | optional | no | — |
| `tmem` | optional | no | — |
| `mbarrier` | optional | no | — |
| `named_barrier` | required | yes | `ptx:41` `barrier.sync 	1;` |
| `cluster_barrier` | optional | no | — |
| `setmaxnreg` | required | yes | `ptx:38` `setmaxnreg.inc.sync.aligned.u32 	256;` |

This report proves compiler structure only. It contains no runtime correctness or performance result.
