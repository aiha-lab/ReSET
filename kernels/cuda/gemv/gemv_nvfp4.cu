// NVFP4 CUDA-core batched GEMV kernel (small-M decode path).

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace nvfp4r {
namespace {

struct MatVecArgs {
    using stride_t = uint64_t;

    int  batch_count;
    int  row_count;
    int  k_packed;
    int  k_full_unused;
    float alpha;

    void* __restrict__ weight;
    void* __restrict__ vec;
    void* __restrict__ weight_sf;
    void* __restrict__ vec_sf;
    void* __restrict__ output;

    stride_t weight_stride_l;
    stride_t vec_stride_l;
    stride_t weight_sf_stride_l;
    stride_t vec_sf_stride_l;
    stride_t output_stride_l;

    stride_t weight_stride_m;
    stride_t vec_stride_m;
    stride_t weight_sf_stride_m;
    stride_t vec_sf_stride_m;
    stride_t output_stride_m;
};

constexpr int kWarpSize = 32;

__device__ __forceinline__
void cvt_fp4x8_to_fp16x8(uint32_t in, uint32_t out_h2[4]) {
    asm volatile(
        "{\n\t"
        ".reg .b8 b0, b1, b2, b3;\n\t"
        "mov.b32 {b0, b1, b2, b3}, %4;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n\t"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n\t"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n\t"
        "}"
        : "=r"(out_h2[0]), "=r"(out_h2[1]), "=r"(out_h2[2]), "=r"(out_h2[3])
        : "r"(in));
}

__device__ __forceinline__
half2 cvt_fp8x2_to_fp16x2(uint16_t in) {
    half2 out;
    uint32_t out_bits;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
                 : "=r"(out_bits) : "h"(in));
    *reinterpret_cast<uint32_t*>(&out) = out_bits;
    return out;
}

__device__ __forceinline__
void fma_f32_f16(float& acc, __half a, __half b) {
    asm volatile("fma.rn.f32.f16 %0, %1, %2, %0;"
                 : "+f"(acc)
                 : "h"(*reinterpret_cast<const uint16_t*>(&a)),
                   "h"(*reinterpret_cast<const uint16_t*>(&b)));
}

__device__ __forceinline__
void ld_cs_u32x4(uint32_t dst[4], const void* src) {
    asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
                 : "l"(src));
}

__device__ __forceinline__
void ld_cs_u32x8(uint32_t dst[8], const void* src) {
    asm volatile(
        "ld.global.L1::no_allocate.L2::evict_first.v8.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(dst[0]),"=r"(dst[1]),"=r"(dst[2]),"=r"(dst[3]),
          "=r"(dst[4]),"=r"(dst[5]),"=r"(dst[6]),"=r"(dst[7])
        : "l"(src));
}

__device__ __forceinline__
void ld_cs_u16(uint16_t* dst, const void* src) {
    asm volatile("ld.global.L1::no_allocate.b16 %0, [%1];"
                 : "=h"(*dst) : "l"(src));
}

__device__ __forceinline__
void ld_ca_u32x4(uint32_t dst[4], const void* src) {
    asm volatile("ld.global.L1::evict_last.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
                 : "l"(src));
}

__device__ __forceinline__
void ld_ca_u32x8(uint32_t dst[8], const void* src) {
    asm volatile(
        "ld.global.L1::evict_last.L2::evict_last.v8.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(dst[0]),"=r"(dst[1]),"=r"(dst[2]),"=r"(dst[3]),
          "=r"(dst[4]),"=r"(dst[5]),"=r"(dst[6]),"=r"(dst[7])
        : "l"(src));
}

__device__ __forceinline__
void ld_ca_u16(uint16_t* dst, const void* src) {
    asm volatile("ld.global.L1::evict_last.b16 %0, [%1];"
                 : "=h"(*dst) : "l"(src));
}

__device__ __forceinline__
void ld_ca_u32(uint32_t* dst, const void* src) {
    asm volatile("ld.global.L1::evict_last.b32 %0, [%1];"
                 : "=r"(*dst) : "l"(src));
}

__device__ __forceinline__
void ld_ca_bf16x16(__nv_bfloat162 dst[4], const void* src) {
    uint32_t r0, r1, r2, r3;
    asm volatile("ld.global.L1::evict_last.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "l"(src));
    dst[0] = *reinterpret_cast<__nv_bfloat162*>(&r0);
    dst[1] = *reinterpret_cast<__nv_bfloat162*>(&r1);
    dst[2] = *reinterpret_cast<__nv_bfloat162*>(&r2);
    dst[3] = *reinterpret_cast<__nv_bfloat162*>(&r3);
}

__device__ __forceinline__
void ld_ca_h2x16(half2 dst[8], const void* src) {
    uint32_t r0, r1, r2, r3;
    asm volatile("ld.global.L1::evict_last.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "l"(src));
    *reinterpret_cast<uint32_t*>(&dst[0]) = r0;
    *reinterpret_cast<uint32_t*>(&dst[2]) = r1;
    *reinterpret_cast<uint32_t*>(&dst[4]) = r2;
    *reinterpret_cast<uint32_t*>(&dst[6]) = r3;
}

__device__ __forceinline__
void mma_m16n8k16_f16_f32(float d[4],
                          const uint32_t a[4], const uint32_t b[2],
                          const float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__
void ldmatrix_x4(uint32_t r[4], uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(smem_addr));
}

__device__ __forceinline__
void ldmatrix_x2_trans(uint32_t r[2], uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(smem_addr));
}

__device__ __forceinline__
void ldmatrix_x2(uint32_t r[2], uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(smem_addr));
}

__device__ __forceinline__
void ld_ca_u32x2(uint32_t dst[2], const void* src) {
    asm volatile("ld.global.L1::evict_last.v2.b32 {%0, %1}, [%2];"
                 : "=r"(dst[0]), "=r"(dst[1]) : "l"(src));
}

__device__ __forceinline__
void ld_ca_u8(uint8_t* dst, const void* src) {
    uint32_t tmp;
    asm volatile("ld.global.L1::evict_last.b8 %0, [%1];" : "=r"(tmp) : "l"(src));
    *dst = static_cast<uint8_t>(tmp);
}

template <typename OutT>
__device__ __forceinline__
void store_scaled(void* out_ptr, size_t off, float v_scaled) {
    if constexpr (std::is_same_v<OutT, __half>) {
        reinterpret_cast<__half*>(out_ptr)[off] = __float2half(v_scaled);
    } else {
        reinterpret_cast<__nv_bfloat16*>(out_ptr)[off] = __float2bfloat16(v_scaled);
    }
}

template <int B_BATCH, int BLOCK_M, int BLOCK_K, int NUM_WARPS,
          typename OutT, int SEG_BYTES = 32,
          bool kStreamWeights = true, bool kFp16Accum = false,
          bool kPersistent = false>
__global__ __launch_bounds__(NUM_WARPS * kWarpSize)
void mv_kernel(const __grid_constant__ MatVecArgs args) {
    static_assert(SEG_BYTES == 16 || SEG_BYTES == 32,
                  "SEG_BYTES must be 16 (v4.b32) or 32 (v8.b32)");
    static_assert(BLOCK_K % SEG_BYTES == 0,
                  "BLOCK_K must be a multiple of SEG_BYTES");
    static_assert(BLOCK_K % 8 == 0,
                  "BLOCK_K must include whole FP8 scale blocks (8 bytes each)");
    static_assert(B_BATCH >= 1, "B_BATCH must be >= 1");
    constexpr int N_U32_PER_SEG     = SEG_BYTES / 4;
    constexpr int N_SF_BLKS         = SEG_BYTES / 8;
    constexpr int TB_SIZE   = NUM_WARPS * kWarpSize;
    constexpr int NUM_COLS  = BLOCK_K / SEG_BYTES;
    constexpr int TB_WIDTH  = (NUM_COLS < TB_SIZE) ? NUM_COLS : TB_SIZE;
    constexpr int TB_HEIGHT = TB_SIZE / TB_WIDTH;
    constexpr int M_PER_THR = BLOCK_M / TB_HEIGHT;
    constexpr int K_PER_THR = NUM_COLS / TB_WIDTH;

    static_assert(M_PER_THR >= 1,
                  "BLOCK_M too small for the chosen NUM_WARPS / BLOCK_K "
                  "(thread-block has more rows than BLOCK_M)");
    static_assert(K_PER_THR >= 1,
                  "BLOCK_K too small for the chosen NUM_WARPS "
                  "(thread-block has more columns than NUM_COLS)");

    constexpr int SF_BYTES_PER_TILE = BLOCK_K / 8;

    constexpr int HALF2_PER_SEG       = SEG_BYTES;

    constexpr int HALF2_PER_SCALE_BLK = 8;

    const int tid        = threadIdx.x;
    const int t_col      = tid % TB_WIDTH;
    const int t_row      = tid / TB_WIDTH;

    const int M     = args.row_count;
    const int K_pkd = args.k_packed;
    const int L     = args.batch_count;

    const int grid_n = M / BLOCK_M;
    const int grid_b = L / B_BATCH;
    const int total_tiles = grid_n * grid_b;
    int tile_id, tile_stride;
    if constexpr (kPersistent) {
        tile_id     = static_cast<int>(blockIdx.x);
        tile_stride = static_cast<int>(gridDim.x);
    } else {
        tile_id     = static_cast<int>(blockIdx.y) * grid_n
                      + static_cast<int>(blockIdx.x);
        tile_stride = total_tiles;
    }

    for (; tile_id < total_tiles; tile_id += tile_stride) {
    const int b_tile     = tile_id / grid_n;
    const int n_tile     = tile_id - b_tile * grid_n;
    const int batch_base = b_tile * B_BATCH;
    const int off_m      = n_tile * BLOCK_M;

    const uint8_t* A_ptr   = static_cast<const uint8_t*>(args.weight)
                             + batch_base * args.weight_stride_l
                             + static_cast<size_t>(off_m) * args.weight_stride_m;
    const uint8_t* SFA_ptr = static_cast<const uint8_t*>(args.weight_sf)
                             + batch_base * args.weight_sf_stride_l
                             + static_cast<size_t>(off_m) * args.weight_sf_stride_m;

    const uint8_t* B_ptrs  [B_BATCH];
    const uint8_t* SFB_ptrs[B_BATCH];
    #pragma unroll
    for (int b = 0; b < B_BATCH; ++b) {
        B_ptrs[b]   = static_cast<const uint8_t*>(args.vec)
                      + static_cast<size_t>(batch_base + b) * args.vec_stride_l;
        SFB_ptrs[b] = static_cast<const uint8_t*>(args.vec_sf)
                      + static_cast<size_t>(batch_base + b) * args.vec_sf_stride_l;
    }

    uint32_t A_raw  [M_PER_THR][K_PER_THR][N_U32_PER_SEG];

    uint32_t SFA_raw[M_PER_THR][K_PER_THR];
    half2    A_h2   [M_PER_THR][K_PER_THR][HALF2_PER_SEG];

    half2    SFA_h2 [M_PER_THR][K_PER_THR][N_SF_BLKS / 2];

    half2 B_h2_tmp [K_PER_THR][HALF2_PER_SEG];

    uint32_t B_raw     [B_BATCH][K_PER_THR][N_U32_PER_SEG];
    uint32_t SFB_raw   [B_BATCH][K_PER_THR];
    half2    SFB_h2_tmp[K_PER_THR][N_SF_BLKS / 2];

    using AccT = std::conditional_t<kFp16Accum, __half, float>;
    AccT master_acc[M_PER_THR][B_BATCH];
    #pragma unroll
    for (int m = 0; m < M_PER_THR; ++m)
        #pragma unroll
        for (int b = 0; b < B_BATCH; ++b) {
            if constexpr (kFp16Accum) master_acc[m][b] = __float2half(0.f);
            else                      master_acc[m][b] = 0.f;
        }

    const int n_iters = K_pkd / BLOCK_K;

    for (int it = 0; it < n_iters; ++it) {

        #pragma unroll
        for (int b = 0; b < B_BATCH; ++b) {
            #pragma unroll
            for (int k = 0; k < K_PER_THR; ++k) {
                const int col = k * TB_WIDTH + t_col;
                if constexpr (SEG_BYTES == 32) {
                    ld_ca_u32x8(B_raw[b][k],   B_ptrs[b]   + col * 32);
                    ld_ca_u32  (&SFB_raw[b][k], SFB_ptrs[b] + col * 4);
                } else {
                    ld_ca_u32x4(B_raw[b][k], B_ptrs[b] + col * 16);
                    uint16_t sf16;
                    ld_ca_u16(&sf16, SFB_ptrs[b] + col * 2);
                    SFB_raw[b][k] = sf16;
                }
            }
        }

        #pragma unroll
        for (int m = 0; m < M_PER_THR; ++m) {
            const int row = m * TB_HEIGHT + t_row;
            const uint8_t* A_row   = A_ptr   + static_cast<size_t>(row) * args.weight_stride_m;
            const uint8_t* SFA_row = SFA_ptr + static_cast<size_t>(row) * args.weight_sf_stride_m;
            #pragma unroll
            for (int k = 0; k < K_PER_THR; ++k) {
                const int col = k * TB_WIDTH + t_col;
                if constexpr (SEG_BYTES == 32) {
                    if constexpr (kStreamWeights)
                        ld_cs_u32x8(A_raw[m][k],   A_row   + col * 32);
                    else
                        ld_ca_u32x8(A_raw[m][k],   A_row   + col * 32);
                    ld_ca_u32  (&SFA_raw[m][k], SFA_row + col * 4);
                } else {
                    if constexpr (kStreamWeights)
                        ld_cs_u32x4(A_raw[m][k], A_row + col * 16);
                    else
                        ld_ca_u32x4(A_raw[m][k], A_row + col * 16);
                    uint16_t sf16;
                    if constexpr (kStreamWeights)
                        ld_cs_u16(&sf16, SFA_row + col * 2);
                    else
                        ld_ca_u16(&sf16, SFA_row + col * 2);
                    SFA_raw[m][k] = sf16;
                }
            }
        }

        A_ptr   += BLOCK_K;
        SFA_ptr += SF_BYTES_PER_TILE;
        #pragma unroll
        for (int b = 0; b < B_BATCH; ++b) {
            B_ptrs[b]   += BLOCK_K;
            SFB_ptrs[b] += SF_BYTES_PER_TILE;
        }

        #pragma unroll
        for (int m = 0; m < M_PER_THR; ++m) {
            #pragma unroll
            for (int k = 0; k < K_PER_THR; ++k) {
                #pragma unroll
                for (int q = 0; q < N_U32_PER_SEG; ++q) {
                    cvt_fp4x8_to_fp16x8(A_raw[m][k][q],
                                        reinterpret_cast<uint32_t*>(&A_h2[m][k][q * 4]));
                }

                #pragma unroll
                for (int s = 0; s < N_SF_BLKS / 2; ++s) {
                    SFA_h2[m][k][s] = cvt_fp8x2_to_fp16x2(
                        static_cast<uint16_t>(SFA_raw[m][k] >> (s * 16)));
                }
            }
        }

        #pragma unroll
        for (int b = 0; b < B_BATCH; ++b) {

            #pragma unroll
            for (int k = 0; k < K_PER_THR; ++k) {
                #pragma unroll
                for (int q = 0; q < N_U32_PER_SEG; ++q) {
                    cvt_fp4x8_to_fp16x8(B_raw[b][k][q],
                                        reinterpret_cast<uint32_t*>(&B_h2_tmp[k][q * 4]));
                }
                #pragma unroll
                for (int s = 0; s < N_SF_BLKS / 2; ++s) {
                    SFB_h2_tmp[k][s] = cvt_fp8x2_to_fp16x2(
                        static_cast<uint16_t>(SFB_raw[b][k] >> (s * 16)));
                }
            }

            #pragma unroll
            for (int m = 0; m < M_PER_THR; ++m) {
                #pragma unroll
                for (int k = 0; k < K_PER_THR; ++k) {
                    #pragma unroll
                    for (int s = 0; s < N_SF_BLKS / 2; ++s) {
                        const int off_lo = s * 2 * HALF2_PER_SCALE_BLK;
                        const int off_hi = off_lo + HALF2_PER_SCALE_BLK;
                        half2 acc_lo_a = __hmul2(A_h2[m][k][off_lo + 0], B_h2_tmp[k][off_lo + 0]);
                        half2 acc_lo_b = __hmul2(A_h2[m][k][off_lo + 1], B_h2_tmp[k][off_lo + 1]);
                        half2 acc_hi_a = __hmul2(A_h2[m][k][off_hi + 0], B_h2_tmp[k][off_hi + 0]);
                        half2 acc_hi_b = __hmul2(A_h2[m][k][off_hi + 1], B_h2_tmp[k][off_hi + 1]);
                        #pragma unroll
                        for (int j = 2; j < HALF2_PER_SCALE_BLK; j += 2) {
                            acc_lo_a = __hfma2(A_h2[m][k][off_lo + j],     B_h2_tmp[k][off_lo + j],     acc_lo_a);
                            acc_lo_b = __hfma2(A_h2[m][k][off_lo + j + 1], B_h2_tmp[k][off_lo + j + 1], acc_lo_b);
                            acc_hi_a = __hfma2(A_h2[m][k][off_hi + j],     B_h2_tmp[k][off_hi + j],     acc_hi_a);
                            acc_hi_b = __hfma2(A_h2[m][k][off_hi + j + 1], B_h2_tmp[k][off_hi + j + 1], acc_hi_b);
                        }
                        const half2  acc_lo = __hadd2(acc_lo_a, acc_lo_b);
                        const half2  acc_hi = __hadd2(acc_hi_a, acc_hi_b);
                        const __half lo     = __hadd(acc_lo.x, acc_lo.y);
                        const __half hi     = __hadd(acc_hi.x, acc_hi.y);
                        if constexpr (kFp16Accum) {
                            half2 scale = __hmul2(SFA_h2[m][k][s], SFB_h2_tmp[k][s]);
                            master_acc[m][b] = __hfma(lo, scale.x, master_acc[m][b]);
                            master_acc[m][b] = __hfma(hi, scale.y, master_acc[m][b]);
                        } else {
                            // Compute the per-block scale product in FP32: each
                            // E4M3 block scale can reach 448, and 448*448 >> the
                            // FP16 max (65504), so __hmul2 would overflow to Inf
                            // (-> NaN) whenever a weight block and its activation
                            // block are both large (e.g. attention-output o_proj).
                            const float sx = __half2float(SFA_h2[m][k][s].x)
                                           * __half2float(SFB_h2_tmp[k][s].x);
                            const float sy = __half2float(SFA_h2[m][k][s].y)
                                           * __half2float(SFB_h2_tmp[k][s].y);
                            master_acc[m][b] = fmaf(__half2float(lo), sx, master_acc[m][b]);
                            master_acc[m][b] = fmaf(__half2float(hi), sy, master_acc[m][b]);
                        }
                    }
                }
            }
        }
    }

    if constexpr (TB_WIDTH > kWarpSize) {
        __shared__ AccT smem[M_PER_THR][B_BATCH][TB_SIZE];
        #pragma unroll
        for (int m = 0; m < M_PER_THR; ++m)
            #pragma unroll
            for (int b = 0; b < B_BATCH; ++b)
                smem[m][b][tid] = master_acc[m][b];
        __syncthreads();
        #pragma unroll
        for (int stride = TB_WIDTH / 2; stride >= kWarpSize; stride /= 2) {
            if ((tid % TB_WIDTH) < stride) {
                #pragma unroll
                for (int m = 0; m < M_PER_THR; ++m) {
                    #pragma unroll
                    for (int b = 0; b < B_BATCH; ++b) {
                        if constexpr (kFp16Accum) {
                            master_acc[m][b] = __hadd(master_acc[m][b],
                                                      smem[m][b][tid + stride]);
                        } else {
                            master_acc[m][b] += smem[m][b][tid + stride];
                        }
                        smem[m][b][tid] = master_acc[m][b];
                    }
                }
            }
            __syncthreads();
        }
    }

    constexpr int WARP_REDUCE_STRIDE = (TB_WIDTH < kWarpSize ? TB_WIDTH : kWarpSize) / 2;
    #pragma unroll
    for (int stride = WARP_REDUCE_STRIDE; stride > 0; stride /= 2) {
        #pragma unroll
        for (int m = 0; m < M_PER_THR; ++m) {
            #pragma unroll
            for (int b = 0; b < B_BATCH; ++b) {
                if constexpr (kFp16Accum) {

                    __half other = __shfl_down_sync(0xffffffffu, master_acc[m][b], stride);
                    master_acc[m][b] = __hadd(master_acc[m][b], other);
                } else {
                    master_acc[m][b] +=
                        __shfl_down_sync(0xffffffffu, master_acc[m][b], stride);
                }
            }
        }
    }

    if (t_col == 0) {
        #pragma unroll
        for (int m = 0; m < M_PER_THR; ++m) {
            const int row        = m * TB_HEIGHT + t_row;
            const int row_global = off_m + row;
            if (row_global < M) {
                #pragma unroll
                for (int b = 0; b < B_BATCH; ++b) {
                    const size_t out_off =
                        static_cast<size_t>(batch_base + b) * args.output_stride_l
                        + static_cast<size_t>(row_global) * args.output_stride_m;
                    float v;
                    if constexpr (kFp16Accum) v = __half2float(master_acc[m][b]);
                    else                      v = master_acc[m][b];
                    store_scaled<OutT>(args.output, out_off, v * args.alpha);
                }
            }
        }
    }

    if constexpr (kPersistent && (TB_WIDTH > kWarpSize)) {
        __syncthreads();
    }
    }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARPS, typename OutT>
__global__ __launch_bounds__(NUM_WARPS * kWarpSize)
void mv_smma_kernel(const __grid_constant__ MatVecArgs args) {
    static_assert(BLOCK_M == 16, "smma kernel uses fixed mma.M=16");
    static_assert(BLOCK_N % 16 == 0, "BLOCK_N must be multiple of 16");
    static_assert(BLOCK_N % NUM_WARPS == 0, "BLOCK_N must divide by NUM_WARPS");
    static_assert((BLOCK_N / NUM_WARPS) % 8 == 0, "N_PER_WARP must be multiple of mma.N=8");
    static_assert(BLOCK_K % 16 == 0, "BLOCK_K must be multiple of mma.K=16");
    static_assert(BLOCK_K >= 32, "BLOCK_K must be >=32 for the v4.b32 weight load layout");
    static_assert((BLOCK_N * BLOCK_K / 2) % (NUM_WARPS * kWarpSize) == 0,
                  "weight tile bytes must divide evenly across threads");

    constexpr int N_PER_WARP = BLOCK_N / NUM_WARPS;
    constexpr int N_MMAS     = N_PER_WARP / 8;
    constexpr int K_MMAS     = BLOCK_K / 16;
    constexpr int TB_SIZE    = NUM_WARPS * kWarpSize;

    constexpr int K_BYTES_PER_THR_W  = (BLOCK_N * BLOCK_K / 2) / TB_SIZE;
    constexpr int K_ELEMS_PER_THR_W  = K_BYTES_PER_THR_W * 2;
    constexpr int N_SCALES_PER_THR_W = K_ELEMS_PER_THR_W / 16;
    constexpr int N_THR_PER_ROW_W    = BLOCK_K / K_ELEMS_PER_THR_W;
    static_assert(K_BYTES_PER_THR_W == 16 || K_BYTES_PER_THR_W == 32,
                  "smma weight load: only 16B (BLOCK_K=64) or 32B (BLOCK_K=128) per-thread "
                  "load paths are wired up.");
    static_assert(N_THR_PER_ROW_W >= 1, "");

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int K_pkd  = args.k_packed;
    const int L      = args.batch_count;
    const int n_off  = static_cast<int>(blockIdx.x) * BLOCK_N;

    extern __shared__ __align__(16) uint8_t smem_storage[];
    half (*A_smem)[BLOCK_K] = reinterpret_cast<half(*)[BLOCK_K]>(smem_storage);
    half (*B_smem)[BLOCK_K] = reinterpret_cast<half(*)[BLOCK_K]>(
        smem_storage + BLOCK_M * BLOCK_K * sizeof(half));

    float D[N_MMAS][4];
    #pragma unroll
    for (int nm = 0; nm < N_MMAS; ++nm) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) D[nm][i] = 0.f;
    }

    const uint8_t* W_base   = static_cast<const uint8_t*>(args.weight)
                              + static_cast<size_t>(n_off) * args.weight_stride_m;
    const uint8_t* SFW_base = static_cast<const uint8_t*>(args.weight_sf)
                              + static_cast<size_t>(n_off) * args.weight_sf_stride_m;
    const uint8_t* X_base   = static_cast<const uint8_t*>(args.vec);
    const uint8_t* SFX_base = static_cast<const uint8_t*>(args.vec_sf);

    const int K_TILES = (K_pkd * 2) / BLOCK_K;

    for (int kt = 0; kt < K_TILES; ++kt) {
        const int k_off_elem = kt * BLOCK_K;
        const int k_off_pkd  = k_off_elem / 2;
        const int k_off_sf   = k_off_elem / 16;

        {
            const int n_tile     = tid / N_THR_PER_ROW_W;
            const int col_chunk  = tid % N_THR_PER_ROW_W;
            const int k_byte_off = col_chunk * K_BYTES_PER_THR_W;
            const int k_elem_off = col_chunk * K_ELEMS_PER_THR_W;

            const uint8_t* w_ptr = W_base
                + static_cast<size_t>(n_tile) * args.weight_stride_m
                + k_off_pkd + k_byte_off;

            constexpr int N_U32_W = K_BYTES_PER_THR_W / 4;
            constexpr int N_HALF2_W = K_ELEMS_PER_THR_W / 2;
            uint32_t w_raw[N_U32_W];
            if constexpr (K_BYTES_PER_THR_W == 16) {
                ld_cs_u32x4(w_raw, w_ptr);
            } else {
                ld_cs_u32x8(w_raw, w_ptr);
            }

            const uint8_t* sfw_ptr = SFW_base
                + static_cast<size_t>(n_tile) * args.weight_sf_stride_m
                + k_off_sf + col_chunk * N_SCALES_PER_THR_W;
            uint32_t sf_raw_u32 = 0;
            if constexpr (N_SCALES_PER_THR_W == 2) {
                uint16_t sf16; ld_cs_u16(&sf16, sfw_ptr);
                sf_raw_u32 = sf16;
            } else {
                ld_ca_u32(&sf_raw_u32, sfw_ptr);
            }

            constexpr int N_SF_PAIRS = N_SCALES_PER_THR_W / 2;
            half2 sf_h2[N_SF_PAIRS];
            #pragma unroll
            for (int s = 0; s < N_SF_PAIRS; ++s) {
                sf_h2[s] = cvt_fp8x2_to_fp16x2(
                    static_cast<uint16_t>(sf_raw_u32 >> (s * 16)));
            }

            half2 w_h2[N_HALF2_W];
            #pragma unroll
            for (int i = 0; i < N_U32_W; ++i) {
                cvt_fp4x8_to_fp16x8(w_raw[i],
                                    reinterpret_cast<uint32_t*>(&w_h2[i * 4]));
            }

            #pragma unroll
            for (int s = 0; s < N_SF_PAIRS; ++s) {
                const half2 sa = half2{sf_h2[s].x, sf_h2[s].x};
                const half2 sb = half2{sf_h2[s].y, sf_h2[s].y};
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int idx_a = s * 16 + i;
                    const int idx_b = s * 16 + 8 + i;
                    w_h2[idx_a] = __hmul2(w_h2[idx_a], sa);
                    w_h2[idx_b] = __hmul2(w_h2[idx_b], sb);
                }
            }

            constexpr int N_UINT4_W = N_HALF2_W / 4;
            half* dst = &B_smem[n_tile][k_elem_off];
            uint4* dst_v4 = reinterpret_cast<uint4*>(dst);
            const uint4* src_v4 = reinterpret_cast<const uint4*>(w_h2);
            #pragma unroll
            for (int i = 0; i < N_UINT4_W; ++i) dst_v4[i] = src_v4[i];
        }

        {
            constexpr int X_BLOCKS_PER_TILE = BLOCK_M * (BLOCK_K / 16);
            if (tid < X_BLOCKS_PER_TILE) {
                const int row     = tid / (BLOCK_K / 16);
                const int blk     = tid % (BLOCK_K / 16);
                const int k_byte  = blk * 8;
                const int k_elem  = blk * 16;

                uint32_t x_raw[2] = {0u, 0u};
                uint16_t sf_raw   = 0;

                if (row < L) {
                    const uint8_t* x_ptr = X_base
                        + static_cast<size_t>(row) * args.vec_stride_l
                        + k_off_pkd + k_byte;
                    ld_ca_u32x2(x_raw, x_ptr);

                    const uint8_t* sfx_ptr = SFX_base
                        + static_cast<size_t>(row) * args.vec_sf_stride_l
                        + k_off_sf + blk;
                    uint8_t sf8;
                    ld_ca_u8(&sf8, sfx_ptr);
                    sf_raw = static_cast<uint16_t>(sf8);
                }

                half2 x_h2[8];
                cvt_fp4x8_to_fp16x8(x_raw[0], reinterpret_cast<uint32_t*>(&x_h2[0]));
                cvt_fp4x8_to_fp16x8(x_raw[1], reinterpret_cast<uint32_t*>(&x_h2[4]));

                if (row < L) {
                    const half2 sf_h2 = cvt_fp8x2_to_fp16x2(sf_raw);
                    const half2 s = half2{sf_h2.x, sf_h2.x};
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) x_h2[i] = __hmul2(x_h2[i], s);
                } else {

                    const __half hzero = __float2half(0.f);
                    const half2  zh2   = half2{hzero, hzero};
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) x_h2[i] = zh2;
                }

                half* dst = &A_smem[row][k_elem];
                uint4* dst_v4 = reinterpret_cast<uint4*>(dst);
                const uint4* src_v4 = reinterpret_cast<const uint4*>(x_h2);
                dst_v4[0] = src_v4[0];
                dst_v4[1] = src_v4[1];
            }
        }

        __syncthreads();

        const int n_warp_off = warp_id * N_PER_WARP;

        #pragma unroll
        for (int km = 0; km < K_MMAS; ++km) {
            const int k_mma_off = km * 16;

            uint32_t a_frag[4];
            {

                const int row    = lane & 0x0F;
                const int colg   = (lane >> 4) & 1;
                const half* src  = &A_smem[row][k_mma_off + colg * 8];
                const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
                ldmatrix_x4(a_frag, addr);
            }

            #pragma unroll
            for (int nm = 0; nm < N_MMAS; ++nm) {
                const int n_mma_off = n_warp_off + nm * 8;

                uint32_t b_frag[2];
                {
                    const int sub_row = lane & 7;
                    const int sub_id  = (lane >> 3) & 1;
                    const half* src   = &B_smem[n_mma_off + sub_row]
                                              [k_mma_off + sub_id * 8];
                    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
                    ldmatrix_x2(b_frag, addr);
                }

                mma_m16n8k16_f16_f32(D[nm], a_frag, b_frag, D[nm]);
            }
        }

        __syncthreads();
    }

    const float alpha = args.alpha;

    #pragma unroll
    for (int nm = 0; nm < N_MMAS; ++nm) {
        const int n_mma_off = n_off + warp_id * N_PER_WARP + nm * 8;
        const int r0 = lane >> 2;
        const int r1 = r0 + 8;
        const int c0 = (lane & 3) << 1;
        const int c1 = c0 + 1;

        if (r0 < L) {
            const size_t off0 = static_cast<size_t>(r0) * args.output_stride_l
                              + static_cast<size_t>(n_mma_off + c0) * args.output_stride_m;
            const size_t off1 = static_cast<size_t>(r0) * args.output_stride_l
                              + static_cast<size_t>(n_mma_off + c1) * args.output_stride_m;
            store_scaled<OutT>(args.output, off0, D[nm][0] * alpha);
            store_scaled<OutT>(args.output, off1, D[nm][1] * alpha);
        }
        if (r1 < L) {
            const size_t off2 = static_cast<size_t>(r1) * args.output_stride_l
                              + static_cast<size_t>(n_mma_off + c0) * args.output_stride_m;
            const size_t off3 = static_cast<size_t>(r1) * args.output_stride_l
                              + static_cast<size_t>(n_mma_off + c1) * args.output_stride_m;
            store_scaled<OutT>(args.output, off2, D[nm][2] * alpha);
            store_scaled<OutT>(args.output, off3, D[nm][3] * alpha);
        }
    }
}

template <int B_BATCH_, int BLOCK_M_, int BLOCK_K_, int NUM_WARPS_,
          int SEG_BYTES_, bool kStreamWeights_, bool kFp16Accum_>
inline void dispatch_outdtype(at::ScalarType out_dtype, dim3 grid, int threads,
                              cudaStream_t stream, const MatVecArgs& args,
                              bool kPersistent_runtime) {
    if (kPersistent_runtime) {
        if (out_dtype == at::kHalf) {
            mv_kernel<B_BATCH_, BLOCK_M_, BLOCK_K_, NUM_WARPS_, __half,
                      SEG_BYTES_, kStreamWeights_, kFp16Accum_, true>
                <<<grid, threads, 0, stream>>>(args);
        } else {
            mv_kernel<B_BATCH_, BLOCK_M_, BLOCK_K_, NUM_WARPS_, __nv_bfloat16,
                      SEG_BYTES_, kStreamWeights_, kFp16Accum_, true>
                <<<grid, threads, 0, stream>>>(args);
        }
    } else {
        if (out_dtype == at::kHalf) {
            mv_kernel<B_BATCH_, BLOCK_M_, BLOCK_K_, NUM_WARPS_, __half,
                      SEG_BYTES_, kStreamWeights_, kFp16Accum_, false>
                <<<grid, threads, 0, stream>>>(args);
        } else {
            mv_kernel<B_BATCH_, BLOCK_M_, BLOCK_K_, NUM_WARPS_, __nv_bfloat16,
                      SEG_BYTES_, kStreamWeights_, kFp16Accum_, false>
                <<<grid, threads, 0, stream>>>(args);
        }
    }
}

at::Tensor launch_mv(at::Tensor A, at::Tensor B, at::Tensor C,
                     at::Tensor SFA, at::Tensor SFB,
                     double alpha) {
    const auto sz = A.sizes();
    const int M  = sz[0];
    const int Kp = sz[1];
    const int L  = sz[2];

    MatVecArgs args{};
    args.batch_count   = L;
    args.row_count     = M;
    args.k_packed      = Kp;
    args.k_full_unused = 0;
    args.alpha         = static_cast<float>(alpha);

    args.weight    = A.data_ptr();
    args.vec       = B.data_ptr();
    args.weight_sf = SFA.data_ptr();
    args.vec_sf    = SFB.data_ptr();
    args.output    = C.data_ptr();

    args.weight_stride_l    = A.stride(2);
    args.vec_stride_l       = B.stride(2);
    args.weight_sf_stride_l = SFA.stride(2);
    args.vec_sf_stride_l    = SFB.stride(2);
    args.output_stride_l    = C.stride(2);

    args.weight_stride_m    = A.stride(0);
    args.vec_stride_m       = B.stride(0);
    args.weight_sf_stride_m = SFA.stride(0);
    args.vec_sf_stride_m    = SFB.stride(0);
    args.output_stride_m    = C.stride(0);

    auto stream_obj = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = stream_obj.stream();
    const auto out_dtype = C.scalar_type();

#define LAUNCH_COMMON(B_BATCH, BLOCK_M, BLOCK_K, NUM_WARPS)                            \
        TORCH_CHECK(L  % (B_BATCH) == 0,                                               \
                    "gemv: L (=", L, ") must be a multiple of B_BATCH=", (B_BATCH));   \
        TORCH_CHECK(M  % (BLOCK_M) == 0,                                               \
                    "gemv: M (=", M, ") must be a multiple of BLOCK_M=", (BLOCK_M));   \
        TORCH_CHECK(Kp % (BLOCK_K) == 0,                                               \
                    "gemv: K_packed (=", Kp, ") must be a multiple of "                \
                    "BLOCK_K=", (BLOCK_K));                                             \
        const int _total_tiles_lc = (M / (BLOCK_M)) * (L / (B_BATCH));                 \
        const int _grid_x_lc = kPersistent                                             \
            ? std::min<int>(kPersistentCTAs, _total_tiles_lc)                          \
            : (M / (BLOCK_M));                                                         \
        const int _grid_y_lc = kPersistent ? 1 : (L / (B_BATCH));                      \
        const dim3 grid(_grid_x_lc, _grid_y_lc);                                       \
        const int  threads = (NUM_WARPS) * kWarpSize

    static const bool kStreamWeights = []() {
        const char* env = std::getenv("NVFP4R_GEMV_STREAM_WEIGHTS");
        if (!env) return true;
        return std::atoi(env) != 0;
    }();

    static const bool kFp16Accum = []() {
        const char* env = std::getenv("NVFP4R_GEMV_FP16_ACCUM");
        if (!env) return false;
        return std::atoi(env) != 0;
    }();

    static const bool kPersistent = []() {
        const char* env = std::getenv("NVFP4R_GEMV_PERSISTENT");
        if (!env) return false;
        return std::atoi(env) != 0;
    }();

    static const int kPersistentCTAs = []() {
        const char* env = std::getenv("NVFP4R_GEMV_PERSISTENT_CTAS");
        if (env) {
            const int v = std::atoi(env);
            if (v > 0) return v;
        }
        int dev = 0;
        cudaGetDevice(&dev);
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        return sms > 0 ? sms : 144;
    }();

#define LAUNCH(B_BATCH, BLOCK_M, BLOCK_K, NUM_WARPS)                                   \
    do {                                                                               \
        LAUNCH_COMMON(B_BATCH, BLOCK_M, BLOCK_K, NUM_WARPS);                           \
        const int   _seg = kSegBytes;                                                  \
        const bool  _str = kStreamWeights;                                             \
        const bool  _fp16a = kFp16Accum;                                               \
        const bool  _persist = kPersistent;                                            \
        if      ( _seg==32 &&  _str &&  _fp16a) dispatch_outdtype<B_BATCH, BLOCK_M, BLOCK_K, NUM_WARPS, 32, true,  true >(out_dtype, grid, threads, stream, args, _persist); \
        else if ( _seg==32 &&  _str && !_fp16a) dispatch_outdtype<B_BATCH, BLOCK_M, BLOCK_K, NUM_WARPS, 32, true,  false>(out_dtype, grid, threads, stream, args, _persist); \
        else if ( _seg==32 && !_str &&  _fp16a) dispatch_outdtype<B_BATCH, BLOCK_M, BLOCK_K, NUM_WARPS, 32, false, true >(out_dtype, grid, threads, stream, args, _persist); \
        else if ( _seg==32 && !_str && !_fp16a) dispatch_outdtype<B_BATCH, BLOCK_M, BLOCK_K, NUM_WARPS, 32, false, false>(out_dtype, grid, threads, stream, args, _persist); \
        else if ( _seg==16 &&  _str &&  _fp16a) dispatch_outdtype<B_BATCH, BLOCK_M, BLOCK_K, NUM_WARPS, 16, true,  true >(out_dtype, grid, threads, stream, args, _persist); \
        else if ( _seg==16 &&  _str && !_fp16a) dispatch_outdtype<B_BATCH, BLOCK_M, BLOCK_K, NUM_WARPS, 16, true,  false>(out_dtype, grid, threads, stream, args, _persist); \
        else if ( _seg==16 && !_str &&  _fp16a) dispatch_outdtype<B_BATCH, BLOCK_M, BLOCK_K, NUM_WARPS, 16, false, true >(out_dtype, grid, threads, stream, args, _persist); \
        else                                    dispatch_outdtype<B_BATCH, BLOCK_M, BLOCK_K, NUM_WARPS, 16, false, false>(out_dtype, grid, threads, stream, args, _persist); \
    } while (0)

#define LAUNCH16(B_BATCH, BLOCK_M, BLOCK_K, NUM_WARPS)                                 \
    do {                                                                               \
        LAUNCH_COMMON(B_BATCH, BLOCK_M, BLOCK_K, NUM_WARPS);                           \
        const bool _str   = kStreamWeights;                                            \
        const bool _fp16a = kFp16Accum;                                                \
        const bool _persist = kPersistent;                                             \
        if      ( _str &&  _fp16a) dispatch_outdtype<B_BATCH, BLOCK_M, BLOCK_K, NUM_WARPS, 16, true,  true >(out_dtype, grid, threads, stream, args, _persist); \
        else if ( _str && !_fp16a) dispatch_outdtype<B_BATCH, BLOCK_M, BLOCK_K, NUM_WARPS, 16, true,  false>(out_dtype, grid, threads, stream, args, _persist); \
        else if (!_str &&  _fp16a) dispatch_outdtype<B_BATCH, BLOCK_M, BLOCK_K, NUM_WARPS, 16, false, true >(out_dtype, grid, threads, stream, args, _persist); \
        else                       dispatch_outdtype<B_BATCH, BLOCK_M, BLOCK_K, NUM_WARPS, 16, false, false>(out_dtype, grid, threads, stream, args, _persist); \
    } while (0)

    static const int kMaxBBatch = []() {
        const char* env = std::getenv("NVFP4R_GEMV_MAX_B_BATCH");
        if (!env) return 4;
        const int v = std::atoi(env);
        return (v == 1 || v == 2 || v == 4 || v == 8) ? v : 4;
    }();

    static const int kSegBytes = []() {
        const char* env = std::getenv("NVFP4R_GEMV_SEG_BYTES");
        if (!env) return 32;
        const int v = std::atoi(env);
        return (v == 16) ? 16 : 32;
    }();

    static const bool kUseSmma = []() {
        const char* env = std::getenv("NVFP4R_GEMV_USE_SMMA");
        if (!env) return false;
        return std::atoi(env) != 0;
    }();

    if (kUseSmma && L >= 1 && L <= 16 && M % 128 == 0 && (Kp * 2) % 128 == 0) {
        constexpr int SMMA_BM = 16;
        constexpr int SMMA_BN = 128;
        constexpr int SMMA_BK = 128;
        constexpr int SMMA_NW = 8;
        const dim3 grid(M / SMMA_BN, 1);
        const int  threads = SMMA_NW * kWarpSize;
        const size_t smem_bytes = (SMMA_BM * SMMA_BK + SMMA_BN * SMMA_BK)
                                  * sizeof(__half);
        if (out_dtype == at::kHalf) {
            mv_smma_kernel<SMMA_BM, SMMA_BN, SMMA_BK, SMMA_NW, __half>
                <<<grid, threads, smem_bytes, stream>>>(args);
        } else {
            mv_smma_kernel<SMMA_BM, SMMA_BN, SMMA_BK, SMMA_NW, __nv_bfloat16>
                <<<grid, threads, smem_bytes, stream>>>(args);
        }
        return C;
    }
#define LAUNCH_FOR_L(BLOCK_M, BLOCK_K, NUM_WARPS)                               \
    do {                                                                         \
        if      (L % 8 == 0 && kMaxBBatch >= 8) { LAUNCH(8, BLOCK_M, BLOCK_K, NUM_WARPS); }  \
        else if (L % 4 == 0 && kMaxBBatch >= 4) { LAUNCH(4, BLOCK_M, BLOCK_K, NUM_WARPS); }  \
        else if (L % 2 == 0 && kMaxBBatch >= 2) { LAUNCH(2, BLOCK_M, BLOCK_K, NUM_WARPS); }  \
        else                                    { LAUNCH(1, BLOCK_M, BLOCK_K, NUM_WARPS); }  \
    } while (0)

#define LAUNCH_FOR_L16(BLOCK_M, BLOCK_K, NUM_WARPS)                              \
    do {                                                                          \
        if      (L % 8 == 0 && kMaxBBatch >= 8) { LAUNCH16(8, BLOCK_M, BLOCK_K, NUM_WARPS); }  \
        else if (L % 4 == 0 && kMaxBBatch >= 4) { LAUNCH16(4, BLOCK_M, BLOCK_K, NUM_WARPS); }  \
        else if (L % 2 == 0 && kMaxBBatch >= 2) { LAUNCH16(2, BLOCK_M, BLOCK_K, NUM_WARPS); }  \
        else                                    { LAUNCH16(1, BLOCK_M, BLOCK_K, NUM_WARPS); }  \
    } while (0)

    if (Kp % 512 == 0 && M % 8 == 0) {
        LAUNCH_FOR_L(8, 512, 4);
    } else if (Kp % 256 == 0 && M % 8 == 0) {
        LAUNCH_FOR_L(8, 256, 2);
    } else if (Kp % 256 == 0 && M % 16 == 0) {
        LAUNCH_FOR_L(16, 256, 4);
    } else if (Kp % 128 == 0 && M % 16 == 0) {

        LAUNCH_FOR_L16(16, 128, 4);
    } else if (Kp % 128 == 0 && M % 8 == 0) {
        LAUNCH_FOR_L16(8, 128, 2);
    } else {
        TORCH_CHECK(false,
                    "gemv: no kernel tile fits (M=", M, ", K_packed=", Kp,
                    "). Need K_packed divisible by 128 and M divisible "
                    "by 8 or 16.");
    }

#undef LAUNCH_FOR_L
#undef LAUNCH

    return C;
}

}

at::Tensor gemv(at::Tensor A,
                at::Tensor B,
                at::Tensor C,
                at::Tensor SFA,
                at::Tensor SFB,
                double alpha) {
    return launch_mv(std::move(A), std::move(B), std::move(C),
                     std::move(SFA), std::move(SFB), alpha);
}

}
