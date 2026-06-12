// NVFP4 Tensor-Core GEMM kernel (prefill / large-M path).

#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>

#include <type_traits>

namespace nvfp4r {
namespace gemm_kernel {

constexpr int kWarpSize = 32;
constexpr int kMmaK     = 64;

[[maybe_unused]] constexpr uint64_t kCacheNormal = 0x1000000000000000ULL;
constexpr uint64_t kCacheFirst  = 0x12F0000000000000ULL;
constexpr uint64_t kCacheLast   = 0x14F0000000000000ULL;

__device__ __forceinline__ constexpr uint64_t encode_desc(uint64_t addr) {
    return (addr & 0x3'FFFFULL) >> 4ULL;
}

__device__ __forceinline__ uint32_t elect_one() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred)
        : "r"(0xFFFFFFFFu));
    return pred;
}

__device__ __forceinline__ void mbar_init(int mbar_addr, int arrive_count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(mbar_addr), "r"(arrive_count));
}

__device__ __forceinline__ void mbar_wait(int mbar_addr, int phase) {
    constexpr uint32_t kSpinTicks = 0x989680u;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "MBAR_LOOP:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni MBAR_DONE;\n\t"
        "bra.uni MBAR_LOOP;\n\t"
        "MBAR_DONE:\n\t"
        "}"
        :: "r"(mbar_addr), "r"(phase), "r"(kSpinTicks));
}

__device__ __forceinline__ void tma_load_1d(
    int dst_smem, const void* src_gmem, int bytes, int mbar_addr, uint64_t cache) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1], %2, [%3], %4;"
        :: "r"(dst_smem), "l"(src_gmem), "r"(bytes), "r"(mbar_addr), "l"(cache));
}

__device__ __forceinline__ void tma_load_3d(
    int dst_smem, const void* tmap, int x, int y, int z,
    int mbar_addr, uint64_t cache) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes."
        "cta_group::1.L2::cache_hint "
        "[%0], [%1, {%2, %3, %4}], [%5], %6;"
        :: "r"(dst_smem), "l"(tmap), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr), "l"(cache)
        : "memory");
}

__device__ __forceinline__ void tcgen05_cp_block_scale(int taddr, uint64_t s_desc) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                 :: "r"(taddr), "l"(s_desc));
}

__device__ __forceinline__ void tcgen05_mma_block_scale(
    uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_tmem, int sfb_tmem, int enable_input_d) {
    constexpr int kDTmem = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}"
        :: "r"(kDTmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
           "r"(sfa_tmem), "r"(sfb_tmem), "r"(enable_input_d));
}

struct LdShape {
    static constexpr char k32x32b[]  = ".32x32b";
    static constexpr char k16x128b[] = ".16x128b";
    static constexpr char k16x256b[] = ".16x256b";
};

struct LdNum {
    static constexpr char x4[]   = ".x4";
    static constexpr char x8[]   = ".x8";
    static constexpr char x16[]  = ".x16";
    static constexpr char x32[]  = ".x32";
    static constexpr char x64[]  = ".x64";
    static constexpr char x128[] = ".x128";
};

template <const char* kShape, const char* kNum>
__device__ __forceinline__ void tcgen05_ld_16regs(float* dst, int row, int col) {
    asm volatile("tcgen05.ld.sync.aligned%17%18.b32 "
                 "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
                 "  %8,  %9, %10, %11, %12, %13, %14, %15}, [%16];"
                 : "=f"(dst[ 0]), "=f"(dst[ 1]), "=f"(dst[ 2]), "=f"(dst[ 3]),
                   "=f"(dst[ 4]), "=f"(dst[ 5]), "=f"(dst[ 6]), "=f"(dst[ 7]),
                   "=f"(dst[ 8]), "=f"(dst[ 9]), "=f"(dst[10]), "=f"(dst[11]),
                   "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
                 : "r"((row << 16) | col), "C"(kShape), "C"(kNum));
}

template <const char* kShape, const char* kNum>
__device__ __forceinline__ void tcgen05_ld_32regs(float* dst, int row, int col) {
    asm volatile("tcgen05.ld.sync.aligned%33%34.b32 "
                 "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
                 "  %8,  %9, %10, %11, %12, %13, %14, %15, "
                 " %16, %17, %18, %19, %20, %21, %22, %23, "
                 " %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                 : "=f"(dst[ 0]), "=f"(dst[ 1]), "=f"(dst[ 2]), "=f"(dst[ 3]),
                   "=f"(dst[ 4]), "=f"(dst[ 5]), "=f"(dst[ 6]), "=f"(dst[ 7]),
                   "=f"(dst[ 8]), "=f"(dst[ 9]), "=f"(dst[10]), "=f"(dst[11]),
                   "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15]),
                   "=f"(dst[16]), "=f"(dst[17]), "=f"(dst[18]), "=f"(dst[19]),
                   "=f"(dst[20]), "=f"(dst[21]), "=f"(dst[22]), "=f"(dst[23]),
                   "=f"(dst[24]), "=f"(dst[25]), "=f"(dst[26]), "=f"(dst[27]),
                   "=f"(dst[28]), "=f"(dst[29]), "=f"(dst[30]), "=f"(dst[31])
                 : "r"((row << 16) | col), "C"(kShape), "C"(kNum));
}

template <const char* kShape, const char* kNum>
__device__ __forceinline__ void tcgen05_ld_64regs(float* dst, int row, int col) {
    asm volatile("tcgen05.ld.sync.aligned%65%66.b32 "
                 "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
                 "  %8,  %9, %10, %11, %12, %13, %14, %15, "
                 " %16, %17, %18, %19, %20, %21, %22, %23, "
                 " %24, %25, %26, %27, %28, %29, %30, %31, "
                 " %32, %33, %34, %35, %36, %37, %38, %39, "
                 " %40, %41, %42, %43, %44, %45, %46, %47, "
                 " %48, %49, %50, %51, %52, %53, %54, %55, "
                 " %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                 : "=f"(dst[ 0]), "=f"(dst[ 1]), "=f"(dst[ 2]), "=f"(dst[ 3]),
                   "=f"(dst[ 4]), "=f"(dst[ 5]), "=f"(dst[ 6]), "=f"(dst[ 7]),
                   "=f"(dst[ 8]), "=f"(dst[ 9]), "=f"(dst[10]), "=f"(dst[11]),
                   "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15]),
                   "=f"(dst[16]), "=f"(dst[17]), "=f"(dst[18]), "=f"(dst[19]),
                   "=f"(dst[20]), "=f"(dst[21]), "=f"(dst[22]), "=f"(dst[23]),
                   "=f"(dst[24]), "=f"(dst[25]), "=f"(dst[26]), "=f"(dst[27]),
                   "=f"(dst[28]), "=f"(dst[29]), "=f"(dst[30]), "=f"(dst[31]),
                   "=f"(dst[32]), "=f"(dst[33]), "=f"(dst[34]), "=f"(dst[35]),
                   "=f"(dst[36]), "=f"(dst[37]), "=f"(dst[38]), "=f"(dst[39]),
                   "=f"(dst[40]), "=f"(dst[41]), "=f"(dst[42]), "=f"(dst[43]),
                   "=f"(dst[44]), "=f"(dst[45]), "=f"(dst[46]), "=f"(dst[47]),
                   "=f"(dst[48]), "=f"(dst[49]), "=f"(dst[50]), "=f"(dst[51]),
                   "=f"(dst[52]), "=f"(dst[53]), "=f"(dst[54]), "=f"(dst[55]),
                   "=f"(dst[56]), "=f"(dst[57]), "=f"(dst[58]), "=f"(dst[59]),
                   "=f"(dst[60]), "=f"(dst[61]), "=f"(dst[62]), "=f"(dst[63])
                 : "r"((row << 16) | col), "C"(kShape), "C"(kNum));
}

template <const char* kShape, const char* kNum>
__device__ __forceinline__ void tcgen05_ld_128regs(float* dst, int row, int col) {
    asm volatile("tcgen05.ld.sync.aligned%129%130.b32 "
                 "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
                 "  %8,  %9, %10, %11, %12, %13, %14, %15, "
                 " %16, %17, %18, %19, %20, %21, %22, %23, "
                 " %24, %25, %26, %27, %28, %29, %30, %31, "
                 " %32, %33, %34, %35, %36, %37, %38, %39, "
                 " %40, %41, %42, %43, %44, %45, %46, %47, "
                 " %48, %49, %50, %51, %52, %53, %54, %55, "
                 " %56, %57, %58, %59, %60, %61, %62, %63, "
                 " %64, %65, %66, %67, %68, %69, %70, %71, "
                 " %72, %73, %74, %75, %76, %77, %78, %79, "
                 " %80, %81, %82, %83, %84, %85, %86, %87, "
                 " %88, %89, %90, %91, %92, %93, %94, %95, "
                 " %96, %97, %98, %99,%100,%101,%102,%103, "
                 "%104,%105,%106,%107,%108,%109,%110,%111, "
                 "%112,%113,%114,%115,%116,%117,%118,%119, "
                 "%120,%121,%122,%123,%124,%125,%126,%127}, [%128];"
                 : "=f"(dst[  0]),"=f"(dst[  1]),"=f"(dst[  2]),"=f"(dst[  3]),
                   "=f"(dst[  4]),"=f"(dst[  5]),"=f"(dst[  6]),"=f"(dst[  7]),
                   "=f"(dst[  8]),"=f"(dst[  9]),"=f"(dst[ 10]),"=f"(dst[ 11]),
                   "=f"(dst[ 12]),"=f"(dst[ 13]),"=f"(dst[ 14]),"=f"(dst[ 15]),
                   "=f"(dst[ 16]),"=f"(dst[ 17]),"=f"(dst[ 18]),"=f"(dst[ 19]),
                   "=f"(dst[ 20]),"=f"(dst[ 21]),"=f"(dst[ 22]),"=f"(dst[ 23]),
                   "=f"(dst[ 24]),"=f"(dst[ 25]),"=f"(dst[ 26]),"=f"(dst[ 27]),
                   "=f"(dst[ 28]),"=f"(dst[ 29]),"=f"(dst[ 30]),"=f"(dst[ 31]),
                   "=f"(dst[ 32]),"=f"(dst[ 33]),"=f"(dst[ 34]),"=f"(dst[ 35]),
                   "=f"(dst[ 36]),"=f"(dst[ 37]),"=f"(dst[ 38]),"=f"(dst[ 39]),
                   "=f"(dst[ 40]),"=f"(dst[ 41]),"=f"(dst[ 42]),"=f"(dst[ 43]),
                   "=f"(dst[ 44]),"=f"(dst[ 45]),"=f"(dst[ 46]),"=f"(dst[ 47]),
                   "=f"(dst[ 48]),"=f"(dst[ 49]),"=f"(dst[ 50]),"=f"(dst[ 51]),
                   "=f"(dst[ 52]),"=f"(dst[ 53]),"=f"(dst[ 54]),"=f"(dst[ 55]),
                   "=f"(dst[ 56]),"=f"(dst[ 57]),"=f"(dst[ 58]),"=f"(dst[ 59]),
                   "=f"(dst[ 60]),"=f"(dst[ 61]),"=f"(dst[ 62]),"=f"(dst[ 63]),
                   "=f"(dst[ 64]),"=f"(dst[ 65]),"=f"(dst[ 66]),"=f"(dst[ 67]),
                   "=f"(dst[ 68]),"=f"(dst[ 69]),"=f"(dst[ 70]),"=f"(dst[ 71]),
                   "=f"(dst[ 72]),"=f"(dst[ 73]),"=f"(dst[ 74]),"=f"(dst[ 75]),
                   "=f"(dst[ 76]),"=f"(dst[ 77]),"=f"(dst[ 78]),"=f"(dst[ 79]),
                   "=f"(dst[ 80]),"=f"(dst[ 81]),"=f"(dst[ 82]),"=f"(dst[ 83]),
                   "=f"(dst[ 84]),"=f"(dst[ 85]),"=f"(dst[ 86]),"=f"(dst[ 87]),
                   "=f"(dst[ 88]),"=f"(dst[ 89]),"=f"(dst[ 90]),"=f"(dst[ 91]),
                   "=f"(dst[ 92]),"=f"(dst[ 93]),"=f"(dst[ 94]),"=f"(dst[ 95]),
                   "=f"(dst[ 96]),"=f"(dst[ 97]),"=f"(dst[ 98]),"=f"(dst[ 99]),
                   "=f"(dst[100]),"=f"(dst[101]),"=f"(dst[102]),"=f"(dst[103]),
                   "=f"(dst[104]),"=f"(dst[105]),"=f"(dst[106]),"=f"(dst[107]),
                   "=f"(dst[108]),"=f"(dst[109]),"=f"(dst[110]),"=f"(dst[111]),
                   "=f"(dst[112]),"=f"(dst[113]),"=f"(dst[114]),"=f"(dst[115]),
                   "=f"(dst[116]),"=f"(dst[117]),"=f"(dst[118]),"=f"(dst[119]),
                   "=f"(dst[120]),"=f"(dst[121]),"=f"(dst[122]),"=f"(dst[123]),
                   "=f"(dst[124]),"=f"(dst[125]),"=f"(dst[126]),"=f"(dst[127])
                 : "r"((row << 16) | col), "C"(kShape), "C"(kNum));
}

__device__ __forceinline__ void tcgen05_ld_32x32bx32 (float* d, int r, int c) { tcgen05_ld_32regs <LdShape::k32x32b , LdNum::x32 >(d, r, c); }
__device__ __forceinline__ void tcgen05_ld_32x32bx64 (float* d, int r, int c) { tcgen05_ld_64regs <LdShape::k32x32b , LdNum::x64 >(d, r, c); }
__device__ __forceinline__ void tcgen05_ld_32x32bx128(float* d, int r, int c) { tcgen05_ld_128regs<LdShape::k32x32b , LdNum::x128>(d, r, c); }

__device__ __forceinline__ void tcgen05_ld_16x128bx8 (float* d, int r, int c) { tcgen05_ld_16regs <LdShape::k16x128b, LdNum::x8  >(d, r, c); }
__device__ __forceinline__ void tcgen05_ld_16x128bx16(float* d, int r, int c) { tcgen05_ld_32regs <LdShape::k16x128b, LdNum::x16 >(d, r, c); }
__device__ __forceinline__ void tcgen05_ld_16x128bx32(float* d, int r, int c) { tcgen05_ld_64regs <LdShape::k16x128b, LdNum::x32 >(d, r, c); }

__device__ __forceinline__ void tcgen05_ld_16x256bx4 (float* d, int r, int c) { tcgen05_ld_16regs <LdShape::k16x256b, LdNum::x4  >(d, r, c); }
__device__ __forceinline__ void tcgen05_ld_16x256bx8 (float* d, int r, int c) { tcgen05_ld_32regs <LdShape::k16x256b, LdNum::x8  >(d, r, c); }
__device__ __forceinline__ void tcgen05_ld_16x256bx16(float* d, int r, int c) { tcgen05_ld_64regs <LdShape::k16x256b, LdNum::x16 >(d, r, c); }

inline void check_cu(CUresult err) {
    if (err == CUDA_SUCCESS) return;
    const char* msg = nullptr;
    if (cuGetErrorString(err, &msg) != CUDA_SUCCESS) msg = "<no error string>";
    TORCH_CHECK(false, "cuTensorMapEncodeTiled error: ", msg);
}

inline void check_cuda(cudaError_t err) {
    if (err == cudaSuccess) return;
    TORCH_CHECK(false, cudaGetErrorString(err));
}

void encode_AB_tmap(
    CUtensorMap* tmap,
    const char*  ptr,
    uint64_t     gmem_h,  uint64_t gmem_w,
    uint32_t     smem_h,  uint32_t smem_w)
{
    constexpr uint32_t rank = 3;
    uint64_t global_dim[rank]      = {256, gmem_h, gmem_w / 256};
    uint64_t global_strides[rank-1]= {gmem_w / 2, 128};
    uint32_t box_dim[rank]         = {256, smem_h, smem_w / 256};
    uint32_t element_strides[rank] = {1, 1, 1};

    auto err = cuTensorMapEncodeTiled(
        tmap,
        CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        rank,
        const_cast<void*>(reinterpret_cast<const void*>(ptr)),
        global_dim,
        global_strides,
        box_dim,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    check_cu(err);
}

template <typename OutT>
__device__ __forceinline__
void store_pair_scaled(OutT* ptr, float a, float b, float alpha) {
    if constexpr (std::is_same_v<OutT, __half>) {
        reinterpret_cast<half2*>(ptr)[0] = __float22half2_rn({a * alpha, b * alpha});
    } else {
        reinterpret_cast<__nv_bfloat162*>(ptr)[0] =
            __float22bfloat162_rn({a * alpha, b * alpha});
    }
}

template <typename OutT>
__device__ __forceinline__
void store_one_scaled(OutT* ptr, float v, float alpha) {
    if constexpr (std::is_same_v<OutT, __half>) {
        *ptr = __float2half(v * alpha);
    } else {
        *ptr = __float2bfloat16(v * alpha);
    }
}

template <typename OutT,
          int K,
          int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K,
          bool kCNMajor,
          int  kNumStages>
__global__ __launch_bounds__(BLOCK_M + 2 * kWarpSize)
void gemm_main_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const char* SFA_ptr,
    const char* SFB_ptr,
    OutT*       C_ptr,
    int M, int N,
    float    alpha)
{
    const int thr = threadIdx.x;
    const int cta = blockIdx.x;

    const int lane = thr % kWarpSize;
    const int wid  = thr / kWarpSize;

    const int nblocks_n = N / BLOCK_N;
    const int cta_m = cta / nblocks_n;
    const int cta_n = cta % nblocks_n;

    const int m_off = cta_m * BLOCK_M;
    const int n_off = cta_n * BLOCK_N;

    constexpr int kNumWarps = BLOCK_M / kWarpSize + 2;

    extern __shared__ __align__(1024) char smem_raw[];
    const int smem_base = static_cast<int>(__cvta_generic_to_shared(smem_raw));
    constexpr int kASize    = BLOCK_M * BLOCK_K / 2;
    constexpr int kBSize    = BLOCK_N * BLOCK_K / 2;
    constexpr int kSFASize  = 128 * BLOCK_K / 16;
    constexpr int kSFBSize  = 128 * BLOCK_K / 16;
    constexpr int kStageSize = kASize + kBSize + kSFASize + kSFBSize;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ int64_t mbar_storage[kNumStages * 2 + 1];
    const int mbar_tma  = static_cast<int>(__cvta_generic_to_shared(mbar_storage));
    const int mbar_mma  = mbar_tma + kNumStages * 8;
    const int mbar_done = mbar_mma + kNumStages * 8;

    constexpr int kSFATmem = BLOCK_N;
    constexpr int kSFBTmem = kSFATmem + 4 * (BLOCK_K / kMmaK);

    if (wid == 0 && elect_one()) {
        for (int i = 0; i < kNumStages * 2 + 1; ++i)
            mbar_init(mbar_tma + i * 8, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (wid == 1) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(smem_base), "r"(BLOCK_N * 2));
    }
    __syncthreads();

    const int n_iters = K / BLOCK_K;

    if (wid == kNumWarps - 2 && elect_one()) {
        const uint64_t cache_a = (M > N) ? kCacheFirst : kCacheLast;
        const uint64_t cache_b = (M > N) ? kCacheLast  : kCacheFirst;

        auto issue_one_stage = [&](int kit, int stage) {
            const int mbar = mbar_tma + stage * 8;
            const int sm_a = smem_base + stage * kStageSize;
            const int sm_b = sm_a + kASize;
            const int sm_sa = sm_b + kBSize;
            const int sm_sb = sm_sa + kSFASize;

            const int k_off = kit * BLOCK_K;
            tma_load_3d(sm_a, &A_tmap, 0, m_off, k_off / 256, mbar, cache_a);
            tma_load_3d(sm_b, &B_tmap, 0, n_off, k_off / 256, mbar, cache_b);

            const int rest_k = K / 16 / 4;
            const char* sfa_src = SFA_ptr + ((m_off / 128) * rest_k + k_off / (16 * 4)) * 512;
            const char* sfb_src = SFB_ptr + ((n_off / 128) * rest_k + k_off / (16 * 4)) * 512;
            tma_load_1d(sm_sa, sfa_src, kSFASize, mbar, cache_a);
            tma_load_1d(sm_sb, sfb_src, kSFBSize, mbar, cache_b);

            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(mbar), "r"(kStageSize) : "memory");
        };

        for (int kit = 0; kit < kNumStages; ++kit)
            issue_one_stage(kit, kit);

        for (int kit = kNumStages; kit < n_iters; ++kit) {
            const int stage = kit % kNumStages;
            const int mma_phase = (kit / kNumStages - 1) % 2;
            mbar_wait(mbar_mma + stage * 8, mma_phase);

            issue_one_stage(kit, stage);
        }
    }

    else if (wid == kNumWarps - 1 && elect_one()) {
        constexpr uint32_t kIDesc = (1U << 7U)
                                  | (1U << 10U)
                                  | ((uint32_t)BLOCK_N >> 3U << 17U)
                                  | ((uint32_t)128 >> 7U << 27U);

        for (int kit = 0; kit < n_iters; ++kit) {
            const int stage = kit % kNumStages;
            const int tma_phase = (kit / kNumStages) % 2;
            mbar_wait(mbar_tma + stage * 8, tma_phase);

            const int sm_a  = smem_base + stage * kStageSize;
            const int sm_b  = sm_a + kASize;
            const int sm_sa = sm_b + kBSize;
            const int sm_sb = sm_sa + kSFASize;

            auto desc_AB = [](int addr) -> uint64_t {
                constexpr int kSBO = 8 * 128;
                return encode_desc(addr) | (encode_desc(kSBO) << 32ULL)
                       | (1ULL << 46ULL) | (2ULL << 61ULL);
            };

            auto desc_SF = [](int addr) -> uint64_t {
                constexpr int kSBO = 8 * 16;
                return encode_desc(addr) | (encode_desc(kSBO) << 32ULL)
                       | (1ULL << 46ULL);
            };

            constexpr uint64_t kSFBase = desc_SF(0);
            const uint64_t sfa_desc_base = kSFBase + (uint64_t(sm_sa) >> 4ULL);
            const uint64_t sfb_desc_base = kSFBase + (uint64_t(sm_sb) >> 4ULL);

            for (int k = 0; k < BLOCK_K / kMmaK; ++k) {
                const uint64_t sfa_d = sfa_desc_base + uint64_t(k) * (512ULL >> 4ULL);
                const uint64_t sfb_d = sfb_desc_base + uint64_t(k) * (512ULL >> 4ULL);
                tcgen05_cp_block_scale(kSFATmem + k * 4, sfa_d);
                tcgen05_cp_block_scale(kSFBTmem + k * 4, sfb_d);
            }

            for (int k1 = 0; k1 < BLOCK_K / 256; ++k1) {
                for (int k2 = 0; k2 < 256 / kMmaK; ++k2) {
                    const uint64_t a_desc = desc_AB(sm_a + k1 * BLOCK_M * 128 + k2 * 32);
                    const uint64_t b_desc = desc_AB(sm_b + k1 * BLOCK_N * 128 + k2 * 32);

                    const int k_sf = k1 * 4 + k2;
                    const int sfa_tmem = kSFATmem + k_sf * 4
                                       + (cta_m % (128 / BLOCK_M)) * (BLOCK_M / 32);
                    const int sfb_tmem = kSFBTmem + k_sf * 4
                                       + (cta_n % (128 / BLOCK_N)) * (BLOCK_N / 32);

                    const int enable_d = (k1 == 0 && k2 == 0) ? kit : 1;
                    tcgen05_mma_block_scale(a_desc, b_desc, kIDesc,
                                            sfa_tmem, sfb_tmem, enable_d);
                }
            }

            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                :: "r"(mbar_mma + stage * 8) : "memory");
        }

        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
            :: "r"(mbar_done) : "memory");
    }

    else if (thr < BLOCK_M) {
        mbar_wait(mbar_done, 0);
        asm volatile("tcgen05.fence::after_thread_sync;");


        if constexpr (kCNMajor) {

            for (int m = 0; m < 32 / 16; ++m) {
                float buf[BLOCK_N / 2];
                if constexpr (BLOCK_N == 128) tcgen05_ld_16x256bx16(buf, wid * 32 + m * 16, 0);
                if constexpr (BLOCK_N ==  64) tcgen05_ld_16x256bx8 (buf, wid * 32 + m * 16, 0);
                if constexpr (BLOCK_N ==  32) tcgen05_ld_16x256bx4 (buf, wid * 32 + m * 16, 0);
                asm volatile("tcgen05.wait::ld.sync.aligned;");

                for (int i = 0; i < BLOCK_N / 8; ++i) {
                    const int row_id = m_off + wid * 32 + m * 16 + lane / 4;
                    const int col_id = n_off + i * 8 + (lane % 4) * 2;

                    store_pair_scaled<OutT>(
                        C_ptr + (row_id + 0) * N + col_id,
                        buf[i * 4 + 0], buf[i * 4 + 1], alpha);
                    store_pair_scaled<OutT>(
                        C_ptr + (row_id + 8) * N + col_id,
                        buf[i * 4 + 2], buf[i * 4 + 3], alpha);
                }
            }
        } else {

            constexpr int kWidth = (BLOCK_N < 64) ? BLOCK_N : 64;
            for (int n = 0; n < BLOCK_N / kWidth; ++n) {
                float buf[kWidth];
                if constexpr (kWidth == 128) tcgen05_ld_32x32bx128(buf, wid * 32, n * kWidth);
                if constexpr (kWidth ==  64) tcgen05_ld_32x32bx64 (buf, wid * 32, n * kWidth);
                if constexpr (kWidth ==  32) tcgen05_ld_32x32bx32 (buf, wid * 32, n * kWidth);
                asm volatile("tcgen05.wait::ld.sync.aligned;");

                for (int i = 0; i < kWidth; ++i)
                    store_one_scaled<OutT>(
                        C_ptr + (n_off + n * kWidth + i) * M + (m_off + thr),
                        buf[i], alpha);
            }
        }

        asm volatile("bar.sync 1, %0;" :: "r"(BLOCK_M) : "memory");
        if (wid == 0)
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                         :: "r"(0), "r"(BLOCK_N * 2));

    }

}

template <typename OutT,
          int K,
          int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K,
          bool kSwapAB,
          bool kCNMajor,
          int  kNumStages>
at::Tensor launch_prefill(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& SFA,
    const at::Tensor& SFB,
          at::Tensor& C,
    float    alpha)
{
    static_assert(BLOCK_K % 256 == 0, "BLOCK_K must be a multiple of 256");

    const int M = A.size(0);
    const int N = B.size(0);

    const char* A_p   = reinterpret_cast<const char*>(A.data_ptr());
    const char* B_p   = reinterpret_cast<const char*>(B.data_ptr());
    const char* SFA_p = reinterpret_cast<const char*>(SFA.data_ptr());
    const char* SFB_p = reinterpret_cast<const char*>(SFB.data_ptr());
    OutT*       C_p   = reinterpret_cast<OutT*>(C.data_ptr());

    int eff_M = M;
    int eff_N = N;
    if constexpr (kSwapAB) {
        std::swap(A_p, B_p);
        std::swap(SFA_p, SFB_p);
        std::swap(eff_M, eff_N);
    }

    CUtensorMap A_tmap, B_tmap;
    encode_AB_tmap(&A_tmap, A_p, eff_M, K, BLOCK_M, BLOCK_K);
    encode_AB_tmap(&B_tmap, B_p, eff_N, K, BLOCK_N, BLOCK_K);

    const int grid    = (eff_M / BLOCK_M) * (eff_N / BLOCK_N);
    const int cta_dim = BLOCK_M + 2 * kWarpSize;
    const int ab_sz   = (BLOCK_M + BLOCK_N) * (BLOCK_K / 2);
    const int sf_sz   = 128 * (BLOCK_K / 16) * 2;
    const int smem_sz = (ab_sz + sf_sz) * kNumStages;

    auto kernel_fn = gemm_main_kernel<
        OutT, K, BLOCK_M, BLOCK_N, BLOCK_K,
        kCNMajor != kSwapAB, kNumStages>;

    if (smem_sz > 48'000) {
        static const bool _smem_set = [&]() {
            cudaFuncSetAttribute(kernel_fn,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem_sz);
            return true;
        }();
        (void)_smem_set;
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    kernel_fn<<<grid, cta_dim, smem_sz, stream>>>(
        A_tmap, B_tmap, SFA_p, SFB_p, C_p, eff_M, eff_N,
        alpha);

    return kCNMajor ? C : C.view({N, M, 1}).transpose(0, 1);
}

template <int Kv, int BlkM, int BlkN, int BlkK, int Stages>
struct PrefillTile {
    static constexpr int  kK = Kv;
    static constexpr int  kM = BlkM;
    static constexpr int  kN = BlkN;
    static constexpr int  kBlkK = BlkK;
    static constexpr int  kStages = Stages;
    static constexpr bool kSwapAB = true;
    static constexpr bool kCNMajor = false;
};

template <typename Tile, typename OutT>
inline at::Tensor run_tile_typed(
    const at::Tensor& A, const at::Tensor& B,
    const at::Tensor& SFA, const at::Tensor& SFB,
          at::Tensor& C, float alpha)
{
    return launch_prefill<
        OutT, Tile::kK, Tile::kM, Tile::kN, Tile::kBlkK,
        Tile::kSwapAB, Tile::kCNMajor, Tile::kStages>(
            A, B, SFA, SFB, C, alpha);
}

template <typename Tile>
inline at::Tensor run_tile(
    const at::Tensor& A, const at::Tensor& B,
    const at::Tensor& SFA, const at::Tensor& SFB,
          at::Tensor& C, float alpha)
{
    if (C.scalar_type() == at::kBFloat16) {
        return run_tile_typed<Tile, __nv_bfloat16>(A, B, SFA, SFB, C, alpha);
    } else {
        return run_tile_typed<Tile, __half>(A, B, SFA, SFB, C, alpha);
    }
}

at::Tensor gemm(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& SFA,
    const at::Tensor& SFB,
          at::Tensor& C,
    double alpha)
{
    const int K_full = A.size(1) * 2;
    const float a = static_cast<float>(alpha);

    switch (K_full) {

        case 16384: C = run_tile<PrefillTile<16384, 128, 64, 256, 8>>(A, B, SFA, SFB, C, a); break;
        case  7168: C = run_tile<PrefillTile< 7168,  64, 64, 512, 5>>(A, B, SFA, SFB, C, a); break;
        case  2048: C = run_tile<PrefillTile< 2048, 128, 64, 256, 8>>(A, B, SFA, SFB, C, a); break;
        case   256: C = run_tile<PrefillTile<  256, 128, 64, 256, 6>>(A, B, SFA, SFB, C, a); break;
        case   512: C = run_tile<PrefillTile<  512, 128, 64, 256, 6>>(A, B, SFA, SFB, C, a); break;
        case  1536: C = run_tile<PrefillTile< 1536, 128, 64, 256, 6>>(A, B, SFA, SFB, C, a); break;
        case  2304: C = run_tile<PrefillTile< 2304, 128, 64, 256, 6>>(A, B, SFA, SFB, C, a); break;

        case  3584: C = run_tile<PrefillTile< 3584, 128, 64, 256, 8>>(A, B, SFA, SFB, C, a); break;
        case  4096: C = run_tile<PrefillTile< 4096, 128, 64, 256, 8>>(A, B, SFA, SFB, C, a); break;
        case  5120: C = run_tile<PrefillTile< 5120, 128, 64, 256, 8>>(A, B, SFA, SFB, C, a); break;
        case  8192: C = run_tile<PrefillTile< 8192, 128, 64, 256, 8>>(A, B, SFA, SFB, C, a); break;
        case 12288: C = run_tile<PrefillTile<12288, 128, 64, 256, 8>>(A, B, SFA, SFB, C, a); break;
        case 13824: C = run_tile<PrefillTile<13824, 128, 64, 256, 8>>(A, B, SFA, SFB, C, a); break;
        case 14336: C = run_tile<PrefillTile<14336, 128, 64, 256, 8>>(A, B, SFA, SFB, C, a); break;
        case 17408: C = run_tile<PrefillTile<17408, 128, 64, 256, 8>>(A, B, SFA, SFB, C, a); break;
        case 18944: C = run_tile<PrefillTile<18944, 128, 64, 256, 8>>(A, B, SFA, SFB, C, a); break;
        case 25600: C = run_tile<PrefillTile<25600, 128, 64, 256, 8>>(A, B, SFA, SFB, C, a); break;

        default:
            TORCH_CHECK(false,
                "nvfp4r::gemm: unsupported K=", K_full,
                " (M=", A.size(0), ", N=", B.size(0),
                "). Add an entry to the dispatch in cuda/gemm/gemm_nvfp4.cu.");
    }
    return C;
}

}

at::Tensor gemm(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& SFA,
    const at::Tensor& SFB,
          at::Tensor& C,
    double alpha) {
    return gemm_kernel::gemm(A, B, SFA, SFB, C, alpha);
}

}
