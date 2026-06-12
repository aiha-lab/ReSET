// PyTorch / TORCH_LIBRARY bindings for nvfp4r ops.

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "api.h"

namespace nvfp4r {
namespace {

inline bool is_fp4_byte_dtype(at::ScalarType t) {
    return t == at::kByte
#if defined(USE_FP4_DTYPES)
        || t == at::kFloat4_e2m1fn_x2
#endif
        ;
}

inline bool is_fp8_scale_dtype(at::ScalarType t) {
    return t == at::kByte || t == at::kFloat8_e4m3fn;
}

void gemv_op(
    const at::Tensor& weight,
    const at::Tensor& weight_scale,
    const at::Tensor& x,
    const at::Tensor& x_scale,
    at::Tensor out,
    double alpha) {
    TORCH_CHECK(weight.is_cuda() && weight_scale.is_cuda() &&
                    x.is_cuda() && x_scale.is_cuda(),
                "gemv: all inputs must be CUDA");

    TORCH_CHECK(is_fp4_byte_dtype(weight.scalar_type()),
                "gemv: weight must be uint8 / fp4_e2m1fn_x2");
    TORCH_CHECK(is_fp8_scale_dtype(weight_scale.scalar_type()),
                "gemv: weight_scale must be uint8 / float8_e4m3fn");
    TORCH_CHECK(is_fp4_byte_dtype(x.scalar_type()),
                "gemv: x must be uint8 / fp4_e2m1fn_x2");
    TORCH_CHECK(is_fp8_scale_dtype(x_scale.scalar_type()),
                "gemv: x_scale must be uint8 / float8_e4m3fn");

    TORCH_CHECK(weight.dim() == 3 && weight_scale.dim() == 3 &&
                    x.dim() == 3 && x_scale.dim() == 3,
                "gemv: inputs must be 3D [L, M_or_N, K/2_or_K/16]");

    TORCH_CHECK(x.is_contiguous() && x_scale.is_contiguous(),
                "gemv: x / x_scale must be contiguous");

    const int64_t L = weight.size(0);
    const int64_t M = weight.size(1);
    const int64_t K_half = weight.size(2);
    const int64_t K = K_half * 2;

    TORCH_CHECK(weight_scale.size(0) == L && weight_scale.size(1) == M &&
                    weight_scale.size(2) * 16 == K,
                "weight_scale shape mismatch with weight");
    TORCH_CHECK(x.size(0) == L && x.size(2) == K_half,
                "x.K mismatch with weight.K");
    TORCH_CHECK(x_scale.size(0) == L && x_scale.size(1) == x.size(1) &&
                    x_scale.size(2) * 16 == K,
                "x_scale shape mismatch with x");

    at::Tensor y = out;
    TORCH_CHECK(y.is_cuda() && y.is_contiguous(), "out must be CUDA contiguous");
    TORCH_CHECK(y.dtype() == at::kHalf || y.dtype() == at::kBFloat16,
                "out must be float16 or bfloat16");
    TORCH_CHECK(y.dim() == 2 && y.size(0) == L && y.size(1) == M,
                "out must be [L, M]");

    const c10::cuda::OptionalCUDAGuard device_guard(weight.device());

    auto A_view  = weight.permute({1, 2, 0});
    auto SFA_v   = weight_scale.permute({1, 2, 0});
    auto B_view  = x.permute({1, 2, 0});
    auto SFB_v   = x_scale.permute({1, 2, 0});
    auto y_view  = y.transpose(0, 1).unsqueeze(1);

    nvfp4r::gemv(A_view, B_view, y_view, SFA_v, SFB_v, alpha);
}

void gemm_op(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& SFA,
    const at::Tensor& SFB,
    at::Tensor C,
    double alpha) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && SFA.is_cuda() && SFB.is_cuda() && C.is_cuda(),
                "gemm: all tensors must be CUDA");
    TORCH_CHECK(is_fp4_byte_dtype(A.scalar_type()) && is_fp4_byte_dtype(B.scalar_type()),
                "gemm: A/B must be uint8 (FP4 packed) or fp4_e2m1fn_x2");
    TORCH_CHECK(is_fp8_scale_dtype(SFA.scalar_type()) && is_fp8_scale_dtype(SFB.scalar_type()),
                "gemm: SFA/SFB must be uint8 or float8_e4m3fn");
    TORCH_CHECK(C.dtype() == at::kHalf || C.dtype() == at::kBFloat16,
                "gemm: C must be float16 or bfloat16");
    const c10::cuda::OptionalCUDAGuard device_guard(A.device());
    nvfp4r::gemm(A, B, SFA, SFB, C, alpha);
}

}
}

TORCH_LIBRARY(nvfp4r, m) {

    m.def(
        "gemv(Tensor weight, Tensor weight_scale, Tensor x, Tensor x_scale, "
        "Tensor(a!) out, float alpha=1.0) -> ()");
    m.def(
        "gemm(Tensor A, Tensor B, Tensor SFA, Tensor SFB, "
        "Tensor(a!) C, float alpha=1.0) -> ()");
}

TORCH_LIBRARY_IMPL(nvfp4r, CUDA, m) {
    m.impl("gemv", &nvfp4r::gemv_op);
    m.impl("gemm", &nvfp4r::gemm_op);
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "nvfp4r CUDA kernels (registered under torch.ops.nvfp4r)";
}
