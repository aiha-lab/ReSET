// Stub fallback for the GEMM op (GEMV-only builds).

#include "api.h"
#include <torch/extension.h>

namespace nvfp4r {

at::Tensor gemm(
    const at::Tensor& ,
    const at::Tensor& ,
    const at::Tensor& ,
    const at::Tensor& ,
          at::Tensor& C,
    double ) {
    TORCH_CHECK(false, "nvfp4r::gemm: not built in this configuration (GEMV-only build). "
                       "Set NVFP4R_ENABLE_GEMM=0 to suppress this code path.");
    return C;
}

}
