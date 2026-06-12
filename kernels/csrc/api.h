// Public C++ API for the nvfp4r kernel namespace.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include <ATen/core/Tensor.h>

namespace nvfp4r {

at::Tensor gemv(
    at::Tensor A,
    at::Tensor B,
    at::Tensor C,
    at::Tensor SFA,
    at::Tensor SFB,
    double alpha = 1.0);

at::Tensor gemm(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& SFA,
    const at::Tensor& SFB,
          at::Tensor& C,
    double alpha = 1.0);

}
