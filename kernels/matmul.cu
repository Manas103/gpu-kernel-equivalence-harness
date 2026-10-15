// Hand-written tiled GEMM: C = A @ B, A is (M,K), B is (K,N), C is (M,N) fp32.
// One templated kernel body, one conversion trait per input dtype, so the
// only thing that changes between fp16, bf16 and tf32 is how a loaded tile
// element gets turned into the float the tile multiply-accumulates in.
//
// TF32 is not a storage type: real tensor cores truncate an fp32 operand's
// mantissa to 10 bits before multiplying. This kernel does not use tensor
// cores at all (that would need wmma intrinsics, out of scope for "hand
// written"); it emulates the same truncation in plain fp32 arithmetic by
// masking the low 13 mantissa bits to zero before every multiply, which
// reproduces the arithmetic TF32 does, not the hardware path that does it.
//
// Compiled as a plain shared library via build.sh and called through
// ctypes (see harness/cuda_kernel.py), not through
// torch.utils.cpp_extension.load: an earlier version of this file used
// <torch/extension.h> and PYBIND11_MODULE, which JIT-compiled fine against
// an older torch/CUDA pairing but failed on this machine's CUDA 13.4 nvcc
// against this torch wheel's bundled ATen headers (a real "need 'typename'
// before a dependent decltype" error inside c10::List_inl.h, a template
// two-phase-lookup strictness mismatch between this nvcc and these
// headers, not a bug in this file). Dropping the torch C++ headers
// entirely avoids that incompatibility; see README Findings.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstring>
#include <cstdint>

#define TILE 16

enum class InputKind { FP16, BF16, TF32 };

__device__ __forceinline__ float load_fp16(const __half* p, int idx) {
    return __half2float(p[idx]);
}
__device__ __forceinline__ float load_bf16(const __nv_bfloat16* p, int idx) {
    return __bfloat162float(p[idx]);
}
__device__ __forceinline__ float truncate_to_tf32(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    bits &= 0xFFFFE000u; // keep sign(1) + exponent(8) + top 10 mantissa bits
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

template <typename StorageT, InputKind KIND>
__global__ void matmul_kernel(const StorageT* __restrict__ A,
                               const StorageT* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        float aVal = 0.0f;
        if (row < M && aCol < K) {
            int idx = row * K + aCol;
            if constexpr (KIND == InputKind::FP16) aVal = load_fp16(A, idx);
            else if constexpr (KIND == InputKind::BF16) aVal = load_bf16(A, idx);
            else aVal = truncate_to_tf32(A[idx]);
        }
        float bVal = 0.0f;
        if (bRow < K && col < N) {
            int idx = bRow * N + col;
            if constexpr (KIND == InputKind::FP16) bVal = load_fp16(B, idx);
            else if constexpr (KIND == InputKind::BF16) bVal = load_bf16(B, idx);
            else bVal = truncate_to_tf32(B[idx]);
        }
        As[threadIdx.y][threadIdx.x] = aVal;
        Bs[threadIdx.y][threadIdx.x] = bVal;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

extern "C" {

void launch_matmul_fp16(const void* A, const void* B, float* C, int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_kernel<__half, InputKind::FP16><<<grid, block>>>(
        reinterpret_cast<const __half*>(A), reinterpret_cast<const __half*>(B), C, M, N, K);
}

void launch_matmul_bf16(const void* A, const void* B, float* C, int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_kernel<__nv_bfloat16, InputKind::BF16><<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(A), reinterpret_cast<const __nv_bfloat16*>(B), C, M, N, K);
}

void launch_matmul_tf32(const void* A, const void* B, float* C, int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_kernel<float, InputKind::TF32><<<grid, block>>>(
        reinterpret_cast<const float*>(A), reinterpret_cast<const float*>(B), C, M, N, K);
}

int matmul_sync_and_get_last_error() {
    cudaDeviceSynchronize();
    return (int)cudaGetLastError();
}

} // extern "C"
