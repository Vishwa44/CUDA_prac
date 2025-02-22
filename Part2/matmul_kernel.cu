#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void matmul_kernel(float* output, float* a, float* b, int M) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < M && row < M) {
        float acc = 0.0f;
        for (unsigned int i = 0; i < M; ++i){
            acc += a[row*M + i] * b[i*M + col];
        }
        output[row*M + col] = acc;
}
}

// helper function for ceiling unsigned integer division
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
  }

torch::Tensor matmul(torch::Tensor A, torch::Tensor B, int M) {
    assert(A.device().type() == torch::kCUDA && B.device().type() == torch::kCUDA);
    assert(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32);

    auto result = torch::empty({M, M}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));

    dim3 threads_per_block(16, 16);     // using 256 threads per block
    dim3 number_of_blocks(cdiv(M, threads_per_block.x),
                          cdiv(M, threads_per_block.y));

    matmul_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
    result.data_ptr<float>(),
    A.data_ptr<float>(),
    B.data_ptr<float>(),
    M);

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}
