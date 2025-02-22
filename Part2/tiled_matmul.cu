#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define TILE_DIM 16

__global__ void tiled_matmul_kernel(float* C, float* A, float* B, int N)
{
    // defining row, col, of the thread
    unsigned int row = blockDim.y*blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x*blockIdx.x + threadIdx.x;
    
    // defining tiled memory 
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    // defining final prod
    float sum = 0.0f;

    for (unsigned int i = 0; i < N/TILE_DIM; ++i)
    {
        //loading tile
        A_s[threadIdx.y][threadIdx.x] = A[TILE_DIM*i + row*N + threadIdx.x];
        B_s[threadIdx.y][threadIdx.x] = B[(TILE_DIM*i + threadIdx.y)*N + col];
        __syncthreads();
        for(unsigned int j = 0; j < TILE_DIM; ++j)
        {
            sum+=A_s[threadIdx.y][j] * B_s[j][threadIdx.x];
        }
        __syncthreads();
    }
    C[row*N + col] = sum;
}


// helper function for ceiling unsigned integer division
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
  }

torch::Tensor tiled_matmul(torch::Tensor A, torch::Tensor B, int M) {
    assert(A.device().type() == torch::kCUDA && B.device().type() == torch::kCUDA);
    assert(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32);

    auto result = torch::empty({M, M}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));

    dim3 threads_per_block(16, 16);     // using 256 threads per block
    dim3 number_of_blocks(cdiv(M, threads_per_block.x),
                          cdiv(M, threads_per_block.y));

    tiled_matmul_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
    result.data_ptr<float>(),
    A.data_ptr<float>(),
    B.data_ptr<float>(),
    M);

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}
