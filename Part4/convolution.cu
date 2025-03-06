#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define FILTER_DIM 5

__constant__ float filter_c[FILTER_DIM][FILTER_DIM];

__global__ void convolution_kernel(float* result, float* mat, int height, int width)
{
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(outRow < height && outCol < width){    
        float sum = 0.0f;
        for(int j=0; j < FILTER_DIM; ++j)
        {
            for(int i=0; i < FILTER_DIM; ++i)
            {
                int mRow = outRow - ((FILTER_DIM-1)/2) + j;
                int mCol = outCol - ((FILTER_DIM-1)/2) + i;
                if (mRow >= 0 && mRow < height && mCol >= 0 && mCol < width)
                {
                    sum += filter_c[j][i] * mat[mRow*width + mCol];
                } 
            }
        }
        result[outRow*width + outCol] = sum;    
    }
}


// helper function for ceiling unsigned integer division
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
  }


torch::Tensor convolution(torch::Tensor mat, torch::Tensor filter, int height, int width) {
    assert(mat.device().type() == torch::kCUDA );//&& filter.device().type() == torch::kCUDA
    assert(mat.dtype() == torch::kFloat32 );//&& filter.dtype() == torch::kFloat32
    
    torch::Tensor filter_cpu = filter.to(torch::kCPU).contiguous();

    assert(filter_cpu.size(0) == FILTER_DIM && filter_cpu.size(1) == FILTER_DIM);

    cudaMemcpyToSymbol(filter_c, filter_cpu.data_ptr<float>(), FILTER_DIM*FILTER_DIM*sizeof(float));

    auto result = torch::empty({height, width}, torch::TensorOptions().dtype(torch::kFloat32).device(filter.device()));

    // M = 256
    dim3 threads_per_block(16, 16);     // using 256 threads per block
    dim3 number_of_blocks(cdiv(height, threads_per_block.x),
                          cdiv(width, threads_per_block.y));

    convolution_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
    result.data_ptr<float>(),
    mat.data_ptr<float>(),
    height,
    width);

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}