#include <cmath>
#define TILE_WIDTH 1024

__global__ void tiled_disctance_calc(float* output, float* a, int N, int M) {
    int row1 = blockIdx.x;
    int row2 = threadIdx.x;
    __shared__ float tile[TILE_WIDTH];
    float acc = 0.0f;
    for(unsigned int i = 0; i < M/TILE_WIDTH; ++i){
        // loading tile
        tile[threadIdx.x] = a[row1*M + i*TILE_WIDTH + threadIdx.x];
        __syncthreads(); //syncing thread
        for(unsigned int j = 0; j < TILE_WIDTH; ++j){
            acc += fabs(tile[j] - a[row2*M + i*TILE_WIDTH + j]);    
        }
        __syncthreads(); //syncing thread
    }
    output[row1*N + row2] = acc;
}

__global__ void disctance_calc(float* output, float* a, int N, int M) {
    int row1 = blockIdx.x; // loading same row1 for all the threads in the block
    int row2 = threadIdx.x;
    float acc = 0.0f;
    for (unsigned int i = 0; i < M; ++i){
        acc += fabs(a[row1*M + i] - a[row2*M + i]); // absolute calculation
    }
    output[row1*N + row2] = acc;
}



torch::Tensor disctance_calc_main(torch::Tensor A, int M, int N) {
    assert(A.device().type() == torch::kCUDA);
    assert(A.dtype() == torch::kFloat32);

    auto result = torch::empty({N, N}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));

    dim3 threads_per_block(N); // Launching the same number of blocks as number of vectors
    dim3 number_of_blocks(N); // Launching the same number of threads per block as number of vectors

    disctance_calc<<<number_of_blocks, threads_per_block, 0>>>(
    result.data_ptr<float>(),
    A.data_ptr<float>(),
    N,
    M);

    return result;
}