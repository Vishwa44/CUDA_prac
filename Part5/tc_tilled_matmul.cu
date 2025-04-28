#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define TILE_DIM 16
#define COARSENING_FACTOR 8
#define COARSENING_FACTOR_ROW 8
#define COARSENING_FACTOR_COL 8


__global__ void block_tiled_2D_matmul_kernel(float* C, float* A, float* B, int M, int N, int K)
{
    // Thread coordinates
    unsigned int row_base = (blockDim.y * blockIdx.y + threadIdx.y)* COARSENING_FACTOR_ROW; // Base row
    unsigned int col_base = (blockDim.x * blockIdx.x + threadIdx.x) * COARSENING_FACTOR_COL; // Base column
    
    // Shared memory for tiles
    __shared__ float As[TILE_DIM * COARSENING_FACTOR_ROW][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM * COARSENING_FACTOR_COL]; // Expanded shared memory for coarsening

    // Each thread computes COARSENING_FACTOR elements along the column direction
    float threadResults[COARSENING_FACTOR_ROW * COARSENING_FACTOR_COL] = {0.0f};
    float regRow[COARSENING_FACTOR_ROW] = {0.0};
    float regCol[COARSENING_FACTOR_COL] = {0.0};

                
    for(unsigned int t = 0; t < K/TILE_DIM; ++t)
    {
        // Load tiles from A - each thread loads COARSENING_FACTOR_ROW elements
        for(unsigned int rc = 0; rc < COARSENING_FACTOR_ROW; ++rc)   
        {
            As[threadIdx.y * COARSENING_FACTOR_ROW + rc][threadIdx.x] = A[(row_base + rc)*K + (t*TILE_DIM + threadIdx.x)];
        }
        // Load tiles from B - each thread loads COARSENING_FACTOR_COL elements
        for(unsigned int cc = 0; cc < COARSENING_FACTOR_COL; ++cc)   
        {
            Bs[threadIdx.y][threadIdx.x * COARSENING_FACTOR_COL + cc] = 
                B[(t*TILE_DIM + threadIdx.y)*N + col_base + cc];
        }
        
        __syncthreads();
        
        // Multiply the tiles together
        for(unsigned int k = 0; k < TILE_DIM; ++k)
        {
            for (unsigned int rc = 0; rc < COARSENING_FACTOR_ROW; ++rc)
            {
                regRow[rc] = As[threadIdx.y * COARSENING_FACTOR_ROW + rc][k];
            }
            for (unsigned int cc = 0; cc < COARSENING_FACTOR_COL; ++cc)
            {
                regCol[cc] = Bs[k][threadIdx.x * COARSENING_FACTOR_COL + cc];
            }
            for (unsigned int rc = 0; rc < COARSENING_FACTOR_ROW; ++rc)
            {
                for (unsigned int cc = 0; cc < COARSENING_FACTOR_COL; ++cc)
                {
                    threadResults[COARSENING_FACTOR_COL*rc + cc] += regRow[rc] * regCol[cc];
                }
            }
        }     
        __syncthreads();
    }
    
    // Write results to global memory
    for(unsigned int rc = 0; rc < COARSENING_FACTOR_ROW; ++rc)
    {
        for(unsigned int cc = 0; cc < COARSENING_FACTOR_COL; ++cc) 
        {
            C[(row_base + rc)*N + col_base + cc] = threadResults[rc*COARSENING_FACTOR_COL + cc];
        }
    }
}

// __global__ void block_tiled_1D_matmul_kernel(float* C, float* A, float* B, int M, int N, int K)
// {
//     // Thread coordinates
//     unsigned int row = blockDim.y * blockIdx.y + threadIdx.y; // Row in output matrix
//     unsigned int col_base = (blockDim.x * blockIdx.x + threadIdx.x) * COARSENING_FACTOR; // Base column
    
//     // Shared memory for tiles
//     __shared__ float As[TILE_DIM][TILE_DIM];
//     __shared__ float Bs[TILE_DIM][TILE_DIM * COARSENING_FACTOR]; // Expanded shared memory for coarsening

//     // Each thread computes COARSENING_FACTOR elements along the column direction
//     float threadResults[COARSENING_FACTOR] = {0.0f};
                
//     for(unsigned int t = 0; t < K/TILE_DIM; ++t)
//     {
//         // Load tile from A - each thread loads one element
//         As[threadIdx.y][threadIdx.x] = A[row*K + (t*TILE_DIM + threadIdx.x)];
        
//         // Load tiles from B - each thread loads COARSENING_FACTOR elements
//         for(unsigned int c = 0; c < COARSENING_FACTOR; ++c)   
//         {
//             Bs[threadIdx.y][threadIdx.x * COARSENING_FACTOR + c] = 
//                 B[(t*TILE_DIM + threadIdx.y)*N + col_base + c];
//         }
        
//         __syncthreads();
        
//         // Multiply the tiles together
//         for(unsigned int k = 0; k < TILE_DIM; ++k)
//         {
//             float Atemp = As[threadIdx.y][k];
//             for (unsigned int c = 0; c < COARSENING_FACTOR; ++c)
//             {
//                 threadResults[c] += Atemp * Bs[k][threadIdx.x * COARSENING_FACTOR + c];
//             }
//         }     
//         __syncthreads();
//     }
    
//     // Write results to global memory
//     for(unsigned int c = 0; c < COARSENING_FACTOR; ++c) 
//     {
//         C[row*N + col_base + c] = threadResults[c];
//     }
// }


// __global__ void tiled_matmul_kernel(float* C, float* A, float* B, int M, int N, int K)
// {
//     // Thread coordinates
//     unsigned int row = blockDim.y * blockIdx.y + threadIdx.y; // Row in output matrix
//     unsigned int col = blockDim.x * blockIdx.x + threadIdx.x; // Column in output matrix
    
//     if (row < M && col < N)
//     {
//         __shared__ float As[TILE_DIM][TILE_DIM];
//         __shared__ float Bs[TILE_DIM][TILE_DIM]; 

//         float sum = 0.0f;
                
//         for(unsigned int t = 0; t < K/TILE_DIM; ++t)
//         {

//             As[threadIdx.y][threadIdx.x] = A[row*K + (t*TILE_DIM + threadIdx.x)];   
//             Bs[threadIdx.y][threadIdx.x] = B[(t*TILE_DIM + threadIdx.y)*N + col];
            
//             __syncthreads();
            
//             // Multiply the tiles together
//             for(unsigned int k = 0; k < TILE_DIM; ++k)
//             {
//                 sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
//             }     
//             __syncthreads();
//         }
        
//         // Write the result to global memory
//         C[row*N + col] = sum;
//     }
// }

// __global__ void matmul_kernel(float* C, float* A, float* B, int M, int N, int K)
// {
//     // Thread coordinates
//     unsigned int row = blockDim.y * blockIdx.y + threadIdx.y; // Row in output matrix
//     unsigned int col = blockDim.x * blockIdx.x + threadIdx.x; // Column in output matrix
    
//     if (row < M && col < N)
//     {

//         float sum = 0.0f;
            
//         for(unsigned int k = 0; k < K; ++k)
//         {
//             sum += A[row*K + k] * B[k*N + col];
//         }     
        
//         // Write the result to global memory
//         C[row*N + col] = sum;
//     }
// }



// helper function for ceiling unsigned integer division
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
  }

torch::Tensor tiled_matmul(torch::Tensor A, torch::Tensor B, int M, int N, int K) {
    assert(A.device().type() == torch::kCUDA && B.device().type() == torch::kCUDA);
    assert(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32);

    auto result = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));

    // dim3 threads_per_block(32, 32);     // using 1024 threads per block
    // dim3 number_of_blocks(cdiv(N, threads_per_block.x),
    //                       cdiv(M, threads_per_block.y));

    // matmul_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
    // result.data_ptr<float>(),
    // A.data_ptr<float>(),
    // B.data_ptr<float>(),
    // M, N, K);

    // dim3 threads_per_block(TILE_DIM, TILE_DIM);     // using 1024 threads per block
    // dim3 number_of_blocks(cdiv(N, threads_per_block.x),
    //                       cdiv(M, threads_per_block.y));

    // tiled_matmul_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
    //     result.data_ptr<float>(),
    //     A.data_ptr<float>(),
    //     B.data_ptr<float>(),
    //     M, N, K);

    // dim3 threads_per_block(TILE_DIM, TILE_DIM);
    // dim3 number_of_blocks(N/(TILE_DIM*COARSENING_FACTOR), M/TILE_DIM);

    // block_tiled_1D_matmul_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
    //     result.data_ptr<float>(),
    //     A.data_ptr<float>(),
    //     B.data_ptr<float>(),
    //     M, N, K);
    
    dim3 threads_per_block(TILE_DIM, TILE_DIM);
    dim3 number_of_blocks(N/(TILE_DIM*COARSENING_FACTOR_COL), M/(TILE_DIM*COARSENING_FACTOR_ROW));
    
    block_tiled_2D_matmul_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
            result.data_ptr<float>(),
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            M, N, K);
    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}
