#include "timer.h"

void vec_add_cpu(float* x, float* y, float* z, int N)
{
    for(unsigned int i = 0; i < N; ++i)
    {
        z[i] = x[i]+ y[i];
    }
}

__global__ void vecadd_kernel(float* x, float* y, float* z, int N)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < N)
    {
        z[i] = x[i] + y[i];
    }
}

void vec_add_gpu(float* x, float* y, float* z, int N)
{
    // Allocate GPU memory
    float *x_d, *y_d, *z_d;
    cudaMalloc((void**)&x_d, N*sizeof(float));
    cudaMalloc((void**)&y_d, N*sizeof(float));
    cudaMalloc((void**)&z_d, N*sizeof(float));

    //copy to GPU memory
    cudaMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // calling the GPU kernel
    const unsigned int numThreadPerBlock = 512;
    const unsigned numBlocks = (N + numThreadPerBlock - 1)/numThreadPerBlock;
    vecadd_kernel<<<numBlocks, numThreadPerBlock>>>(x_d, y_d, z_d, N);

    //copy from GPU
    cudaMemcpy(z, z_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    //deallocate GPU memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}

int main(int argc, char**argv)
{
    cudaDeviceSynchronize();

    //Allocate memory and initilaize the data
    Timer timer;
    unsigned int N = (argc > 1)?(atoi(argv[1])):(1<<32);
    float* x = (float*) malloc(N*sizeof(float));
    float* y = (float*) malloc(N*sizeof(float));
    float* z = (float*) malloc(N*sizeof(float));
    for (unsigned int i = 0; i < N; ++i)
    {
        x[i] = rand();
        y[i] = rand();
    }
    // vector addition on CPU
    startTime(&timer);
    vec_add_cpu(x, y, z, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU Time", "");

    //vector addition on GPU
    startTime(&timer);
    vec_add_gpu(x, y, z, N);
    stopTime(&timer);
    printElapsedTime(timer, "GPU Time", "");

    //free memory
    free(x);
    free(y);
    free(z);

    return 0;
}