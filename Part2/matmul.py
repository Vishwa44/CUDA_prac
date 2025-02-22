import torch
from torch.utils.cpp_extension import load_inline
from pathlib import Path
import time

def compile_extension_tiled():
    cuda_source = Path("Part2/tiled_matmul.cu").read_text()
    cpp_source = "torch::Tensor tiled_matmul(torch::Tensor A, torch::Tensor B, int M);"

    # Load the CUDA kernel as a PyTorch extension
    matmul_extension = load_inline(
        name="matmul_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["tiled_matmul"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return matmul_extension


def compile_extension_std():
    cuda_source = Path("Part2/matmul_kernel.cu").read_text()
    cpp_source = "torch::Tensor matmul(torch::Tensor A, torch::Tensor B, int M);"

    # Load the CUDA kernel as a PyTorch extension
    matmul_extension = load_inline(
        name="matmul_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["matmul"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return matmul_extension


def main():
    ext_std = compile_extension_std()
    ext_tiled = compile_extension_tiled()
    M = 1024
    a = torch.rand(M, M).cuda()
    b = torch.rand(M, M).cuda()
    st = time.time()
    c_torch = torch.matmul(a, b)
    print("using torch: ", c_torch.shape)
    print("using torch time ", time.time() - st)
    print(c_torch)
    st = time.time()
    c_cuda = ext_std.matmul(a, b, M)
    print("using CUDA: ", c_cuda.shape)
    print("using CUDA time ", time.time() - st)
    print(c_cuda)
    st = time.time()
    c_cuda_tiled = ext_tiled.tiled_matmul(a, b, M)
    print("using tiled CUDA: ", c_cuda_tiled.shape)
    print("using tiled CUDA time ", time.time() - st)
    print(c_cuda_tiled)
    
if __name__ == "__main__":
    main()
