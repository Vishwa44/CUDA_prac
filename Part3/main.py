import torch
from torch.utils.cpp_extension import load_inline
from pathlib import Path
import time

def compile_extension():
    cuda_source = Path("Part3/distance_mat.cu").read_text()
    cpp_source = "torch::Tensor disctance_calc_main(torch::Tensor A, int M, int N);"

    # Load the CUDA kernel as a PyTorch extension
    distance_ext = load_inline(
        name="distance_ext",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["disctance_calc_main"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return distance_ext



def main():
    ext = compile_extension()
    N = 1024
    M = 102400
    a = torch.rand(N, M).cuda()
    st = time.time()
    # c_torch = torch.cdist(a, a, 1)
    # print("using PyTorch: ", c_torch.shape)
    # print("using PyTorch time ", time.time() - st)
    # print(c_torch)
    st = time.time()
    c_cuda = ext.disctance_calc_main(a, M, N)
    print("using CUDA: ", c_cuda.shape)
    print("using CUDA time ", time.time() - st)
    print(c_cuda)
    
if __name__ == "__main__":
    main()
