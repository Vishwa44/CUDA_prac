import torch
from torch.utils.cpp_extension import load_inline
from pathlib import Path
import time

def compile_extension():
    cuda_source = Path("Part4/convolution.cu").read_text()
    cpp_source = "torch::Tensor convolution(torch::Tensor mat, torch::Tensor filter, int height, int width);"

    # Load the CUDA kernel as a PyTorch extension
    convolution_ext = load_inline(
        name="convolution_ext",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["convolution"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return convolution_ext



def main():
    ext = compile_extension()
    height = 256
    width = 256
    filter_dim = 5
    a = torch.rand(height, width).cuda()
    filter = torch.rand(filter_dim, filter_dim).cuda()
    st = time.time()
    result = torch.nn.functional.conv2d(a.view(1, 1, height, width), filter.view(1, 1, filter_dim, filter_dim), padding=(filter_dim//2, filter_dim//2), bias=None, stride=1)
    print("using PyTorch: ", result.shape)
    print("using PyTorch time ", time.time() - st)
    print(result)
    st = time.time()
    c_cuda = ext.convolution(a, filter, height, width)
    print("using CUDA: ", c_cuda.shape)
    print("using CUDA time ", time.time() - st)
    print(c_cuda)
    
if __name__ == "__main__":
    main()
