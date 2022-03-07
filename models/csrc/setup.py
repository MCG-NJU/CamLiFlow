from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_ext_modules():
    return [
        CUDAExtension(
            name='_correlation_cuda',
            sources=[
                'correlation/correlation.cpp',
                'correlation/correlation_forward_kernel.cu',
                'correlation/correlation_backward_kernel.cu'
            ],
            include_dirs=['correlation']
        ),
        CUDAExtension(
            name='_furthest_point_sampling_cuda',
            sources=[
                'furthest_point_sampling/furthest_point_sampling.cpp',
                'furthest_point_sampling/furthest_point_sampling_kernel.cu'
            ],
            include_dirs=['furthest_point_sampling']
        ),
        CUDAExtension(
            name='_k_nearest_neighbor_cuda',
            sources=[
                'k_nearest_neighbor/k_nearest_neighbor.cpp',
                'k_nearest_neighbor/k_nearest_neighbor_kernel.cu'
            ],
            include_dirs=['k_nearest_neighbor']
        )
    ]


setup(
    name='csrc',
    ext_modules=get_ext_modules(),
    cmdclass={'build_ext': BuildExtension}
)
