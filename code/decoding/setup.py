from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='torchac',
    ext_modules=[
        cpp_extension.CUDAExtension('torchac', 
                                    ['torchac_kernel.cu']),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)
