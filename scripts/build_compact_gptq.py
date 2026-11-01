"""CPU-only extension compilation; fixed SM89, no GPU discovery or execution.

Run from the pinned/patched AutoGPTQ source tree, in compact-minicpm. CUDA_HOME,
CC/CXX and LD_LIBRARY_PATH must identify the recorded CUDA12.4/Clang14 toolchain.
The checked-in header and equivalent dtype-dispatch patch reduces compiler memory and adapts
PyTorch2.6 without changing kernel math.
"""
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='', TORCH_CUDA_ARCH_LIST='8.9',
                  MAX_JOBS='1', BUILD_CUDA_EXT='1', COMPILE_MARLIN='0')
import torch
import setuptools


def forbidden(*args, **kwargs):
    raise RuntimeError('GPU discovery/initialization forbidden during CPU compilation')


torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda._lazy_init = forbidden
torch.cuda.get_device_capability = forbidden
original_setup = setuptools.setup


def selected_setup(**kwargs):
    # The pinned runtime explicitly disables ExLlama v2 and Marlin. Compile only
    # ExLlama v1 plus the generic fallback kernels it can actually select.
    allowed = {'autogptq_cuda_64', 'autogptq_cuda_256', 'exllama_kernels'}
    kwargs['ext_modules'] = [e for e in kwargs.get('ext_modules', []) if e.name in allowed]
    return original_setup(**kwargs)


setuptools.setup = selected_setup
setup = Path('setup.py').read_text()
# The fork overrides the selected compiler. Honor the explicitly recorded host compiler.
setup = setup.replace('os.environ["CC"] = "g++"', 'pass').replace('os.environ["CXX"] = "g++"', 'pass')
sys.argv = ['setup.py', 'bdist_wheel', '--dist-dir', os.environ['GS_WHEEL_DIR']]
exec(compile(setup, 'setup.py', 'exec'), {'__name__': '__main__', '__file__': 'setup.py'})
assert not torch.cuda.is_initialized()
