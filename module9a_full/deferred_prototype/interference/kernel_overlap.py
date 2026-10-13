"""CPU event simulator, with no GPU imports or runtime profiling."""
import ctypes
from pathlib import Path
import numpy as np
from .occupancy import resource_vector,active_grid_blocks
LIB=Path(__file__).with_name('_kernel_overlap.so')
_backend=None
def backend():
 global _backend
 if _backend is None:
  _backend=ctypes.CDLL(str(LIB));_backend.simulate.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double)];_backend.simulate.restype=ctypes.c_double
 return _backend

def kernel_array(kernels,limits,raw=False):
 a=[]
 for k in kernels:
  v=resource_vector(k,limits);a.append([k['duration_profile_ns']/1e6 if raw else k['duration_corrected_ms'],v[0],v[2],v[3],active_grid_blocks(k,limits),k['max_active_blocks_per_sm']])
 return np.ascontiguousarray(a,dtype=np.float64)

def simulate_arrays(a,b,start,limits,policy='FCFS_CONTEXT_SERIAL',first=0):
 if policy not in ['FCFS_CONTEXT_SERIAL','EQUAL_PARTITION']:raise ValueError(policy)
 caps=np.array([limits.warps,limits.blocks,limits.registers,limits.shared_bytes],dtype=np.float64)
 ptr=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
 return backend().simulate(ptr(a),len(a),ptr(b),len(b),*start,0 if policy=='FCFS_CONTEXT_SERIAL' else 1,first,ptr(caps))
