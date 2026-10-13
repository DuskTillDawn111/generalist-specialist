"""Native Triton cache bytes/keys/lookup, same-directory atomic publication.

Linux Landlock ABI1 forbids cross-directory rename (EXDEV), even inside one
allowed output tree. This changes only staging location, never GPU kernels,
compiler flags, precision, task mapping, batching or inference decisions.
"""
import os,uuid
from pathlib import Path
from triton.runtime.cache import FileCacheManager
from .isolation import ROOT

class SameDirectoryCacheManager(FileCacheManager):
 def put(self,data,filename,binary=True):
  if not self.cache_dir:raise RuntimeError('Could not create or locate cache dir')
  assert self.lock_path is not None
  filepath=self._make_path(filename);parent=Path(filepath).parent.resolve()
  if not parent.is_relative_to(ROOT):raise PermissionError('Triton cache must be inside Module35')
  binary=isinstance(data,bytes)
  if not binary:data=str(data)
  temp_path=parent/f'.tmp.pid_{os.getpid()}_{uuid.uuid4()}_{Path(filename).name}'
  try:
   with open(temp_path,'wb' if binary else 'w') as stream:stream.write(data)
   os.replace(temp_path,filepath)
  finally:
   if temp_path.exists():temp_path.unlink()
  return filepath
