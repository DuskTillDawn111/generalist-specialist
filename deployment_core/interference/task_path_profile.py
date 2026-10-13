"""Offline ordered launch traces and explicitly anchored duration reconstruction."""
import json, math
from pathlib import Path
from dataclasses import dataclass
@dataclass(frozen=True)
class TaskPathProfile:
 model: str
 task: str
 standalone_ms: float
 requests: list
 metadata_complete: bool
 source: dict
 @classmethod
 def load(cls,path):
  d=json.loads(Path(path).read_text())
  obj=cls(d['model'],d['task'],d['standalone_s0_ms'],d['requests'],d['metadata_complete'],d['source'])
  if not math.isfinite(obj.standalone_ms) or obj.standalone_ms<=0:raise ValueError('positive s0 required')
  for r in obj.requests:
   if not r['kernels']:raise ValueError('empty trace')
   if not math.isclose(math.fsum(k['duration_corrected_ms'] for k in r['kernels']),obj.standalone_ms,rel_tol=1e-13,abs_tol=1e-10):raise ValueError('duration anchor mismatch')
  return obj

def reconstruct(profile_ns,standalone_ms):
 if not profile_ns or standalone_ms<=0 or any(v<=0 for v in profile_ns):raise ValueError('positive durations required')
 total=math.fsum(profile_ns);values=[standalone_ms*v/total for v in profile_ns]
 values[-1]+=standalone_ms-math.fsum(values)
 if values[-1]<=0:raise ValueError('nonpositive reconstruction')
 return values
