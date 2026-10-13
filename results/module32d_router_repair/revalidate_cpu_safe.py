"""Safe post-engineering reproduction; explicitly redirects inherited output sinks.

The original frozen CPU fixture is retained as incident evidence. Run THIS entry
point for reproduction; all test artifacts go into Module32D/cpu_safe_revalidation.
No GPU inference and no writes to any historical experiment directory.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cpu_validation as cpu
from m32d import runtime,common
runtime.configure()
import module30.telemetry as inherited
original_save=common.save
original_csvsave=common.csvsave
def save(n,v):return original_save('cpu_safe_revalidation/'+n,v)
def csvsave(n,v,fields=None):return original_csvsave('cpu_safe_revalidation/'+n,v,fields)
inherited.OUT=common.OUT/'cpu_safe_revalidation'
inherited.save=save
cpu.save=save;cpu.csvsave=csvsave
if __name__=='__main__':
 print('SAFE_EQUIVALENCE',cpu.equivalence(),flush=True)
 print('SAFE_CPU_BENCHMARK',cpu.benchmark(),flush=True)
