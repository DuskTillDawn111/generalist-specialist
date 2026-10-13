"""Identical native backend/settings under inherited confinement and Python audit."""
import sys,runpy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from extra.isolation import install,VIOLATIONS,ROOT
install()
import os
os.environ['TRITON_CACHE_MANAGER']='extra.triton_cache:SameDirectoryCacheManager'
target=sys.argv[1];sys.argv=sys.argv[1:];sys.path.insert(0,str(Path(target).parent))
try:runpy.run_path(target,run_name='__main__')
finally:
 import json,os
 (ROOT/'work/control_guards'/f'{os.getpid()}_exit.json').write_text(json.dumps(dict(pid=os.getpid(),unexpected_writes=VIOLATIONS))+'\n')
