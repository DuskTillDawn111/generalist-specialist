"""Run current solvers with the independently proved development affine bound."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.spillover_context_20261022 import search_development as runner
from tools.spillover_context_20261022.development_affine_bounds import DevelopmentAffineBounds

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--algorithm',choices=['milp','dfs','plain_dfs'],default='milp');p.add_argument('--seeds',type=int,nargs='+',default=[107])
    p.add_argument('--time-limit',type=float,default=180);p.add_argument('--denominator',type=int,default=32);p.add_argument('--K',type=int,default=2)
    a=p.parse_args();runner.DevelopmentBounds=DevelopmentAffineBounds
    runner.search(a.output,a.algorithm,a.seeds,a.time_limit,a.denominator,a.K,bounds='derived')
