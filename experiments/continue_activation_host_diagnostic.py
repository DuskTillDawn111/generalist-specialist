"""Run the frozen passive host/GC queue and analysis; preserve all terminal failures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True,type=Path);a=p.parse_args()
    base=a.base.resolve();path=base/'state.json'
    if path.exists():raise FileExistsError(path)
    design=json.loads((base/'design.json').read_text())
    state=dict(status='host_diagnostic_collecting',physical_predictor_qualified=False,formal_utility_qualified=False)
    save(path,state)
    def check():
        for r in design['source_receipts']:
            if any(hashlib.sha256(Path(r[k]).read_bytes()).hexdigest()!=r['sha256'] for k in ('path','snapshot')):
                raise ValueError('Frozen host diagnostic source drift: '+r['path'])
    def run(script,*args):
        check();subprocess.run([sys.executable,str(ROOT/'experiments'/script),*map(str,args)],cwd=ROOT,check=True);check()
    try:
        run('run_activation_queue.py','--queue',base/'queue.json')
        state['status']='host_diagnostic_analyzing';save(path,state)
        run('analyze_activation_host_diagnostic.py','--base',base,'--output',base/'description.json')
        report=json.loads((base/'description.json').read_text())
        state.update(status='host_diagnostic_completed',observer_repeat_check_passed=report['observer_repeat_check_passed'],issues=report['issues'])
        save(path,state)
    except BaseException as exc:
        state.update(status='host_diagnostic_failed',error=type(exc).__name__+': '+str(exc));save(path,state);raise


if __name__=='__main__':main()
