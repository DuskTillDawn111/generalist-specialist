"""Serial native capture queue with explicit upstream completion and receipts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue',required=True,type=Path)
    args=p.parse_args();queue=json.loads(args.queue.read_text());state_path=Path(queue['state'])
    if state_path.exists():raise FileExistsError(state_path)
    state=dict(status='waiting',completed=[],errors=[],queue=str(args.queue.resolve()));save(state_path,state)
    for condition in queue.get('after',[]):
        deadline=time.monotonic()+21600
        while True:
            path=Path(condition['path']);upstream=json.loads(path.read_text()) if path.exists() else {}
            if upstream.get('status')==condition['status']:break
            if 'failed' in upstream.get('status','') or upstream.get('errors'):
                state.update(status='upstream_failed',upstream=upstream);save(state_path,state);return 1
            if time.monotonic()>deadline:raise TimeoutError('Capture queue prerequisite timed out')
            time.sleep(2)
    for directory in map(Path,queue['captures']):
        state.update(status='collecting',current=str(directory));save(state_path,state)
        with (directory/'collector.log').open('x') as log:
            result=subprocess.run([sys.executable,str(ROOT/'experiments/run_activation_capture.py'),
                                   '--output',str(directory)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state['errors'].append(str(directory));state['status']='capture_failed';save(state_path,state);return 1
        state['completed'].append(str(directory));save(state_path,state)
    state.update(status='capture_plan_completed',current=None);save(state_path,state);return 0


if __name__=='__main__':raise SystemExit(main())
