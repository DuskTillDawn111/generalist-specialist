"""Queued OCR-2 full-input captures, serialized with other native wrappers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True,type=Path)
    args=p.parse_args();base=args.base.resolve();path=base/'replica-continuation-v1.json'
    if path.exists():raise FileExistsError(path)
    status=dict(status='waiting_for_initial_captures',completed=[],errors=[]);save(path,status)
    deadline=time.monotonic()+21600
    while True:
        upstream=json.loads((base/'continuation-v1.json').read_text())
        if upstream['status']=='initial_capture_plan_completed':break
        if upstream['status'] in {'capture_failed','reference_capture_failed'}:
            status.update(status='upstream_failed',upstream=upstream);save(path,status);return 1
        if time.monotonic()>deadline:raise TimeoutError('Initial capture wait exceeded')
        time.sleep(2)
    for name in ('solo-round1','solo-round2','overlap-round1','overlap-round2'):
        directory=base/'training-v2/g2-image-ocr2'/name
        status.update(status='collecting',current=str(directory));save(path,status)
        with (directory/'collector.log').open('x') as log:
            result=subprocess.run([sys.executable,str(ROOT/'experiments/run_activation_capture.py'),
                                   '--output',str(directory)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status['errors'].append(str(directory));status['status']='capture_failed';save(path,status);return 1
        status['completed'].append(str(directory));save(path,status)
    status.update(status='replica_capture_plan_completed',current=None);save(path,status);return 0


if __name__=='__main__':raise SystemExit(main())
