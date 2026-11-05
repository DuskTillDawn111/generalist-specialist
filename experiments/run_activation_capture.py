"""Bind each native training/validation capture to its actual same-model GPU."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def gpu():
    result=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version',
                           '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
    lines=result.stdout.strip().splitlines()
    if len(lines)!=1:raise ValueError('Explicit single-GPU experiment environment required')
    name,uuid,memory,driver=[s.strip() for s in lines[0].split(',')]
    return dict(name=name,uuid=uuid,memory_mib=float(memory),driver_version=driver)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();directory=args.output.resolve();path=directory/'hardware.json'
    if path.exists() or (directory/'raw.json').exists():raise FileExistsError('Existing native capture receipt')
    # All queued wrappers serialize here, while the collector's original lock
    # still detects unrelated GPU experiments. Waiting never starts a worker.
    scheduler=(ROOT/'artifacts/.gpu-campaign-scheduler.lock').open('a')
    fcntl.flock(scheduler,fcntl.LOCK_EX)
    plan=json.loads((directory/'plan.json').read_text());context=json.loads((directory/'context.json').read_text())
    before=gpu();expected=context['hardware']
    if expected!={'policy':'same_model','name':before['name']} or before['memory_mib']<49140:
        raise ValueError('Native collector requires matched GPU model and memory budget')
    pool=ROOT/'artifacts/static-study-v6-20261021/inputs/runtime-inputs.jsonl'
    pool_bytes=pool.read_bytes()
    if hashlib.sha256(pool_bytes).hexdigest()!=context['input_pool_sha256']:
        raise ValueError('Frozen native input pool changed')
    input_receipts=[]
    for item in (json.loads(line) for line in pool_bytes.decode().splitlines()):
        sample=item['sample']
        if 'input' not in sample:continue
        actual=hashlib.sha256(Path(sample['input']).read_bytes()).hexdigest()
        if actual!=sample['input_sha256']:raise ValueError('Frozen native input content changed: '+item['pool_id'])
        input_receipts.append(dict(pool_id=item['pool_id'],path=sample['input'],sha256=actual))
    with (directory/'input-verification.json').open('x') as out:
        json.dump(dict(input_pool_sha256=context['input_pool_sha256'],inputs=input_receipts),out,indent=2);out.write('\n')
    receipt=dict(schema='gsdeploy.native_capture_hardware.v1',policy='same_model',
        before=before,started_unix=time.time(),execution_context_sha256=plan['execution_context_sha256'],
        input_verification_sha256=hashlib.sha256((directory/'input-verification.json').read_bytes()).hexdigest(),
        wrapper_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    result=subprocess.run([sys.executable,str(ROOT/'experiments/collect_activation_training.py'),
                           'collect','--output',str(directory)],cwd=ROOT)
    after=gpu();receipt.update(after=after,finished_unix=time.time(),collector_exit_code=result.returncode)
    raw_path=directory/'raw.json'
    receipt['raw_sha256']=hashlib.sha256(raw_path.read_bytes()).hexdigest() if raw_path.exists() else None
    receipt['same_device_during_capture']=(before==after)
    with path.open('x') as out:json.dump(receipt,out,ensure_ascii=False,indent=2);out.write('\n')
    fcntl.flock(scheduler,fcntl.LOCK_UN);scheduler.close()
    return result.returncode if receipt['same_device_during_capture'] else 1


if __name__=='__main__':raise SystemExit(main())
