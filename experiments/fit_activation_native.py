"""Fit explicit native activity laws from completed matched training captures."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gsdeploy.forecasting.native_training import fit_native_captures


def read(path):return json.loads(path.read_text())


def load_bundle(directory):
    bundle={name:read(directory/(name+'.json')) for name in ('raw','plan','context')}
    receipts=read(directory/'sources.json')
    expected=bundle['context']['source_content']
    if [{k:r[k] for k in ('path','sha256')} for r in receipts]!=expected:
        raise ValueError('Saved collector source receipts differ from execution context')
    for receipt in receipts:
        if hashlib.sha256(Path(receipt['snapshot']).read_bytes()).hexdigest()!=receipt['sha256']:
            raise ValueError('Collector source snapshot is missing or changed: '+receipt['path'])
    if bundle['raw']['kind']=='measured':
        hardware=read(directory/'hardware.json');expected=bundle['context']['hardware']
        if (hardware.get('schema')!='gsdeploy.native_capture_hardware.v1'
                or hardware.get('collector_exit_code')!=0 or hardware.get('same_device_during_capture') is not True
                or hardware.get('raw_sha256')!=hashlib.sha256((directory/'raw.json').read_bytes()).hexdigest()
                or hardware.get('input_verification_sha256')!=hashlib.sha256((directory/'input-verification.json').read_bytes()).hexdigest()
                or hardware.get('execution_context_sha256')!=bundle['raw']['execution_context_sha256']
                or hardware.get('before')!=hardware.get('after')
                or hardware['before']['name']!=expected['name'] or expected['policy']!='same_model'
                or hardware['before']['memory_mib']<49140):
            raise ValueError('Capture lacks matching actual GPU model and immutable raw receipt')
    bundle['source']=str(directory/'raw.json')+':sha256:'+hashlib.sha256((directory/'raw.json').read_bytes()).hexdigest()
    return bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',action='append',required=True,type=Path)
    parser.add_argument('--pool',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--max-work-error',type=float,default=.2)
    parser.add_argument('--block-seconds',type=float,default=5.)
    parser.add_argument('--order-aware',action='store_true',help='Separate exploratory post-prime order descriptor; original fit remains unchanged')
    args=parser.parse_args()
    bundles=[load_bundle(p.resolve()) for p in args.capture]
    pool_bytes=args.pool.read_bytes();pool_hash=hashlib.sha256(pool_bytes).hexdigest()
    if any(b['context']['input_pool_sha256']!=pool_hash for b in bundles):
        raise ValueError('Frozen training input pool content changed')
    pool=[json.loads(line) for line in pool_bytes.decode().splitlines()]
    result=fit_native_captures(bundles,pool,max_work_error=args.max_work_error,block_seconds=args.block_seconds,
                               order_aware=args.order_aware)
    # Exclusive output creation preserves prior fit receipts and failed fits.
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:
        json.dump(result,out,ensure_ascii=False,indent=2,allow_nan=False);out.write('\n')
    law=result['profiles']['native_activity_law']
    print(json.dumps(dict(output=str(args.output),status=result['status'],
        progress_groups=len(law['progress_groups']),
        rejected_groups=[r for r in law['fit_diagnostics'] if not r['accepted']],
        resources=result['resource_diagnostic'],admission_enabled=False),ensure_ascii=False))


if __name__=='__main__':main()
