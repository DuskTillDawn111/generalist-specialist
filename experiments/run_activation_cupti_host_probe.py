"""Serialized one-worker Nsight capture, immutable export, and host API analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import save


def receipt(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);a=p.parse_args();base=a.base.resolve()
    state_path=base/'state.json'
    if state_path.exists():raise FileExistsError(state_path)
    design=json.loads((base/'design.json').read_text());directory=Path(design['capture'])
    state=dict(status='cupti_host_probe_collecting',physical_predictor_qualified=False,formal_utility_qualified=False);save(state_path,state)
    def check():
        for r in design['source_receipts']:
            if any(hashlib.sha256(Path(r[k]).read_bytes()).hexdigest()!=r['sha256'] for k in ('path','snapshot')):
                raise ValueError('Frozen source drift: '+r['path'])
        if hashlib.sha256(Path(design['nsys']['tool']).read_bytes()).hexdigest()!=design['nsys']['executable_sha256']:
            raise ValueError('Nsight executable changed')
    def run(command,log):
        check()
        with log.open('x') as stream:subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        check()
    try:
        run([sys.executable,str(ROOT/'experiments/run_activation_capture.py'),'--output',str(directory)],directory/'collector.log')
        report=Path(design['expected_nsys_report'])
        if not report.is_file():raise ValueError('Nsight worker produced no completed report')
        state['status']='cupti_host_probe_exporting';save(state_path,state)
        sql=base/'trace.sqlite'
        run([design['nsys']['tool'],'export','--type=sqlite','--force-overwrite=false','--output='+str(sql),str(report)],base/'export.log')
        save(base/'export-receipt.json',dict(report=receipt(report),sqlite=receipt(sql),nsys=design['nsys']))
        state['status']='cupti_host_probe_analyzing';save(state_path,state)
        run([sys.executable,str(ROOT/'experiments/analyze_activation_cupti_host_probe.py'),'--base',str(base),
             '--output',str(base/'description.json')],base/'analysis.log')
        d=json.loads((base/'description.json').read_text());state.update(status='cupti_host_probe_completed',output_changes=len(d['output_changes']))
        save(state_path,state)
    except BaseException as exc:
        state.update(status='cupti_host_probe_failed',error=type(exc).__name__+': '+str(exc));save(state_path,state);raise


if __name__=='__main__':main()
