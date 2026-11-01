"""Run all actual compact preprocessors sequentially on CPU, without checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while block:=f.read(1048576):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',default='configs/study.json')
    p.add_argument('--environments',default='configs/compact_environments.json')
    p.add_argument('--protocol',default='configs/compact_protocol.json')
    p.add_argument('--inventory',required=True)
    p.add_argument('--classmap',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=['quality','calibration','all'],default='all')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    study=json.loads(Path(a.study).read_text())
    envs=json.loads(Path(a.environments).read_text())
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    from gsdeploy.backends.compact import fingerprint
    protocol_sha=fingerprint(json.loads(Path(a.protocol).read_text()))
    for phase in ['quality','calibration'] if a.phase=='all' else [a.phase]:
        manifest=study[phase+'_preparation']['manifest']
        entries=json.loads(Path(manifest).read_text())['tasks']
        for candidate in study['generalists']+study['specialists']:
            slot=candidate['slot'];dest=out/('preprocessing-'+phase+'-'+slot)
            report=dest/'preprocessing.json'
            tasks=study['active_tasks'] if slot.startswith('generalist_') else [slot]
            expected={t:entries[t]['sha256'] for t in tasks}
            if report.is_file():
                previous=json.loads(report.read_text())
                if (previous['status']=='PASS_CPU_PREPROCESSING' and not previous['limited_diagnostic']
                        and previous['protocol_fingerprint']==protocol_sha and previous['input_manifest_sha256']==expected):
                    print('VERIFIED_EXISTING',phase,slot,previous['successful'],flush=True);continue
                raise ValueError('Changed or failed report; preserve it and choose a new output root: '+str(report))
            python=envs['models'][candidate['model']]['python']
            cmd=[python,'-m','gsdeploy.profiling.preprocessing','--study',str(Path(a.study).resolve()),
                '--protocol',str(Path(a.protocol).resolve()),'--inventory',str(Path(a.inventory).resolve()),
                '--classmap',str(Path(a.classmap).resolve()),'--slot',slot,'--output',str(dest),'--manifest',str(Path(manifest).resolve())]
            if dest.exists():cmd.append('--resume')
            print('CHECKING',phase,slot,flush=True)
            with (out/('preprocessing-'+phase+'-'+slot+'.log')).open('w') as stream:
                result=subprocess.run(cmd,cwd=root,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1',
                    'OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'},stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('CPU processor failed: '+phase+':'+slot+' exit='+str(result.returncode))
            data=json.loads(report.read_text());print(data['status'],data['successful'],flush=True)
            if data['failures']:raise ValueError(data['failures'][:3])


if __name__=='__main__':main()
