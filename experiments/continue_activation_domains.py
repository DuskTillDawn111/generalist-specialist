"""Process completed measured domains while the serialized GPU queue continues."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import save,POOL


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    args=p.parse_args();base=args.base.resolve();state_path=base/'domain-continuation-v1.json'
    if state_path.exists():raise FileExistsError(state_path)
    state=dict(status='waiting_for_domains',domains=[],errors=[]);save(state_path,state)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def run(script,argv,target):
        with target.with_suffix('.log').open('x') as log:
            r=subprocess.run([sys.executable,str(ROOT/'experiments'/script),*map(str,argv)],
                             cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        return r.returncode==0
    for resident,label in [('g2-image-ocr2','g2-ocr2'),('specialists-ocr1','specialists-ocr1'),('specialists-ocr2','specialists-ocr2')]:
        item=dict(resident=resident,status='waiting_for_complete_receipts');state['domains'].append(item);save(state_path,state)
        names=['solo-round1','solo-round2','overlap-round1','overlap-round2','idle-round1','idle-round2']
        captures=[base/'training-v2'/resident/n for n in names]
        deadline=time.monotonic()+21600
        while not all((d/'hardware.json').exists() and (d/'raw.json').exists() for d in captures):
            if time.monotonic()>deadline:raise TimeoutError('Native domain captures incomplete')
            for n in ['idle-continuation-v1.json','specialist-continuation-v1.json','specialist-idle-continuation-v1.json']:
                path=base/n
                if path.exists():
                    upstream=json.loads(path.read_text())
                    if 'failed' in upstream.get('status','') or upstream.get('errors'):
                        state.update(status='upstream_failed',upstream=upstream);save(state_path,state);return 1
            time.sleep(2)
        prefix=base/f'native-domain-{label}-v1'
        fitted=prefix.with_suffix('.json');energy=base/f'native-domain-{label}-energy-v1.json'
        temporal=base/f'native-domain-{label}-temporal-v1.json';replay=base/f'native-domain-{label}-replay-v1.json'
        item['status']='fitting';state['status']='fitting';save(state_path,state)
        argv=[]
        for directory in captures:argv+=['--capture',directory]
        argv+=['--pool',POOL,'--order-aware','--output',fitted]
        stages=[('fit_activation_native.py',argv,fitted),
                ('fit_activation_energy.py',['--fit',fitted,'--output',energy],energy),
                ('fit_activation_temporal.py',['--fit',energy,'--output',temporal,'--ridge','.1'],temporal),
                ('replay_activation_training.py',['--fit',temporal,'--output',replay],replay)]
        success=True
        for script,argv,target in stages:
            item.update(status=script,output=str(target));save(state_path,state)
            if not run(script,argv,target):
                item.update(status='command_failed',failed_script=script);state['errors'].append(str(target));success=False;break
        if success:
            f=json.loads(temporal.read_text());r=json.loads(replay.read_text())
            item.update(status='training_replay_passed_unvalidated' if r['training_replay_accuracy_within_threshold'] else 'training_replay_failed',
                        fit=str(temporal),replay=str(replay),reasons=r['reasons'],latency_errors=r['latency_errors'],
                        fit_diagnostics=[{k:d[k] for k in ('family','task','accepted','max_relative_work_error')}
                                         for d in f['profiles']['native_activity_law']['fit_diagnostics']],
                        idle_energy_fit_accepted=f.get('idle_energy_diagnostic',{}).get('accepted'))
        save(state_path,state)
    state.update(status='domain_development_stages_evaluated',formal_utility_qualified=False);save(state_path,state)
    return 0


if __name__=='__main__':raise SystemExit(main())
