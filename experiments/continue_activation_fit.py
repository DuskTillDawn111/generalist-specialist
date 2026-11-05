"""Fit and independently check the initial physical domain after queued captures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True,type=Path)
    args=p.parse_args();base=args.base.resolve();state=base/'fit-continuation-v1.json'
    if state.exists():
        status=json.loads(state.read_text())
        if status['status']!='waiting_for_initial_captures':raise FileExistsError(state)
        status['restart_while_waiting_unix']=time.time()
    else:status=dict(status='waiting_for_initial_captures',errors=[])
    save(state,status)
    deadline=time.monotonic()+21600
    while True:
        initial=json.loads((base/'continuation-v1.json').read_text())
        if initial['status']=='initial_capture_plan_completed':break
        if initial['status'] in {'capture_failed','reference_capture_failed'}:
            status.update(status='upstream_failed',upstream=initial);save(state,status);return 1
        if time.monotonic()>deadline:raise TimeoutError('Initial capture wait budget exhausted')
        time.sleep(2)
    def execute(name,arguments):
        status.update(status=name);save(state,status)
        with (base/(name+'.log')).open('x') as log:
            result=subprocess.run([sys.executable,*map(str,arguments)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(name+' exited '+str(result.returncode))
    try:
        root=base/'training-v2';capacity=[]
        for suffix in ('','-round2'):
            for task in ('ocr','image'):
                for n in (1,2):capacity.extend(['--capture',root/f'g2-image-ocr{n}'/f'capacity-{task}{suffix}'])
        execute('capacity_analysis_v1',[ROOT/'experiments/analyze_activation_capacity.py',*capacity,
                                        '--output',base/'capacity-summary-v1.json'])
        captures=[]
        for name in ('solo-round1','solo-round2','overlap-round1','overlap-round2'):
            captures.extend(['--capture',root/'g2-image-ocr1'/name])
        fit_path=base/'native-fit-v1.json'
        execute('native_fit_v1',[ROOT/'experiments/fit_activation_native.py',*captures,
            '--pool',ROOT/'artifacts/static-study-v6-20261021/inputs/runtime-inputs.jsonl','--output',fit_path])
        fit=json.loads(fit_path.read_text());law=fit['profiles']['native_activity_law']
        rejected=[g for g in law['fit_diagnostics'] if not g['accepted']]
        status['training_fit_rejected_groups']=rejected;status['resource_diagnostic']=fit['resource_diagnostic']
        if rejected or not fit['resource_diagnostic']['accepted']:
            status.update(status='requires_training_supplement',admission_enabled=False);save(state,status);return 0
        validation=base/'native-validation-v1'
        execute('validation_prepare_v1',[ROOT/'experiments/validate_activation_native.py','prepare',
            '--output',validation,'--fit',fit_path,'--reference',base/'references-v2'])
        for index in (1,2):
            execute(f'validation_collect_v1_round{index}',[ROOT/'experiments/run_activation_capture.py',
                '--output',validation/f'validation-round{index}'])
        execute('validation_score_v1',[ROOT/'experiments/validate_activation_native.py','score','--output',validation])
        report=json.loads((validation/'validation.json').read_text())
        status.update(status='initial_development_validation_completed',
            development_accuracy_passed=report['development_accuracy_passed'],reasons=report['reasons'],
            admission_enabled=False,formal_utility_qualified=False)
        save(state,status);return 0
    except Exception as exc:
        status.update(status='failed');status['errors'].append(repr(exc));save(state,status);raise


if __name__=='__main__':raise SystemExit(main())
