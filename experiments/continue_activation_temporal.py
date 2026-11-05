"""Continue measured fits, executable replay, then frozen heldout GPU tests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True,type=Path)
    args=p.parse_args();base=args.base.resolve();state_path=base/'temporal-continuation-v1.json'
    if state_path.exists():raise FileExistsError(state_path)
    state=dict(status='waiting_for_controlled_captures',contexts=[],errors=[]);save(state_path,state)
    deadline=time.monotonic()+21600
    while True:
        path=base/'controlled-continuation-v1.json';upstream=json.loads(path.read_text()) if path.exists() else {}
        if upstream.get('status')=='capture_plan_completed':break
        if 'failed' in upstream.get('status','') or upstream.get('errors'):
            state.update(status='upstream_failed',upstream=upstream);save(state_path,state);return 1
        if time.monotonic()>deadline:raise TimeoutError('Controlled capture wait exceeded')
        time.sleep(2)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def run(script,argv,log):
        with log.open('x') as out:
            result=subprocess.run([sys.executable,str(ROOT/'experiments'/script),*map(str,argv)],
                                  cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,env=env)
        return result.returncode==0
    for resident,label,names,seeds in [
        ('g2-image-ocr1','controlled',('solo-round1','solo-round2','overlap-round1','overlap-round2','controlled-round1','controlled-round2'),(6110561,6110562)),
        ('g2-image-ocr2','ocr2',('solo-round1','solo-round2','overlap-round1','overlap-round2'),(6110571,6110572))]:
        item=dict(resident=resident,status='fitting');state['contexts'].append(item)
        state.update(status='fitting',current=resident);save(state_path,state)
        fitted=base/f'native-fit-{label}-v1.json';temporal=base/f'native-fit-{label}-temporal-v1.json'
        replay=base/f'native-{label}-training-replay-v1.json'
        argv=[]
        for name in names:argv+=['--capture',base/'training-v2'/resident/name]
        argv+=['--pool',ROOT/'artifacts/static-study-v6-20261021/inputs/runtime-inputs.jsonl','--order-aware','--output',fitted]
        if not run('fit_activation_native.py',argv,fitted.with_suffix('.log')):
            item['status']='fit_command_failed';state['errors'].append(str(fitted));save(state_path,state);continue
        if not run('fit_activation_temporal.py',['--fit',fitted,'--ridge','.1','--output',temporal],temporal.with_suffix('.log')):
            item['status']='temporal_command_failed';state['errors'].append(str(temporal));save(state_path,state);continue
        fit=json.loads(temporal.read_text());law=fit['profiles']['native_activity_law']
        diagnostics=law['fit_diagnostics'];order=fit['profiles'].get('native_order_law',{})
        item.update(fit=str(temporal),fit_groups=[{k:g[k] for k in ('family','task','accepted','max_relative_work_error')} for g in diagnostics])
        if (not diagnostics or any(not g['accepted'] for g in diagnostics)
                or any(not g['accepted'] for g in order.get('fit_diagnostics',[]))
                or not fit['resource_diagnostic']['accepted']):
            item['status']='requires_training_supplement';save(state_path,state);continue
        item['status']='replaying_training';state['status']='replaying_training';save(state_path,state)
        if not run('replay_activation_training.py',['--fit',temporal,'--output',replay],replay.with_suffix('.log')):
            item['status']='replay_command_failed';state['errors'].append(str(replay));save(state_path,state);continue
        report=json.loads(replay.read_text());item.update(training_replay=str(replay),latency_errors=report['latency_errors'],replay_reasons=report['reasons'])
        if not report['training_replay_accuracy_within_threshold']:
            item['status']='requires_training_supplement';save(state_path,state);continue
        validation=base/f'independent-{label}-v1'
        if not run('validate_activation_native.py',['prepare','--fit',temporal,'--reference',base/'references-v2',
            '--output',validation,'--seeds',*seeds],base/f'independent-{label}-prepare-v1.log'):
            item['status']='validation_prepare_failed';state['errors'].append(str(validation));save(state_path,state);continue
        item.update(status='collecting_independent_validation',validation=str(validation))
        state['status']='collecting_independent_validation';save(state_path,state)
        protocol=json.loads((validation/'protocol.json').read_text());collected=True
        for directory in map(Path,protocol['runs']):
            item['current_capture']=str(directory);save(state_path,state)
            if not run('run_activation_capture.py',['--output',directory],directory/'collector.log'):
                item['status']='validation_capture_failed';state['errors'].append(str(directory));collected=False;break
        if not collected:save(state_path,state);continue
        item['status']='scoring_independent_validation';state['status']=item['status'];save(state_path,state)
        if not run('validate_activation_native.py',['score','--output',validation],validation/'score.log'):
            item['status']='validation_score_failed';state['errors'].append(str(validation));save(state_path,state);continue
        result=json.loads((validation/'validation.json').read_text())
        item.update(status='development_accuracy_passed' if result['development_accuracy_passed'] else 'independent_validation_failed',
            validation_reasons=result['reasons'],validation_latency_errors=result['latency_errors'])
        save(state_path,state)
    state.update(status='development_stages_evaluated',current=None,formal_utility_qualified=False);save(state_path,state)
    return 0


if __name__=='__main__':raise SystemExit(main())
