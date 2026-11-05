"""Run fresh observer checks, then gated full-pool baselines in one context.

No concurrent training or predictor qualification follows automatically from
these diagnostics. Failures and all raw captures are retained for inspection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import POOL, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('observer-base','solo-base','reference'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--solo-seed',type=int,required=True)
    a=p.parse_args();base=a.observer_base.resolve();solo=a.solo_base.resolve()
    if a.solo_seed in (6110490,6110491,6110492):raise ValueError('Reserved confirmation seed')
    if solo.exists():raise FileExistsError(solo)
    state_path=base/'state.json'
    if state_path.exists():raise FileExistsError(state_path)
    design=json.loads((base/'design.json').read_text())
    sources={r['path']:r['sha256'] for r in design['source_receipts']}
    names=['continue_activation_scoped_candidate.py','prepare_activation_scoped_solo.py',
        'analyze_activation_solo_repeats.py','analyze_activation_solo_quality.py',
        'analyze_activation_completion_template.py','fit_activation_completion_bands.py',
        'activation_completion_provider.py','fit_activation_declared_idle_energy.py']
    for name in names:
        path=ROOT/'experiments'/name;sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('design.json','observer-criteria.json','queue.json'):
        path=base/name;sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    reference=a.reference.resolve()/'system-frozen.json'
    sources[str(reference)]=hashlib.sha256(reference.read_bytes()).hexdigest()
    frozen=[]
    for i,(path,sha) in enumerate(sorted(sources.items())):
        snapshot=base/'pipeline-source-snapshot'/str(i)/Path(path).name
        snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(Path(path).read_bytes())
        frozen.append(dict(path=path,sha256=sha,snapshot=str(snapshot)))
    save(base/'pipeline-plan.json',dict(schema='gsdeploy.scoped_candidate_pipeline.v1',
        solo_output=str(solo),solo_seeds=[a.solo_seed,a.solo_seed+1],source_receipts=frozen,
        observer_gate_required=True,physical_predictor_qualified=False,formal_utility_qualified=False))
    state=dict(status='observer_collecting',physical_predictor_qualified=False,formal_utility_qualified=False)
    save(state_path,state)
    def check():
        for r in frozen:
            if any(hashlib.sha256(Path(r[k]).read_bytes()).hexdigest()!=r['sha256'] for k in ('path','snapshot')):
                raise ValueError('Frozen candidate pipeline source drift: '+r['path'])
    def run(script,*args):
        check();subprocess.run([sys.executable,str(ROOT/'experiments'/script),*map(str,args)],cwd=ROOT,check=True);check()
    try:
        run('run_activation_queue.py','--queue',base/'queue.json')
        state['status']='observer_analyzing';save(state_path,state)
        run('analyze_activation_scoped_observer.py','--base',base,'--output',base/'description.json')
        gate=json.loads((base/'description.json').read_text())
        if not gate['observer_repeat_check_passed'] or gate['issues']:
            state.update(status='observer_gate_rejected',issues=gate['issues']);save(state_path,state);return 2
        solo.mkdir(parents=True,exist_ok=False);captures=[]
        for i in (1,2):
            output=solo/f'solo-round{i}'
            run('prepare_activation_scoped_solo.py','--observer-report',base/'description.json',
                '--context-capture',base/'semantic-round1','--output',output,'--seed',a.solo_seed+i-1)
            captures.append(output)
        save(solo/'queue.json',dict(schema='gsdeploy.native_capture_queue.v1',state=str(solo/'continuation.json'),
            captures=list(map(str,captures))))
        state['status']='full_pool_solo_collecting';save(state_path,state)
        run('run_activation_queue.py','--queue',solo/'queue.json')
        state['status']='full_pool_solo_analyzing';save(state_path,state)
        capture_args=[v for capture in captures for v in ('--capture',capture)]
        run('analyze_activation_solo_repeats.py',*capture_args,'--output',solo/'description.json')
        run('analyze_activation_solo_quality.py',*capture_args,'--reference',a.reference.resolve(),
            '--output',solo/'quality-description.json')
        run('analyze_activation_completion_template.py',*capture_args,'--pool',POOL,
            '--output',solo/'completion-template-diagnostic.json')
        repeat=json.loads((solo/'description.json').read_text())
        state.update(status='full_pool_baseline_diagnostic_completed',
            changed_output_profiles=repeat['changed_output_profiles'],
            concurrent_training_pending=True,independent_predictor_validation_pending=True)
        save(state_path,state);return 0
    except BaseException as exc:
        state.update(status='candidate_pipeline_failed',error=type(exc).__name__+': '+str(exc))
        save(state_path,state);raise


if __name__=='__main__':raise SystemExit(main())
