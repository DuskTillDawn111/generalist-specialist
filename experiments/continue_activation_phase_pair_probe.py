"""Continue from accepted scoped baselines into a small concurrent training probe.

All later owner/input domains and independent qualification remain required.
This controller preserves failed fits and never schedules independent data
before the complete matched training replay passes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import POOL, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('observer-base','solo-base','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--upstream-pid',type=int,required=True)
    a=p.parse_args();observer=a.observer_base.resolve();solo=a.solo_base.resolve();base=a.output.resolve()
    if any(s in (6110491,6110492) for s in range(a.seed,a.seed+3)):raise ValueError('Reserved confirmation seeds')
    base.mkdir(parents=True,exist_ok=False);state_path=base/'state.json'
    upstream_plan=json.loads((observer/'pipeline-plan.json').read_text())
    sources={r['path']:r['sha256'] for r in upstream_plan['source_receipts']}
    for name in ('continue_activation_phase_pair_probe.py','prepare_activation_phase_pair_probe.py',
                 'activation_phase_crossed_design.py','audit_activation_band_support.py',
                 'validate_activation_completion_bands.py','fit_activation_semantic_bands_v2.py'):
        path=ROOT/'experiments'/name;sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    frozen=[]
    for i,(path,sha) in enumerate(sorted(sources.items())):
        snapshot=base/'source-snapshot'/str(i)/Path(path).name;snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes(Path(path).read_bytes());frozen.append(dict(path=path,snapshot=str(snapshot),sha256=sha))
    save(base/'pending-plan.json',dict(schema='gsdeploy.phase_pair_probe_pipeline.v1',source_receipts=frozen,
        upstream_state=str(observer/'state.json'),upstream_pid=a.upstream_pid,solo_base=str(solo),
        seeds=list(range(a.seed,a.seed+3)),observer_and_baseline_gates_required=True,
        high_order_and_input_domains_pending=True,independent_validation_scheduled=False,
        physical_predictor_qualified=False,formal_utility_qualified=False))
    state=dict(status='waiting_for_matched_full_pool_baselines',physical_predictor_qualified=False,formal_utility_qualified=False)
    save(state_path,state)
    def check():
        for r in frozen:
            if any(hashlib.sha256(Path(r[k]).read_bytes()).hexdigest()!=r['sha256'] for k in ('path','snapshot')):
                raise ValueError('Frozen pair-probe pipeline source drift: '+r['path'])
    def run(script,*args):
        check();subprocess.run([sys.executable,str(ROOT/'experiments'/script),*map(str,args)],cwd=ROOT,check=True);check()
    try:
        check()
        while True:
            upstream=json.loads((observer/'state.json').read_text())
            status=upstream['status']
            if status=='full_pool_baseline_diagnostic_completed':break
            if 'failed' in status or 'rejected' in status:
                state.update(status='upstream_rejected',upstream=upstream);save(state_path,state);return 2
            proc=Path('/proc')/str(a.upstream_pid)/'cmdline'
            if not proc.exists() or b'continue_activation_scoped_candidate.py' not in proc.read_bytes():
                # Recheck the terminal state after observing process exit.
                upstream=json.loads((observer/'state.json').read_text())
                if upstream['status']=='full_pool_baseline_diagnostic_completed':break
                raise RuntimeError('Upstream controller exited before successful baseline completion')
            time.sleep(2)
        state['status']='preparing_phase_pair_probe';save(state_path,state)
        captures=base/'captures'
        run('prepare_activation_phase_pair_probe.py','--solo-base',solo,'--observer-report',observer/'description.json',
            '--output',captures,'--seed',a.seed)
        state['status']='phase_pair_probe_collecting';save(state_path,state)
        run('run_activation_queue.py','--queue',captures/'queue.json')
        state['status']='phase_pair_probe_fitting';save(state_path,state)
        directories=[solo/f'solo-round{i}' for i in (1,2)]+[captures/f'phase-pair-round{i}' for i in (1,2,3)]
        args=[x for directory in directories for x in ('--capture',directory)]
        run('fit_activation_completion_bands.py',*args,'--observer-report',observer/'description.json',
            '--pool',POOL,'--output',base/'fit.json')
        run('audit_activation_band_support.py','--fit',base/'fit.json','--output',base/'support-audit.json')
        fit=json.loads((base/'fit.json').read_text())
        if not fit['training_fit_gates_passed']:
            state.update(status='phase_pair_training_fit_rejected',
                rejected_groups=[g for g in fit['profiles']['derived_host_call_band_model']['progress_law']['fit_diagnostics'] if not g['accepted']],
                energy_diagnostic=fit['energy_diagnostic']);save(state_path,state);return 2
        state['status']='phase_pair_training_replaying';save(state_path,state)
        run('validate_activation_completion_bands.py','--fit',base/'fit.json','--output',base/'training-replay.json')
        replay=json.loads((base/'training-replay.json').read_text())
        state.update(status='phase_pair_training_replay_completed' if replay['training_replay_accuracy_within_threshold'] else 'phase_pair_training_replay_rejected',
            replay_reasons=replay['reasons'],higher_order_and_input_domains_pending=True,independent_validation_pending=True)
        save(state_path,state);return 0 if replay['training_replay_accuracy_within_threshold'] else 2
    except BaseException as exc:
        state.update(status='phase_pair_pipeline_failed',error=type(exc).__name__+': '+str(exc));save(state_path,state);raise


if __name__=='__main__':raise SystemExit(main())
