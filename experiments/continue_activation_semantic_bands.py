"""Gate fresh semantic training, one declared fit and complete FIFO replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    args=parser.parse_args();protocol=json.loads(args.plan.read_text())
    state_path=Path(protocol['state'])
    if state_path.exists():raise FileExistsError(state_path)
    state=dict(status='waiting_for_observer_gate',formal_utility_qualified=False,
        pipeline_plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest())
    save(state_path,state)
    def verify():
        for source in protocol['source_receipts']:
            if any(hashlib.sha256(Path(source[key]).read_bytes()).hexdigest()!=source['sha256'] for key in ('path','snapshot')):
                raise ValueError('Frozen semantic pipeline source drift: '+source['path'])
        for name,expected in protocol['input_sha256'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
                raise ValueError('Frozen semantic pipeline input changed: '+name)
    def run(label,command):
        verify();state.update(status=label);save(state_path,state)
        with (Path(protocol['output'])/(label+'.log')).open('x') as log:
            subprocess.run([sys.executable,*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        verify()
    try:
        gate=Path(protocol['observer_report']);deadline=time.monotonic()+21600
        while not gate.exists():
            upstream_path=Path(protocol['observer_state'])
            upstream=json.loads(upstream_path.read_text()) if upstream_path.exists() else {}
            if upstream.get('errors') or 'failed' in upstream.get('status',''):
                state.update(status='not_run_observer_capture_failed',upstream=upstream);save(state_path,state);return
            if time.monotonic()>deadline:raise TimeoutError('Observer gate wait limit')
            time.sleep(2)
        report=json.loads(gate.read_text());state['observer_report_sha256']=hashlib.sha256(gate.read_bytes()).hexdigest()
        if report.get('observer_repeat_check_passed') is not True or report.get('issues'):
            state.update(status='not_run_observer_gate_failed',issues=report.get('issues'));save(state_path,state);return
        run('collecting_fresh_training',['experiments/run_activation_queue.py','--queue',protocol['queue']])
        queue=json.loads(Path(protocol['queue']).read_text())
        fit=Path(protocol['output'])/'fit.json'
        command=['experiments/fit_activation_semantic_bands.py','--observer-report',str(gate),
                 '--pool',protocol['pool'],'--output',str(fit)]
        for capture in queue['captures']:command+=['--capture',capture]
        run('fitting_declared_band_model',command)
        fitted=json.loads(fit.read_text());state['fit_sha256']=hashlib.sha256(fit.read_bytes()).hexdigest()
        if not fitted['training_fit_gates_passed']:
            state.update(status='training_fit_rejected',energy_accepted=fitted['energy_diagnostic']['accepted'],
                rejected_groups=[row for row in fitted['profiles']['derived_host_call_band_model']['progress_law']['fit_diagnostics'] if not row['accepted']])
            save(state_path,state);return
        replay=Path(protocol['output'])/'training-replay.json'
        run('replaying_full_training_fifo',['experiments/validate_activation_semantic_bands.py','--fit',str(fit),'--output',str(replay)])
        result=json.loads(replay.read_text())
        state.update(status='training_replay_passed_independent_validation_required' if result['training_replay_accuracy_within_threshold']
            else 'training_replay_rejected',reasons=result['reasons'],latency_errors=result['latency_errors'],
            independent_accuracy_qualified=False)
        save(state_path,state)
    except BaseException as exc:
        state.update(status='semantic_pipeline_failed',error=type(exc).__name__+':'+str(exc));save(state_path,state)
        raise


if __name__=='__main__':main()
