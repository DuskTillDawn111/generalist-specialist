"""Collect and score the prospectively sealed known-input validation once."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import save
from experiments.fit_activation_native import load_bundle
from experiments.replay_activation_aggregate import score_prediction


def assess_cells(report, plan, policy):
    thresholds = policy['thresholds']
    reasons = []
    if not report['complete_cohort']:
        reasons.append('INCOMPLETE_COHORT')
    for job in plan['jobs']:
        for task in ('asr', 'image', 'ocr', 'text'):
            key = job['id']+'/task:'+task
            cell = report['aggregate_metrics'].get(key)
            if cell is None or cell['n'] != job['design']['offered_per_task']:
                reasons.append(key+':MISSING_OR_CHANGED_COHORT')
            elif not cell['descriptive_within_latency_tolerances']:
                reasons.append(key+':LATENCY_TOLERANCE')
    for cell in report['throughput']:
        if cell['relative_error'] is None or cell['relative_error'] > thresholds['fixed_window_throughput_relative_error_max']:
            reasons.append(cell['job_id']+':THROUGHPUT_TOLERANCE')
    if len(report['throughput']) != len(plan['jobs']):
        reasons.append('MISSING_THROUGHPUT_CELLS')
    energy = report['episode']['relative_energy_error']
    if energy is None or energy > thresholds['full_gpu_board_episode_energy_relative_error_max']:
        reasons.append('EPISODE_ENERGY_TOLERANCE')
    return reasons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base','solo-base','reference'):
        p.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args()
    base = a.base.resolve()
    protocol = json.loads((base/'protocol.json').read_text())
    policy = json.loads((base/'policy-frozen.json').read_text())
    state = dict(status='checking_sealed_validation', physical_predictor_qualified=False,
                 formal_utility_qualified=False)
    if (base/'state.json').exists():
        raise FileExistsError('Validation controller already started')
    save(base/'state.json', state)
    def check():
        for name, key in [('fit-frozen.json','fit_sha256'), ('policy-frozen.json','policy_sha256'), ('queue.json','queue_sha256')]:
            if hashlib.sha256((base/name).read_bytes()).hexdigest() != protocol[key]:
                raise ValueError('Frozen validation artifact changed: '+name)
        for r in protocol['source_receipts']:
            if any(hashlib.sha256(Path(r[k]).read_bytes()).hexdigest() != r['sha256'] for k in ('path','snapshot')):
                raise ValueError('Validation executable source changed: '+r['path'])
        for directory, receipt in zip(protocol['captures'], protocol['predictions']):
            for name, key in [('plan.json','plan_sha256'),('prediction-sealed.json','prediction_sha256')]:
                if hashlib.sha256((Path(directory)/name).read_bytes()).hexdigest() != receipt[key]:
                    raise ValueError('Sealed request plan/predictions changed')
    try:
        check()
        state['status'] = 'collecting_two_frozen_history_runs'
        save(base/'state.json', state)
        with (base/'queue.log').open('x') as stream:
            subprocess.run([sys.executable, str(ROOT/'experiments/run_activation_queue.py'), '--queue', str(base/'queue.json')],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        check()
        state['status'] = 'scoring_sealed_predictions_and_outputs'
        save(base/'state.json', state)
        reports, reasons = [], []
        for directory in map(Path, protocol['captures']):
            hardware = json.loads((directory/'hardware.json').read_text())
            if hardware['started_unix'] * 1e9 <= protocol['prediction_sealed_unix_ns']:
                raise ValueError('Collection started before prospective prediction seal')
            bundle = load_bundle(directory)
            if bundle['raw']['split'] != 'development_validation':
                raise ValueError('Heldout history split changed')
            prediction = json.loads((directory/'prediction-sealed.json').read_text())
            report = score_prediction(prediction, bundle, policy)
            report['tolerance_reasons'] = assess_cells(report, bundle['plan'], policy)
            reports.append(report)
            reasons.extend(report['run_id']+':'+reason for reason in report['tolerance_reasons'])
        audit = base/'capture-audit.json'
        command = [sys.executable, str(ROOT/'experiments/audit_activation_aggregate_capture.py'),
            '--baseline', str(a.solo_base.resolve()/'solo-round1'), '--baseline', str(a.solo_base.resolve()/'solo-round2'),
            '--reference', str(a.reference.resolve()), '--output', str(audit)]
        for directory in protocol['captures']:
            command.extend(['--capture', directory])
        with (base/'audit.log').open('x') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        quality = json.loads(audit.read_text())
        for run in quality['runs']:
            reasons.extend(run['run_id']+':'+issue for issue in run['issues'])
        check()
        save(base/'validation.json', dict(schema='gsdeploy.aggregate_known_input_history_validation.v1',
            protocol_sha256=hashlib.sha256((base/'protocol.json').read_bytes()).hexdigest(),
            reports=reports, reasons=reasons, known_input_history_accuracy_within_tolerances=not reasons,
            execution_integrity_passed=all(not r['issues'] for r in quality['runs']),
            input_overlap_with_training=True, independent_input_qualification=False,
            inference_scope='two predeclared new arrival/routing histories, 24 inputs/task/routing/run; descriptive, no population confidence guarantee',
            physical_predictor_qualified=False, formal_utility_qualified=False, candidate_ranking_qualified=False))
        state.update(status='known_input_history_validation_completed',
                     accuracy_within_tolerances=not reasons, reasons=reasons)
        save(base/'state.json', state)
    except BaseException as exc:
        state.update(status='known_input_history_validation_failed', error=type(exc).__name__+': '+str(exc))
        save(base/'state.json', state)
        raise


if __name__ == '__main__':
    main()
