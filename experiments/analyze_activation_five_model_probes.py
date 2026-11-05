"""Compare five-model offload to its resident-zero and matched four-model controls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.analyze_activation_shared_probes import summarize
from experiments.analyze_activation_probe_quality import describe


def compare(candidate, baseline, candidate_quality, baseline_quality):
    if candidate['shared_trace_sha256'] != baseline['shared_trace_sha256']:
        raise ValueError('Different offered cohorts cannot form a matched pair')
    return dict(
        baseline_id=baseline['id'],
        delta_arrival_through_drain_joules=(candidate['arrival_through_drain_joules']
                                           - baseline['arrival_through_drain_joules']),
        delta_mean_e2e_ms={t: candidate['task_metrics'][t]['e2e_ms']['mean']
                          - baseline['task_metrics'][t]['e2e_ms']['mean']
                          for t in candidate['task_metrics']},
        delta_deadline_miss_fraction={t: candidate['task_metrics'][t]['observed_deadline_miss_fraction']
                                     - baseline['task_metrics'][t]['observed_deadline_miss_fraction']
                                     for t in candidate['task_metrics']},
        delta_native_quality_loss={t: candidate_quality['per_task'][t]['native_loss']
                                   - baseline_quality['per_task'][t]['native_loss']
                                   for t in candidate_quality['per_task']},
        delta_probe_Q_norm=(candidate_quality['observed_probe_Q_norm']
                            - baseline_quality['observed_probe_Q_norm']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', action='append', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    reference = json.loads((args.reference/'system-frozen.json').read_text())
    runs = []
    for directory in args.capture:
        plan = json.loads((directory/'plan.json').read_text())
        control_path = Path(plan['matched_control_plan'])
        if hashlib.sha256(control_path.read_bytes()).hexdigest() != plan['matched_control_plan_sha256']:
            raise ValueError('Original matched control plan changed')
        if (plan['resident_domain'] != dict(generalist_2=1, text=1, image=1, ocr=1, asr=1)
                or plan.get('not_for_training') is not True
                or plan.get('all_specialists_remain_resident') is not True):
            raise ValueError('Declared five-model diagnostic domain required')
        measured = summarize(directory, reference['latency_reference_ms'])
        quality = describe(directory, reference['catalog'])
        control = summarize(control_path.parent, reference['latency_reference_ms'])
        control_quality = describe(control_path.parent, reference['catalog'])
        jobs = {job['id']: job for job in plan['jobs']}
        qualities = {batch['id']: batch for batch in quality['batches']}
        controls = {batch['shared_trace_sha256']: batch for batch in control['batches']}
        control_qualities = {batch['shared_trace_sha256']: batch for batch in control_quality['batches']}
        zeros = {batch['shared_trace_sha256']: batch for batch in measured['batches']
                 if jobs[batch['id']]['text_generalist_share'] == 0
                 and jobs[batch['id']]['asr_generalist_share'] == 0}
        pairs = []
        for batch in measured['batches']:
            job = jobs[batch['id']]
            trace = batch['shared_trace_sha256']
            zero = zeros[trace]
            batch['text_generalist_share'] = job['text_generalist_share']
            pairs.append(dict(id=batch['id'], shared_trace_sha256=trace,
                text_generalist_share=job['text_generalist_share'],
                asr_generalist_share=job['asr_generalist_share'],
                versus_five_model_zero=compare(batch, zero, qualities[batch['id']], qualities[zero['id']]),
                versus_four_model_specialists=compare(batch, controls[trace], qualities[batch['id']],
                                                     control_qualities[trace])))
        runs.append(dict(measurements=measured, actual_output_quality=quality,
                         matched_control=control, matched_control_quality=control_quality,
                         comparisons=pairs))
    report = dict(schema='gsdeploy.five_model_pressure_description.v1', runs=runs,
        formal_utility_qualified=False, capacity_qualified=False, predictor_qualified=False, U=None,
        limitations=['45 second offered intervals, with drain; no steady-state or rho certificate',
                     'five-model zero offload includes idle G2 residency',
                     'historical four-model controls precede these runs; chronology is not counterbalanced',
                     'within-run coordinate order is shuffled but initial thermal/clock states are not identical',
                     'matched cohorts include routing uniforms; actual route fractions vary with finite sampling',
                     'whole episode startup and matching pressure normalization remain required'])
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), runs=len(runs))))


if __name__ == '__main__':
    main()
