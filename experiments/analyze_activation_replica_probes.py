"""Describe extra-specialist pressure controls against matched four/five-model probes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.analyze_activation_shared_probes import summarize
from experiments.analyze_activation_probe_quality import describe
from experiments.analyze_activation_five_model_probes import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', action='append', type=Path, required=True)
    parser.add_argument('--five-model', action='append', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads((args.reference/'system-frozen.json').read_text())
    five = {}
    for directory in args.five_model:
        measured = summarize(directory, reference['latency_reference_ms'])
        quality = describe(directory, reference['catalog'])
        if measured['seed'] in five:
            raise ValueError('Duplicate five-model seed')
        five[measured['seed']] = (measured, {batch['id']: batch for batch in quality['batches']})
    runs = []
    for directory in args.capture:
        plan = json.loads((directory/'plan.json').read_text())
        if (plan['mode'] != 'replica_pressure_probe' or plan['not_for_training'] is not True
                or plan['resident_domain'] not in (dict(text=2,image=1,ocr=1,asr=1),
                                                   dict(text=1,image=1,ocr=1,asr=2))):
            raise ValueError('Declared specialist replica control required')
        control_path = Path(plan['matched_control_plan'])
        if hashlib.sha256(control_path.read_bytes()).hexdigest() != plan['matched_control_plan_sha256']:
            raise ValueError('Matched control plan changed')
        measured = summarize(directory, reference['latency_reference_ms'])
        quality = describe(directory, reference['catalog'])
        qualities = {batch['id']: batch for batch in quality['batches']}
        control = summarize(control_path.parent, reference['latency_reference_ms'])
        control_quality = describe(control_path.parent, reference['catalog'])
        controls = {batch['shared_trace_sha256']: batch for batch in control['batches']}
        control_qualities = {batch['shared_trace_sha256']: batch for batch in control_quality['batches']}
        five_measured, five_quality = five[measured['seed']]
        pairs = []
        for batch in measured['batches']:
            trace = batch['shared_trace_sha256']
            pairs.append(dict(id=batch['id'], shared_trace_sha256=trace,
                versus_four_model_specialists=compare(batch, controls[trace], qualities[batch['id']],
                                                     control_qualities[trace]),
                versus_five_model_coordinates=[compare(batch, other, qualities[batch['id']], five_quality[other['id']])
                    for other in five_measured['batches'] if other['shared_trace_sha256'] == trace]))
        runs.append(dict(measurements=measured, actual_output_quality=quality, comparisons=pairs))
    report = dict(schema='gsdeploy.specialist_replica_pressure_description.v1', runs=runs,
        formal_utility_qualified=False, capacity_qualified=False, U=None,
        limitations=['finite 45 second intervals and drain, not rho or steady-state qualification',
                     'replica measurements follow historical four/five-model controls; no counterbalanced chronology claim',
                     'whole episode startup energy and pressure normalization not included',
                     'both text and ASR duplicated together exceed the unchanged catalog memory constraint'])
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), runs=len(runs))))


if __name__ == '__main__':
    main()
