"""Score actual finite-probe outputs with frozen references; never calculate formal U."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.fit_activation_native import load_bundle
from gsdeploy.quality import quality_digest, sample_totals
from gsdeploy.normalization import quality_loss
from gsdeploy.tasks import page_ocr_text, score_result


def describe(directory, data):
    bundle = load_bundle(directory)
    raw = bundle['raw']
    if (raw['status'] != 'completed' or raw['split'] != 'development_shared_pressure'
            or any(raw.get(k) for k in ('errors', 'runtime_errors', 'telemetry_errors'))):
        raise ValueError('Complete clean probe output cohort required')
    quality = data['quality']
    if quality_digest(quality['references']) != quality['identity']['reference_sha256']:
        raise ValueError('Frozen quality references changed')
    rows = {r['request_id']: r for r in raw['request_records']}
    if len(rows) != len(raw['request_records']):
        raise ValueError('Duplicate probe output identity')
    batches, seen = [], set()
    for job, batch in zip(bundle['plan']['jobs'], raw['batches']):
        if job['id'] != batch['id'] or seen & set(batch['request_ids']):
            raise ValueError('Probe batch identity or cohort changed')
        seen.update(batch['request_ids'])
        totals = defaultdict(lambda: [0., 0., 0])
        by_model = defaultdict(lambda: [0., 0., 0])
        scores = []
        for rid in batch['request_ids']:
            row = rows[rid]
            task = row['task']
            family = raw['resident_aliases'][row['model']]
            result = dict(row['worker_response'])
            if row['status'] != 'completed' or result.get('success') is not True:
                raise ValueError('Failed or unverified output in probe')
            result['task'] = result.get('task', result.get('task_type'))
            if task == 'ocr':
                result['prediction'] = page_ocr_text(result['prediction'])
            reference = quality['references'][task][row['pool_id']]
            score = score_result(reference, result)
            a, z = sample_totals(task, score)
            for target in (totals[task], by_model[(family, task)]):
                target[0] += a
                target[1] += z
                target[2] += 1
            scores.append(dict(request_id=rid, task=task, family=family, pool_id=row['pool_id'],
                               error_numerator=a, reference_denominator=z))
        def metric(task, values):
            a, z, n = values
            return dict(requests=n, error_numerator=a, reference_denominator=z,
                        native_loss=a/z, normalized_loss=quality_loss(data, task, a/z))
        per_task = {t: metric(t, v) for t, v in totals.items()}
        rates = job['rates']
        normalized = sum(rates[t]/sum(rates.values())*v['normalized_loss'] for t, v in per_task.items())
        batches.append(dict(id=job['id'], shared_trace_sha256=job['shared_trace_sha256'],
            per_task=per_task, per_model_task={f+'/'+t: metric(t, v) for (f, t), v in by_model.items()},
            observed_probe_Q_norm=normalized, requests=scores))
    if seen != set(rows) or len(batches) != len(bundle['plan']['jobs']):
        raise ValueError('Missing or unassigned probe output cohort')
    return dict(source=bundle['source'], batches=batches)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', action='append', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = json.loads((args.reference/'system-frozen.json').read_text())['catalog']
    runs = [describe(c.resolve(), data) for c in args.capture]
    report = dict(schema='gsdeploy.actual_probe_output_quality.v1', runs=runs,
        reference_identity=data['quality']['identity'], U=None, formal_utility_qualified=False,
        scope='actual finite-probe output quality against unchanged references and anchors; no energy denominator')
    with args.output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(output=str(args.output), runs=len(runs))))


if __name__ == '__main__':
    main()
