"""Public input-size strata and measured specialist/generalist output complementarity."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import POOL
from experiments.fit_activation_native import load_bundle
from gsdeploy.forecasting.native_training import validate_capture
from gsdeploy.quality import sample_totals
from gsdeploy.tasks import page_ocr_text, score_result


def public_size(item):
    task, sample = item['task_type'], item['sample']
    if task == 'text':
        return len(('\n'.join(str(sample.get(k, '')) for k in ('context', 'question'))).encode('utf-8'))
    if task == 'asr':
        # Stored paths can retain .wav while the actual container is FLAC.
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'json', sample['input']], check=True, capture_output=True, text=True, timeout=10)
        return float(json.loads(result.stdout)['format']['duration'])
    from PIL import Image
    with Image.open(sample['input']) as image:
        return image.width*image.height


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    reference = json.loads((args.base/'references-v2/system-frozen.json').read_text())
    pool = [json.loads(line) for line in POOL.read_text().splitlines()]
    rows = defaultdict(list)
    features = {r['pool_id']: public_size(r) for r in pool}
    by_task = defaultdict(list)
    for r in pool:
        by_task[r['task_type']].append(r['pool_id'])
    # Assign strata before reading any model outputs or service times.
    strata = {}
    for task, inputs in by_task.items():
        ordered = sorted(inputs, key=lambda k: (features[k], k))
        for i, pool_id in enumerate(ordered):
            strata[pool_id] = 'lower_size_half' if i < len(inputs)//2 else 'upper_size_half'
    sources = []
    for resident in ['g2-image-ocr1', 'specialists-ocr1']:
        for round_no in (1, 2):
            bundle = load_bundle(args.base/'training-v2'/resident/f'solo-round{round_no}')
            validate_capture(bundle)
            if bundle['plan']['mode'] != 'solo':
                raise ValueError('Repeated full-input solo measurements required')
            raw = bundle['raw']
            sources.append(bundle['source'])
            for row in raw['request_records']:
                task = row['task']
                family = raw['resident_aliases'][row['model']]
                if resident.startswith('g2') and family != 'generalist_2':
                    continue
                result = dict(row['worker_response'])
                if row['status'] != 'completed' or result.get('success') is not True:
                    raise ValueError('Complete verified actual output required')
                result['task'] = result.get('task', result.get('task_type'))
                if task == 'ocr':
                    result['prediction'] = page_ocr_text(result['prediction'])
                score = score_result(reference['catalog']['quality']['references'][task][row['pool_id']], result)
                numerator, denominator = sample_totals(task, score)
                rows[(task, row['pool_id'], family)].append(dict(
                    run_id=raw['run_id'], error_numerator=numerator, reference_denominator=denominator,
                    handler_ms=(result['end_perf_ns']-result['start_perf_ns'])/1e6))
    summaries, inputs = {}, []
    for task, ids in by_task.items():
        strata_summary = {}
        for stratum in ['all', 'lower_size_half', 'upper_size_half']:
            selected = [k for k in ids if stratum == 'all' or strata[k] == stratum]
            methods = {}
            for family in [task, 'generalist_2']:
                cohort = [r for k in selected for r in rows[(task, k, family)]]
                if len(cohort) != 2*len(selected):
                    raise ValueError('Every model/input requires both independent solo rounds')
                methods[family] = dict(inputs=len(selected), observations=len(cohort),
                    native_quality_loss=sum(r['error_numerator'] for r in cohort)/sum(r['reference_denominator'] for r in cohort),
                    mean_handler_ms=statistics.mean(r['handler_ms'] for r in cohort),
                    public_size_range=[min(features[k] for k in selected), max(features[k] for k in selected)])
            wins = dict(generalist_lower_loss=0, specialist_lower_loss=0, equal_loss=0)
            for k in selected:
                losses = {}
                for family in [task, 'generalist_2']:
                    cohort = rows[(task, k, family)]
                    losses[family] = sum(r['error_numerator'] for r in cohort)/sum(r['reference_denominator'] for r in cohort)
                key = ('generalist_lower_loss' if losses['generalist_2'] < losses[task]-1e-12 else
                       'specialist_lower_loss' if losses[task] < losses['generalist_2']-1e-12 else 'equal_loss')
                wins[key] += 1
            strata_summary[stratum] = dict(methods=methods, per_input_quality_complementarity=wins)
        summaries[task] = strata_summary
        for k in ids:
            inputs.append(dict(task=task, pool_id=k, public_size=features[k], stratum=strata[k],
                               observations={family: rows[(task, k, family)] for family in [task, 'generalist_2']}))
    report = dict(schema='gsdeploy.public_input_stratum_diagnosis.v1', sources=sources,
        input_pool_sha256=hashlib.sha256(POOL.read_bytes()).hexdigest(),
        public_size_descriptors=dict(text='UTF-8 context/question bytes', asr='ffprobe container duration seconds',
                                     image='width * height pixels', ocr='width * height pixels'),
        strata_policy='equal rank halves from public input size before loading outputs; ties broken by pool ID',
        summaries=summaries, inputs=inputs, formal_utility_qualified=False,
        limitations=['public size is a descriptor, not a semantic difficulty label',
                    'solo calls use different resident domains and priming histories; between-model timings do not isolate model effects',
                    'quality complementarity uses eventual outputs and must not become an online routing oracle',
                    'no input, anchor, utility weight, or scenario is changed by this diagnostic'])
    with args.output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(output=str(args.output), summaries=summaries), ensure_ascii=False))


if __name__ == '__main__':
    main()
