"""Complete locked analysis with descriptive revision metadata for controls.

The frozen analyzer omitted revision when constructing its two latency-only
controls. No measurement, estimator, admission, selection, or SLO change.
"""
from module14.common import OUT, RAW, read, save, now, ref
from module14 import analyze


def main():
    assert not (OUT / 'hq_quality_slo.json').exists()
    baseline = read(OUT / 'baseline_references.json')['tasks']
    original_aggregate = analyze.aggregate

    def aggregate_with_control_revision(entry, factor):
        if 'revision' not in entry:
            assert (entry['task'], entry['model_id']) in {
                ('image', 'resnet50'), ('ocr', 'paddleocr')
            }
            entry = {**entry, 'revision': baseline[entry['task']]['revision']}
        return original_aggregate(entry, factor)

    runs = sorted(RAW.glob('*summary.json'))
    assert len(runs) == 30
    assert all(read(p)['success'] for p in runs)
    save('postprocessing_compatibility_amendment.json', {
        'at_utc': now(),
        'reason': 'Locked analyzer omitted descriptive revision for two latency-only baseline controls; KeyError before selection/SLO write',
        'failed_analysis_log': ref(OUT / 'analysis_progress.log'),
        'runner': ref(__file__),
        'frozen_analyzer_unchanged': ref(analyze.__file__),
        'raw_gpu_summaries_before': [ref(p) for p in runs],
        'change': 'Supply only existing frozen baseline revision to sparse control dictionaries; no numeric field or evaluation change',
        'measurement_repeated': False,
        'checkpoint_preprocessing_estimator_selection_SLO_rule_unchanged': True,
        'HQ_SLO_had_been_written': False,
    })
    analyze.aggregate = aggregate_with_control_revision
    analyze.main()
    amendment = read(OUT / 'postprocessing_compatibility_amendment.json')
    assert all(ref(r['path'])['sha256'] == r['sha256']
               for r in amendment['raw_gpu_summaries_before'])


if __name__ == '__main__':
    main()
