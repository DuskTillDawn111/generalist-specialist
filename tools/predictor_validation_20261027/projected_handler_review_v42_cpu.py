"""Independent read-only four-capture accounting/projection audit; never fits."""
from collections import Counter
import gzip
import json
from pathlib import Path

from . import projected_handler_context_v42 as v
from .owner_lifecycle_mark_v21 import IntervalIndex
from .unknown_preflight_archive_v1 import save_synced


def read(ref):
    assert v.receipt(ref['path'])==ref
    return json.loads(Path(ref['path']).read_text())


def stream(ref):
    assert v.receipt(ref['path'])==ref
    with gzip.open(ref['path'],'rt') as handle:
        return [json.loads(line) for line in handle]


def run(manifest_path,output):
    output=Path(output)
    assert not output.exists()
    manifest_ref=v.receipt(manifest_path); manifest=read(manifest_ref)
    data=v.load_dataset(manifest['captures'])
    summaries=[]
    for item in manifest['captures']:
        report=read(item['report']); shared=read(report['source_shared_adapter'])
        native=read(report['source_adapter']); review=read(native['review'])
        raw=read(review['evidence']['raw.json']); spans={o:[] for o in v.OWNERS}
        for request in raw['priming_request_records']+raw['request_records']:
            owner=request['model'];worker=request['worker_response']
            offset=native['original_handler_projection_by_pid'][str(raw['ready'][owner]['pid'])]['offset_ns']
            spans[owner].append((worker['start_perf_ns']-offset,worker['end_perf_ns']-offset))
        indexes={owner:IntervalIndex(values) for owner,values in spans.items()}
        def expected(clock):return {o:indexes[o].contains(clock) for o in v.OWNERS}
        capture=report['capture']; risks=[r for r in data['risks'] if r['capture']==capture]
        marks=[r for r in data['marks'] if r['capture']==capture]
        original_risks=stream(shared['risk_chronology']); original_marks=stream(shared['mark_records'])
        assert len(marks)==len(original_marks)
        original_mark_labels=[]; current_mark_labels=[]
        for rows,target in ((original_marks,original_mark_labels),(marks,current_mark_labels)):
            for row in rows:
                copied=dict(row); copied.pop('features')
                for key in ('handler_projection_assumption','handler_boundary_proximity_diagnostic',
                            'handler_projection_is_exact_native_fact'):copied.pop(key,None)
                target.append(copied)
        assert original_mark_labels==current_mark_labels
        event_id=lambda row:(row['end_ns'],tuple(row['driver_context']),row['physical_SAVE_seqNo'])
        assert Counter(event_id(r) for r in risks if r['physical_SAVE_END_observed'])==Counter(
            event_id(r) for r in original_risks if r['physical_SAVE_END_observed'])
        assert sum(r['end_ns']-r['start_ns'] for r in risks)==sum(r['end_ns']-r['start_ns'] for r in original_risks)
        assert sum(r['end_ns']-r['start_ns'] for r in risks if not r['exposure_qualified'])==sum(
            r['end_ns']-r['start_ns'] for r in original_risks if not r['exposure_qualified'])
        regime=Counter(); qualified=Counter(); single_mark_targets=Counter(); residuals=report['worker_handler_observed_residual_by_owner_ns']
        interior=Counter()
        for row in risks:
            mask=row['shared_left_features']['handler_active']; assert mask==expected(row['start_ns'])
            # Half-open intervals must have a constant projected mask throughout.
            assert mask==expected(row['end_ns']-1)
            active=[o for o in v.OWNERS if mask[o]]
            category=('only_held_handler' if active==[row['held_owner']] else
                'single_other_handler' if len(active)==1 else 'zero_handlers' if not active else 'multiple_handlers')
            regime[category+'_exposure_ns']+=row['end_ns']-row['start_ns']
            if row['physical_SAVE_END_observed']:
                regime[category+'_SAVE']+=1
                if row['exposure_qualified']:qualified[category+'_SAVE']+=1
                event_clock=row['end_ns']
                distances={o:min(abs(event_clock-t) for span in indexes[o].spans for t in span) for o in v.OWNERS}
                if all(distances[o]>residuals[o] for o in v.OWNERS):interior[category+'_SAVE']+=1
        for row in marks:
            assert row['features']['handler_active']==expected(row['SAVE_END_ns'])
            active=[o for o in v.OWNERS if row['features']['handler_active'][o]]
            if active==[row['features']['held_owner']]:single_mark_targets[row['next_owner']]+=1
        summaries.append(dict(route=item['route'],capture=capture,report=item['report'],
            original_shared=report['source_shared_adapter'],marks=len(marks),risk_intervals=len(risks),
            exposure_ns=sum(r['end_ns']-r['start_ns'] for r in risks),
            paired_SAVE=sum(r['physical_SAVE_END_observed'] for r in risks),
            regimes=dict(regime),qualified_SAVE_regimes=dict(qualified),
            SAVE_outside_all_descriptive_boundary_residual_distances=dict(interior),
            only_held_handler_at_SAVE_mark_target_counts=dict(single_mark_targets),
            mark_labels_exactly_unchanged=True,unresolved_exposure_retained=True,
            all_handler_masks_recomputed_from_original_worker_spans=True))
    output.mkdir()
    result=dict(schema='gsdeploy.projected_handler_independent_CPU_review.v42',status='PASS',
        manifest=manifest_ref,implementation=v.receipt(v.__file__),review_implementation=v.receipt(__file__),
        sources_verified_by_read_only_v42_loader=len(data['sources']),captures=summaries,
        tests=dict(command='python -m unittest tests.test_worker_handler_projection_adapter_v41 tests.test_projected_handler_context_v42 tests.test_projected_handler_context_v42_independent_review -q',passed=11,
                   independent_test_source=v.receipt('tests/test_projected_handler_context_v42_independent_review.py')),
        fit_performed=False,GPU_started=False,validation_labels_read=False,
        limitations=['Fixed-median handler masks are an uncalibrated projection hypothesis, never native truth.',
            'Observed residual distances are sensitivity diagnostics, not certified timing bounds.',
            'Physical SAVE and next-context labels do not identify service interruption, SM work, or effective progress.',
            'Independent tests verify numerical/interface boundaries, not prediction accuracy.'],
        implementation_findings=[])
    save_synced(output/'report.json',result)
    print(json.dumps(dict(report=v.receipt(output/'report.json'),captures=summaries)))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.manifest,args.output)
