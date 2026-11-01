"""Supplement the fixed v57 arithmetic audit with exact metadata identities."""
import argparse
import json
from pathlib import Path

from .native_collect import receipt
from .projected_state_save_hazard_v57 import PARAMETERS, SCHEMA, ORIGINAL_MANIFEST_SHA256
from .unknown_preflight_archive_v1 import save_synced


def run(model_path, report_path, review_path, output):
    refs = [receipt(p) for p in (model_path, report_path, review_path, __file__)]
    model, report, review = [json.loads(Path(r['path']).read_text()) for r in refs[:3]]
    if (review['status'] != 'PASS' or review['model'] != refs[0] or review['author_report'] != refs[1]
            or model['schema'] != SCHEMA or model['parameters'] != PARAMETERS
            or model['original_training_manifest']['sha256'] != ORIGINAL_MANIFEST_SHA256
            or receipt(model['original_training_manifest']['path']) != model['original_training_manifest']
            or report['by_state'] != model['fitted'] or report['summary'] != model['coverage']):
        raise ValueError('Exact original parameters, source manifest and reported model metadata required')
    folds = report['leave_one_whole_capture_out']
    heldout = [f['held_out_capture'] for f in folds]
    if len(heldout) != 4 or set(heldout) != set(model['captures']) or len(set(heldout)) != 4:
        raise ValueError('Exactly one distinct holdout per original capture required')
    values = model['fitted']; coverage = model['coverage']
    expected = dict(strata=64, exposed_strata=sum(v['exposure_ns'] > 0 for v in values.values()),
        positive_event_strata=sum(v['events'] > 0 for v in values.values()),
        zero_event_exposed_strata=sum(v['exposure_ns'] > 0 and v['events'] == 0 for v in values.values()),
        unseen_strata=[k for k,v in values.items() if not v['exposure_ns']],
        totals={k:sum(v[k] for v in values.values()) for k in
                ('rows','exposure_ns','events','unresolved_rows','unresolved_exposure_ns','unresolved_events')},
        by_holder={o:dict(exposed_masks=[k.split(':')[1] for k,v in values.items() if k.startswith(o+':') and v['exposure_ns']],
            unseen_masks=[k.split(':')[1] for k,v in values.items() if k.startswith(o+':') and not v['exposure_ns']])
            for o in model['parameters']['mask_bit_order']})
    # JSON sort_keys may reorder dictionary entries. Lists encode sets only
    # for these coverage members; all actual numbers are checked exactly.
    def normalize(value):
        if isinstance(value,dict):return {k:normalize(v) for k,v in value.items()}
        if isinstance(value,list):return sorted(value)
        return value
    if normalize(expected) != normalize(coverage):
        raise ValueError('All independently calculated coverage fields must agree')
    for ref in refs:
        if receipt(ref['path']) != ref:raise ValueError('Supplement source drift')
    out = Path(output); out.mkdir(parents=True,exist_ok=False)
    result = dict(schema='gsdeploy.projected_state_v57_actual_review_metadata_supplement.v1',
        status='PASS', sources=refs, actual_model=refs[0], actual_arithmetic_review=refs[2],
        original_parameters_and_training_manifest_exact=True, four_unique_complete_holdouts=True,
        reported_by_state_and_full_coverage_exact=True, independent_prediction_accuracy_validated=False,
        GPU_started=False, validation_labels_read=False)
    save_synced(out/'report.json',result); return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','report','review','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(run(a.model,a.report,a.review,a.output)))
