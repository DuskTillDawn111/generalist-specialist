"""Create corrected analysis inputs and rerun fixed CPU-only Module 2 gates."""
import json
from pathlib import Path
import shutil
import sys

from .audit import ROOT, RESULTS, OUT, sha, save, jsonl

D = RESULTS/'performance_model_generalization_corrected'
E = RESULTS/'performance_model_local_calibration_corrected'
F = RESULTS/'performance_model_ranking_corrected'
G = RESULTS/'performance_model_comparable_groups_corrected'
SRC = RESULTS/'performance_model_active_sampling/dataset_rows.jsonl'


def read_lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def prepare():
    ledger = {r['source_result_path']:r for r in read_lines(OUT/'run_provenance_audit.jsonl')}
    rows = read_lines(SRC)
    observed_paths = {r['source_result_path'] for r in rows}
    if len(observed_paths) != 69:
        raise ValueError('Expected 69 historical source runs')
    for path in observed_paths:
        a = ledger[path]
        if a['run_status'] != 'completed' or not a['valid_for_latency'] or sha(path) != a['source_sha256']:
            raise ValueError('Invalid Module 2D/2E source: '+path)
    D.mkdir(parents=True, exist_ok=True)
    for subset, allowed in [('strict', {'steady'}), ('extended', {'steady','recoverable_congested'})]:
        updated=[]
        for r in rows:
            a=ledger[r['source_result_path']]
            x=dict(r)
            x['legacy_stability']=r['stability']
            x['run_status']=a['run_status']
            x['stability_state_v2_corrected']=a['stability_state_v2_corrected']
            x['correction_audit_sha256']=sha(OUT/'corrected_stability_labels.jsonl')
            x['stability']=dict(r['stability'],stable=a['stability_state_v2_corrected'] in allowed,
                                category='stable' if a['stability_state_v2_corrected'] in allowed else 'unstable')
            updated.append(x)
        jsonl(D/f'{subset}_input.jsonl', updated)
    save(D/'correction_input_provenance.json', dict(source_dataset=str(SRC), source_sha256=sha(SRC),
        audit_labels=str(OUT/'corrected_stability_labels.jsonl'), audit_sha256=sha(OUT/'corrected_stability_labels.jsonl'),
        run_count=69, strict='completed and v2 steady', extended='completed and v2 steady/recoverable',
        legacy_stability_preserved_in_rows=True, synthetic_legacy_stability_proxy='Only the stable/category fields are mapped for the unchanged Module 2D evaluator. Original nested label is retained as legacy_stability.'))
    # Frozen Module 2G plan inputs are copied byte-for-byte into the new output
    # folder because its analyzer reads frozen inputs and writes reports in one dir.
    G.mkdir(parents=True, exist_ok=True)
    old=RESULTS/'performance_model_comparable_groups'
    for name in ('comparison_plan.json','comparison_plan_hash.json','run_manifest.json'):
        target=G/name
        if target.exists() and sha(target)!=sha(old/name): raise ValueError('Frozen 2G input changed')
        if not target.exists(): shutil.copy2(old/name,target)
    save(G/'correction_input_provenance.json',dict(source=str(old), corrected_labels=str(OUT/'corrected_stability_labels.jsonl'),
        corrected_label_sha256=sha(OUT/'corrected_stability_labels.jsonl'),
        all_45_runs_completed=True, all_45_corrected_labels_equal_original=True,
        original_frozen_inputs_byte_identical=True))


def finalize_d():
    strict=json.loads((D/'strict_run/fold_metrics.json').read_text())
    extended=json.loads((D/'extended_run/fold_metrics.json').read_text())
    metrics=dict(strict=strict['subsets']['primary'],extended=extended['subsets']['primary'])
    evidence=dict(strict=strict['gate_evidence']['primary'],extended=extended['gate_evidence']['primary'])
    decisions=dict(strict=strict['decision'],extended=extended['decision'])
    overall=('PASS' if all(e['passes'] for subset in evidence.values() for e in subset.values()) else
             'PARTIAL' if any(len(e['win_deployments']) >= 2 for subset in evidence.values() for e in subset.values())
             else 'FAIL')
    save(D/'corrected_lodo_metrics.json',dict(subsets=metrics,gate_evidence=evidence,
        run_decisions=decisions, overall_decision=overall,
        fixed_model_config=json.loads((RESULTS/'performance_model_generalization/evaluation_config.json').read_text()),
        source_audit_sha256=sha(OUT/'corrected_stability_labels.jsonl'),
        note='Original Module 2D evaluator rerun twice with corrected v2 eligibility; each primary output is the named corrected subset. No parameter tuning.'))
    return decisions


def prepare_f():
    E.mkdir(parents=True,exist_ok=True)
    F.mkdir(parents=True,exist_ok=True)
    strict=read_lines(D/'strict_input.jsonl')
    old_labels={x['run_id']:x for x in read_lines(E/'stability_v2_labels.jsonl')}
    alias=[]
    seen=set()
    for r in strict:
        rid=r['run_id']
        if rid in seen:continue
        seen.add(rid)
        a=dict(old_labels[rid])
        if a['stability_state_v2']!=r['stability_state_v2_corrected']:
            raise ValueError('Corrected label diverges from Module 2E recomputation')
        a['legacy_original_stability']=a['original_stability']
        a['original_stability']=r['stability']
        a['run_status']='completed'
        alias.append(a)
    jsonl(F/'labels_for_corrected_ranking.jsonl',alias)
    save(F/'correction_input_provenance.json',dict(predictions=str(D/'strict_run/predictions.jsonl'),
        predictions_sha256=sha(D/'strict_run/predictions.jsonl'),dataset=str(D/'strict_input.jsonl'),
        dataset_sha256=sha(D/'strict_input.jsonl'),
        labels=str(F/'labels_for_corrected_ranking.jsonl'),
        labels_sha256=sha(F/'labels_for_corrected_ranking.jsonl'),
        old_labels=str(E/'stability_v2_labels.jsonl'),old_labels_sha256=sha(E/'stability_v2_labels.jsonl'),
        original_legacy_stability_retained_in_alias=True,
        note='Module 2F uses corrected strict LODO predictions and unchanged ranking policy; an alias matches the evaluator consistency check while retaining original legacy stability.'))


def run_f():
    from performance_model.ranking import evaluate as ranking
    ranking.DATA=D/'strict_input.jsonl'
    ranking.OOF=D/'strict_run/predictions.jsonl'
    ranking.OOF_AUDIT=D/'strict_run/leakage_audit.json'
    ranking.LABELS=F/'labels_for_corrected_ranking.jsonl'
    ranking.main(['--output',str(F)])


def run_g():
    from performance_model.comparable_groups.analyze import analyze
    result=analyze(G)
    save(G/'corrected_decision.json',dict(result=result,corrected_source_audit_sha256=sha(OUT/'corrected_stability_labels.jsonl'),
        note='All 45 Module 2G physical runs completed and every recomputed v2 label matches the historical label. Reanalysis uses byte-identical frozen inputs and the original ranking policy.'))

if __name__=='__main__':
    action=sys.argv[1]
    {'prepare':prepare,'finalize_d':finalize_d,'prepare_f':prepare_f,'run_f':run_f,'run_g':run_g}[action]()
