"""One-attempt CPU-only H600 primary capacity CV and numerical audit.

Waits only for the eight primary collectors to finish writing their native raw
files. Scored files are completion indicators, never numerical inputs. No seal,
independent outcome, repeat, preflight, H90 evidence or GPU action is used.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
import traceback

from . import h600_capacity_v1 as capacity
from .native_flow import checked,receipt,save
from .probe_validation import read

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/predictor-gpu-20261027'


def emit(event,**data):
    print(json.dumps(dict(event=event,time_ns=time.perf_counter_ns(),**data)),flush=True)


def check_sources(config):
    for ref in config['source_receipts']:checked(ref)


def prepare(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA must be explicitly hidden from this CPU watcher')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        if os.environ.get(key)!='1':raise ValueError('Single-thread CPU environment required')
    protocol_path=BASE/'capacity-h600-v1/protocol.json';protocol=capacity.verify_protocol(protocol_path)
    suite=capacity.verify_suite(protocol['training_suite']['path'])
    refs=[receipt(__file__),receipt(protocol_path),protocol['training_suite'],*protocol['source_receipts'],*capacity.sources()]
    unique={}
    for ref in refs:
        checked(ref)
        if ref['path'] in unique and unique[ref['path']]!=ref:raise ValueError('Conflicting frozen source receipts')
        unique[ref['path']]=ref
    config=dict(schema='gsdeploy.h600_capacity_training_CPU_watch.v1',created_ns=time.perf_counter_ns(),
        protocol=receipt(protocol_path),training_suite=protocol['training_suite'],
        primary_jobs=suite['jobs'],source_receipts=list(unique.values()),poll_seconds=30,
        numerical_sources='Only eight original H600 primary native raw files; scored files are existence-only completion indicators',
        output=str(output),GPU_started=False,seal_authorized=False,qualification_authorized=False,
        independent_results_read=False,one_attempt=True,retries=False)
    output.mkdir(parents=True,exist_ok=False);save(output/'configuration.json',config)
    return config


def numerical_audit(config,report):
    """Re-derive predictions and observed means from immutable primary raws."""
    bundle=capacity.weighted.load_training(config['training_suite']['path'])
    if len(bundle['runs'])!=8 or bundle['seeds']!=capacity.POLICY['training_seeds']:
        raise ValueError('Exactly eight H600 primary runs required in numerical audit')
    for ref in report['source_receipts']:checked(ref)
    if (report.get('training_only') is not True or report.get('independent_results_read') is not False or
            report.get('independent_validation') is not False or report.get('qualification_passed') is not False or
            report.get('final_gain_eligible') is not False or len(report['cases'])!=8):
        raise ValueError('Capacity report scope or complete training-case count mismatch')
    by_id={r['flow_id']:r for r in bundle['runs']};seen=set();rows=[];groups=[]
    for case in report['cases']:
        fid=case['flow_id']
        if fid in seen or fid not in by_id:raise ValueError('Duplicate or undeclared capacity CV case')
        seen.add(fid);heldout=by_id[fid];seed=heldout['seed'];training=[r for r in bundle['runs'] if r['seed']!=seed]
        if len(training)!=4 or any(r['raw_source']==heldout['raw_source'] for r in training):
            raise ValueError('Complete heldout seed exclusion failed')
        candidate=next(c for c in heldout['plan']['candidates'] if c['id']==heldout['candidate_id'])
        catalog=read(checked(heldout['plan']['catalog']))
        prediction=capacity.weighted.estimate(heldout['plan'],candidate,training,catalog)
        evaluation=capacity.weighted.evaluate(prediction,heldout['observations'])
        if (case['prediction']!=prediction or case['evaluation']!=evaluation or
                case['training_flow_ids']!=[r['flow_id'] for r in training] or
                case['heldout_raw']!=heldout['raw_source'] or case['heldout_plan']!=heldout['plan_source'] or
                case['seed']!=seed or case['candidate_id']!=heldout['candidate_id']):
            raise ValueError('Capacity CV numeric values or original primary identity changed')
        if (prediction['threshold']!=.8 or prediction['tolerance']!=1e-8 or
                prediction['original_capacity_factors']!=catalog['capacity_factors'] or
                prediction['training_seeds']!=[s for s in bundle['seeds'] if s!=seed]):
            raise ValueError('Original g(n), rho or complete same-seed exclusion changed')
        # The frozen CV report describes relative error with observed mean as
        # denominator. Preserve it; separately show the unchanged prospective
        # service-check arithmetic whose denominator is the frozen prediction.
        service=capacity.service_checks(prediction,heldout['observations'])
        rows.append(dict(flow_id=fid,seed=seed,candidate_id=heldout['candidate_id'],
            heldout_raw=heldout['raw_source'],training_raws=[r['raw_source'] for r in training],
            complete_seed_exclusion_verified=True,numeric_prediction_recomputed=True,
            numeric_evaluation_recomputed=True,original_capacity_factors=catalog['capacity_factors'],
            planning_rho_by_family=prediction['rho_by_family'],
            planning_rho_threshold_satisfied=prediction['rho_threshold_satisfied'],
            original_service_threshold_training_diagnostic=service,
            all_training_service_diagnostics_within_20_percent=all(c['passed'] is True for c in service),
            independent_qualification_passed=False,final_gain_eligible=False))
        groups.extend(evaluation['groups'])
    paired=[g for g in groups if g['weighted_mean_relative_error'] is not None and g['pooled_mean_relative_error'] is not None]
    summary=dict(requested_groups=len(groups),known_weighted_groups=sum(g['weighted_mean_relative_error'] is not None for g in groups),
        unknown_weighted_groups=sum(g['weighted_mean_relative_error'] is None for g in groups),paired_groups=len(paired),
        paired_weighted_macro_mean_relative_error=statistics.mean(g['weighted_mean_relative_error'] for g in paired) if paired else None,
        paired_pooled_macro_mean_relative_error=statistics.mean(g['pooled_mean_relative_error'] for g in paired) if paired else None,
        weighted_better_groups=sum(g['weighted_mean_relative_error']<g['pooled_mean_relative_error'] for g in paired),
        weighted_worse_groups=sum(g['weighted_mean_relative_error']>g['pooled_mean_relative_error'] for g in paired))
    if summary!=report['summary']:raise ValueError('Capacity CV summary differs from recomputed original arithmetic')
    for ref in [*bundle['source_receipts'],*config['source_receipts']]:checked(ref)
    all_service=[c for r in rows for c in r['original_service_threshold_training_diagnostic']]
    return dict(schema='gsdeploy.h600_capacity_training_CV_numeric_audit.v1',source=receipt(__file__),
        report=receipt(Path(config['output'])/'cv/report.json'),protocol=config['protocol'],rows=rows,summary=summary,
        exact_primary_raw_count=8,all_original_sources_verified=True,numeric_recomputation_passed=True,
        original_service_threshold=.2,original_service_error_denominator='Frozen predicted worker mean',
        training_service_group_counts=dict(total=len(all_service),within_threshold=sum(c['passed'] is True for c in all_service),
            over_threshold=sum(c['passed'] is False for c in all_service),unknown=sum(c['passed'] is None for c in all_service)),
        original_rho_threshold=.8,original_rho_tolerance=1e-8,source_receipts=bundle['source_receipts'],
        independent_results_read=False,independent_validation=False,qualification_passed=False,
        original_failures_preserved=True,final_gain_eligible=False,GPU_started=False)


def run(output):
    config=prepare(output);output=Path(config['output'])
    emit('waiting_for_eight_primary_training_runs',output=str(output),poll_seconds=30)
    try:
        # Native raw is fully written before its separate scored-raw is created.
        # Never parse scored outputs or wait for the two repeat collectors.
        while not all((Path(j['output'])/name).is_file() for j in config['primary_jobs'] for name in ('raw.json','scored-raw.json')):
            time.sleep(30)
        check_sources(config)
        save(output/'started-CPU.json',dict(created_ns=time.perf_counter_ns(),source=receipt(__file__),GPU_started=False))
        emit('starting_primary_inventory')
        inventory=capacity.inventory(config['protocol']['path'],output/'inventory.json')
        if inventory['completed']!=8:raise ValueError('All eight primary inventory units must verify')
        emit('starting_primary_capacity_CV')
        report=capacity.crossvalidate(config['protocol']['path'],output/'cv')
        emit('starting_independent_CPU_numeric_recomputation')
        audit=numerical_audit(config,report);save(output/'numerical-audit.json',audit);check_sources(config)
        lines=['H600 primary-only capacity training CV','',
            'Eight native primary training sources only. Both full seed folds exclude all four heldout routes. Original g(n), rho=.8+1e-8, estimator and source modules are unchanged. No repeats, H90, preflight or independent outputs used.',
            '',f"Frozen CV summary: {report['summary']}",'',
            'CV errors above use the frozen report denominator (observed mean). The separate table below uses the original prospective 20% service-error rule with the frozen predicted worker mean as denominator; these are training diagnostics only.',
            '', '| Flow | Original planning rho check | Every service group within 20% |','|---|---|---|']
        for r in audit['rows']:lines.append(f"| {r['flow_id']} | {r['planning_rho_threshold_satisfied']} | {r['all_training_service_diagnostics_within_20_percent']} |")
        lines+=['',f"Service group counts: {audit['training_service_group_counts']}",'',
            'All numeric predictions, evaluations and summaries were recomputed from immutable raw sources. All source receipts verified. No prospective seal, independent qualification or gain claim is produced. Prior FAIL/UNKNOWN remains unchanged.']
        (output/'results.md').write_text('\n'.join(lines)+'\n')
        completion=dict(status='COMPLETED',created_ns=time.perf_counter_ns(),source=receipt(__file__),
            report=receipt(output/'cv/report.json'),inventory=receipt(output/'inventory.json'),
            numerical_audit=receipt(output/'numerical-audit.json'),summary=report['summary'],
            training_service_group_counts=audit['training_service_group_counts'],
            seal_performed=False,qualification_passed=False,independent_results_read=False,GPU_started=False)
        save(output/'completed-CPU.json',completion);emit('completed_H600_capacity_training_CV',**completion)
    except BaseException as exc:
        failure=dict(status='FAILED_STOPPED_NO_RETRY',created_ns=time.perf_counter_ns(),error=repr(exc),
            source=receipt(__file__),GPU_started=False,seal_performed=False,independent_results_read=False)
        save(output/'failed-CPU.json',failure);emit('failed_H600_capacity_training_CV',**failure)
        traceback.print_exc();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
