"""Execute v3 against an unfrozen design, preserving UNKNOWN and full cohort.

This produces reviewable CPU wiring results, never a prediction seal, physical
validation or permission to launch the GPU collector. Frozen protocols use the
separate forecast_gate_v3 command.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from .forecast_gate_v2 import _public_cohort,_to_cell,_write
from .forecast_gate_v3 import execute_bundle,prediction_source_paths
from .native_collect import receipt
from .prospective_validation_prepare_v3 import DESIGN_SCHEMA


def diagnose(design_path,output):
    design=json.loads(Path(design_path).read_text())
    if design.get('schema')!=DESIGN_SCHEMA or design.get('status')!='DESIGN_DRAFT_NOT_FROZEN':
        raise ValueError('Only explicit unfrozen CPU design diagnostics accepted')
    if design.get('not_for_gpu_execution') is not True:raise ValueError('Design must prohibit GPU execution')
    jobs=json.loads(Path(design['jobs']['path']).read_text())
    if receipt(design['jobs']['path'])!=design['jobs']:raise ValueError('Changed draft job list')
    candidates={c['id']:c for c in design['candidates']}
    expected={(r['run_id'],c) for r in design['validation_domain'] for c in candidates}
    if {(j['run_id'],j['candidate_id']) for j in jobs}!=expected or len(jobs)!=len(expected):
        raise ValueError('Every predeclared draft job required')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    cells=[];results=[]
    for index,job in enumerate(jobs):
        candidate=candidates[job['candidate_id']];bundle_path=design['model_paths'][candidate['model_id']]
        bundle=json.loads(Path(bundle_path).read_text());public,cohort=_public_cohort(json.loads(Path(job['public_inputs_path']).read_text()))
        expected_domain=next(r['requests'] for r in design['validation_domain'] if r['run_id']==job['run_id'])
        if {rid:{k:row[k] for k in ('request_id','pool_id','task')} for rid,row in cohort.items()}!={row['request_id']:row for row in expected_domain}:
            raise ValueError('Draft public inputs dropped or changed validation domain')
        try:result=execute_bundle(bundle,public,allow_trained_hypothesis=True)
        except UnknownEvidence as exc:
            result=dict(status='UNKNOWN',reason=str(exc),full_public_cohort=cohort,
                GPU_started=False,physical_accuracy_validated=False,physical_gain_certified=False,
                measured_target_fallback_used=False)
        _write(output/f'forecast-{index:04d}.json',result)
        cells.append(_to_cell(result,cohort,candidate,job['run_id'],design,bundle,public['phase_start_ns']))
        results.append(dict(job=job,bundle=receipt(bundle_path),result=receipt(output/f'forecast-{index:04d}.json'),
                            status=result.get('status','MODEL_EXECUTED'),reason=result.get('reason')))
    _write(output/'diagnostic-cells.json',cells)
    report=dict(schema='gsdeploy.forecast_gate_draft_wiring_diagnostic.v3',
        status='UNFROZEN_CPU_WIRING_DIAGNOSTIC',design=receipt(design_path),results=results,
        diagnostic_cells=receipt(output/'diagnostic-cells.json'),
        request_rows=sum(len(c['requests']) for c in cells),candidate_cells=len(cells),
        full_declared_cohort_retained=True,models_actually_instantiated=True,
        source_snapshot=[receipt(p) for p in prediction_source_paths()],
        unknown_cells=sum(row['status']=='UNKNOWN' for row in results),
        predictions_sealed=False,frozen_protocol_created=False,GPU_started=False,
        independent_validation_performed=False,prediction_accuracy_established=False,physical_gain_certified=False)
    _write(output/'report.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=diagnose(args.design,args.output)
    print(json.dumps({key:result[key] for key in ('status','request_rows','candidate_cells','unknown_cells','predictions_sealed')}))


if __name__=='__main__':main()
