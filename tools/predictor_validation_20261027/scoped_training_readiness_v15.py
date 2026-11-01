"""CPU-only actual source-closure readiness; no protocol, seal or GPU work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from . import learned_campaign_seal_v15 as sealer
from . import scoped_training_evidence_v15 as scoped
from .multi_native_composite_v5 import checked
from .native_collect import receipt, save
from .training_source_guard_v11 import availability_training_sources


def run(plan_path,index_path,configuration_path,manifest_path,output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    sources=[receipt(p) for p in (plan_path,index_path,configuration_path,manifest_path,
        sealer.__file__,scoped.__file__,__file__)]
    plan=checked(sources[0]);configuration=checked(sources[2]);checks=[]
    started=time.perf_counter()
    try:
        sealer._preserved_plan(plan);checks.append(dict(check='actual_preserved_plan',status='PASS'))
        training=sealer._complete_index(index_path)
        checks.append(dict(check='actual_complete_index',status='PASS',records=training['records'],
            independent_raw_runs=len(training['raw_sha256'])))
        manifest=checked(training['artifacts']['source-manifest.json'])
        lineage=checked(training['artifacts']['lineage-audit.json'])
        contexts=[c['context'] for c in manifest['captures']]
        raw=[c['raw'] for c in manifest['captures']]
        smoke=[r['reference'] for r in lineage['raw_references']
            if r['classification']=='INHERITED_RUNTIME_SMOKE_PROVENANCE_ONLY_NOT_INDEXED']
        guard=scoped.training_json_read_guard(raw,inherited_smoke_references=smoke)
        availability=checked(configuration['base_model_configuration']['availability_model'])
        audit=availability_training_sources(availability['source_DAG_reports'])
        if any(a['raw']['sha256'] not in training['raw_sha256'] for a in audit):
            raise ValueError('Actual availability fit raw omitted from the complete index')
        checks.append(dict(check='actual_availability_training_raw_index_coverage',status='PASS',audit=audit))
        roots=[str(Path(plan_path).resolve()),str(Path(configuration_path).resolve()),training['index']['path'],
            *[r['path'] for r in training['strict_reviews']]]
        resolution=scoped.resolve_closure(roots,manifest_path,training_context_references=contexts,
            before_json_read=guard,metadata_boundary_policy=scoped.operational_metadata_boundary_policy())
        save(out/'source-resolution.json',resolution)
        checks.append(dict(check='actual_scoped_operational_training_closure',status='PASS',
            resolution=receipt(out/'source-resolution.json'),materialized_sources=len(resolution['materialized_sources']),
            historical_substitutions=len(resolution['historical_substitutions']),
            effective_opaque_boundary_edges=resolution['effective_opaque_boundary_edges'],
            boundaries=resolution['boundaries']))
        status='ACTUAL_OPERATIONAL_TRAINING_CLOSURE_READY_UNDER_DECLARED_HISTORICAL_BOUNDARIES'
        error=None
    except Exception as exc:
        status='ACTUAL_SCOPED_PRESEAL_READINESS_FAILURE';error=dict(type=type(exc).__name__,reason=str(exc))
        if hasattr(exc,'scoped_closure_diagnostic'):
            save(out/'failure-diagnostic.json',exc.scoped_closure_diagnostic)
            error['diagnostic']=receipt(out/'failure-diagnostic.json')
    unchanged=all(receipt(s['path'])==s for s in sources)
    report=dict(schema='gsdeploy.actual_scoped_preseal_readiness.v15',status=status,sources=sources,
        checks=checks,error=error,seconds=time.perf_counter()-started,sources_unchanged=unchanged,
        numeric_preflight_checked=False,numeric_preflight_prerequisite_not_bypassed=True,
        full_historical_transitive_closure_verified=False,validation_labels_read=False,
        protocol_frozen=False,predictions_sealed=False,GPU_started=False)
    save(out/'report.json',report)
    print(json.dumps(dict(report=receipt(out/'report.json'),status=status,error=error,seconds=report['seconds'])))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('plan','training-index','configuration','historical-manifest','output'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    run(args.plan,args.training_index,args.configuration,args.historical_manifest,args.output)
