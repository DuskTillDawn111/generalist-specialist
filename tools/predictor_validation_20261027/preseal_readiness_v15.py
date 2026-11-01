"""Read-only actual-data preseal prerequisites; never freeze, seal or collect."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

from . import learned_campaign_seal_v15 as sealer
from .historical_evidence_v4 import resolve_closure
from .native_collect import receipt


def run(plan_path,index_path,configuration_path,manifest_path,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    sources=[receipt(sealer.__file__),receipt(__file__),receipt(plan_path),receipt(index_path),
             receipt(configuration_path),receipt(manifest_path)]
    plan=json.loads(Path(plan_path).read_text());configuration=json.loads(Path(configuration_path).read_text())
    source_manifest_path=Path(index_path).with_name('source-manifest.json')
    source_manifest=json.loads(source_manifest_path.read_text())
    allowed_raw={c['raw']['path'] for c in source_manifest['captures']}
    opened=[];blocked=[];checks=[];training=None
    original_read_text=Path.read_text
    def guarded_read_text(path,*args,**kwargs):
        name=str(path.resolve());base=path.name.lower();lower=name.lower()
        if ((base=='raw.json' and name not in allowed_raw)
                or base in ('observed.json','observed-metrics.json','predictions.json')
                or ('validation' in lower and any(t in base for t in ('labels','observed','outcomes')))):
            blocked.append(name)
            raise ValueError('READONLY_PRESEAL_GUARD_REFUSED_NONINDEXED_RAW_OR_VALIDATION_OUTCOME: '+name)
        opened.append(name)
        return original_read_text(path,*args,**kwargs)
    with patch.object(Path,'read_text',guarded_read_text):
        for label,call in (
            ('actual_preserved_plan',lambda:sealer._preserved_plan(plan)),
            ('actual_complete_index',lambda:sealer._complete_index(index_path)),
        ):
            start=time.perf_counter()
            try:
                value=call()
                if label=='actual_complete_index':training=value
                checks.append(dict(check=label,status='PASS',seconds=time.perf_counter()-start,
                    result=value))
            except Exception as exc:
                checks.append(dict(check=label,status='FAIL',seconds=time.perf_counter()-start,
                    exception=type(exc).__name__,reason=str(exc)))
        if training:
            roots=[str(Path(plan_path).resolve()),str(Path(configuration_path).resolve()),
                training['index']['path'],*[r['path'] for r in training['strict_reviews']]]
            start=time.perf_counter()
            try:
                resolution=resolve_closure(roots,manifest_path)
                (output/'source-resolution.json').write_text(json.dumps(resolution,indent=2)+'\n')
                checks.append(dict(check='actual_resolve_closure',status='PASS',seconds=time.perf_counter()-start,
                    roots=roots,resolution=receipt(output/'source-resolution.json'),
                    materialized_sources=len(resolution['materialized_sources']),
                    historical_substitutions=len(resolution['historical_substitutions'])))
            except Exception as exc:
                checks.append(dict(check='actual_resolve_closure',status='FAIL',seconds=time.perf_counter()-start,
                    roots=roots,exception=type(exc).__name__,reason=str(exc)))
    unchanged=all(receipt(s['path'])==s for s in sources)
    report=dict(schema='gsdeploy.learned_campaign_preseal_readiness.v15',
        status='ACTUAL_READONLY_PREREQUISITES_PASS' if all(c['status']=='PASS' for c in checks) else 'ACTUAL_PRESEAL_PREREQUISITE_FAILURE',
        sources=sources,checks=checks,current_sources_unchanged=unchanged,
        plan_request_count=len(plan['trace']),candidate_count=len(plan['candidates']),
        configuration_schema=configuration['schema'],
        parsed_json_paths=sorted(set(opened)),refused_parse_paths=blocked,
        indexed_training_raw_count=len(allowed_raw),numeric_preflight_checked=False,
        numeric_preflight_prerequisite_bypassed=False,protocol_frozen=False,
        predictions_sealed=False,GPU_started=False,validation_labels_read=False)
    (output/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(report=receipt(output/'report.json'),status=report['status'],
        checks=[{k:c[k] for k in ('check','status','seconds','reason') if k in c} for c in checks]),indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('plan','training-index','configuration','historical-manifest','output'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    run(args.plan,args.training_index,args.configuration,args.historical_manifest,args.output)
