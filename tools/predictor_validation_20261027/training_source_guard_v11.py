"""Reject nontraining availability lineage before fitting or prediction."""
from pathlib import Path

from .multi_native_composite_v5 import checked


def availability_training_sources(report_references):
    if len(report_references) < 2:
        raise ValueError('Independent training capture reports required')
    audits = [];raw_ids = set()
    for reference in report_references:
        report = checked(reference)
        if (report.get('schema') != 'gsdeploy.recorded_DAG_pending_frontier.v8'
                or report.get('status') != 'SOURCE_DAG_PENDING_FRONTIER_EXTRACTED'
                or report.get('validation_labels_used') is not False):
            raise ValueError('Explicit training-only pending-frontier report required')
        named = {}
        for name in ('raw.json','plan.json','context.json'):
            refs = [s for s in report['sources'] if Path(s['path']).name == name]
            if len(refs) != 1:
                raise ValueError('Unique original training raw, plan and context required')
            named[name] = refs[0]
        raw,plan,context = (checked(named[name]) for name in ('raw.json','plan.json','context.json'))
        rows = raw.get('request_records',[])
        primers = raw.get('priming_request_records',[])
        if (raw.get('kind') != 'measured' or raw.get('status') != 'completed'
                or raw.get('errors') or not raw.get('request_records')
                or any(r.get('split') != 'training' or r.get('status') != 'completed' for r in rows)
                or any(r.get('status') != 'completed' or r.get('phase') != 'priming'
                       or r.get('split') not in (None,'training') for r in primers)
                or plan.get('split') != 'training' or plan.get('training_only') is not True
                or not plan.get('jobs') or any(j.get('split') != 'training' for j in plan['jobs'])
                or context.get('training_only') is not True):
            raise ValueError('Only complete measured training requests and plans may fit availability')
        from .native_collect import context_identity
        identity = context_identity(context)
        if (raw.get('execution_context_sha256') != identity
                or plan.get('execution_context_sha256') != identity):
            raise ValueError('Training raw/plan/context identities differ')
        if named['raw.json']['sha256'] in raw_ids:
            raise ValueError('Duplicate raw is not independent training')
        raw_ids.add(named['raw.json']['sha256'])
        audits.append(dict(report=reference,raw=named['raw.json'],plan=named['plan.json'],
                           context=named['context.json'],measurement_requests=len(raw['request_records'])))
    return audits
