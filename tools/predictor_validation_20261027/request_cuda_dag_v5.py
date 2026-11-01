"""Convert strict startup-aware light training traces without relabeling raw.

Retains v2 DAG semantics and each original full execution context. Canonical
NVTX kernels are counted once; profiler phases are never concatenated as a
second copy of the same kernel denominator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cuda_dependency_dag_v2 import audit_sqlite
from .kernel_sequence_audit_v2 import IDENTITY_FIELDS
from .native_collect import receipt, save
from .request_cuda_dag_v2 import request_plan
from .small_training_domain_bridge_review_v4 import load_report


def owner_requests(raw, trace, sql_ref, alias):
    rows = [r for r in raw['request_records'] if r['model'] == alias]
    requests = {r['request_id']:dict(row=r, kernels=[], source_traces=[sql_ref], clock_mapping=None) for r in rows}
    if len(requests) != len(rows):
        raise ValueError('Unique actual request identities required')
    pid = raw['ready'][alias]['pid']
    seen = set()
    canonical = [(r.get('stage'), k) for r in trace['nvtx_ranges'] for k in r.get('kernels', [])]
    canonical.extend((None, k) for k in trace.get('unattributed_kernels', []))
    for stage, kernel in canonical:
        identity = tuple(kernel.get(k) for k in IDENTITY_FIELDS)
        if identity in seen:
            raise ValueError('Duplicate canonical kernel identity')
        seen.add(identity)
        rid = stage[0] if stage else None
        if rid not in requests:
            continue
        if kernel['pid'] != pid:
            raise ValueError('Training request kernel does not belong to actual READY owner')
        row = requests[rid]
        row['kernels'].append(dict(kernel, source_sqlite=sql_ref))
        row['clock_mapping'] = trace['clock_mappings'].get(str(pid))
    if any(not r['kernels'] or not r['clock_mapping'] or r['clock_mapping'].get('valid') is not True for r in requests.values()):
        raise ValueError('Complete actual request kernels and valid clock mapping required')
    return requests


def convert(review_path, output):
    report, plan, raw = load_report(review_path, 'light-kernel')
    if (any(len(j['inputs']) != 1 or j.get('split') != 'training' for j in plan['jobs'])
            or any(r.get('split') != 'training' or r.get('status') != 'completed' for r in raw['request_records'])):
        raise ValueError('Only complete isolated light-domain training requests may supply DAG templates')
    intervals = sorted((r['worker_response']['start_perf_ns'], r['worker_response']['end_perf_ns']) for r in raw['request_records'])
    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError('Template training handlers must be isolated')
    context = json.loads(Path(report['context']['path']).read_text())
    if (context['measurement_domain'] != 'light-kernel' or context['affine_observation'] is not False
            or plan['execution_context_sha256'] != raw['execution_context_sha256']):
        raise ValueError('Original unhooked light-domain identities required')
    profiles, sources = [], [receipt(review_path), report['raw'], report['plan'], report['context']]
    for alias, owner in report['owners'].items():
        if not owner['request_ids']:
            continue
        trace = json.loads(Path(owner['parsed']['path']).read_text())
        sql = owner['sqlite']
        if (receipt(owner['parsed']['path']) != owner['parsed'] or receipt(sql['path']) != sql
                or sql not in trace['sources']):
            raise ValueError('Unchanged canonical parsed trace and exact SQLite source required')
        for source in trace['sources']:
            if receipt(source['path']) != source:
                raise ValueError('Changed canonical trace evidence')
        full = audit_sqlite(sql['path'])
        requests = owner_requests(raw, trace, sql, alias)
        if set(requests) != set(owner['request_ids']):
            raise ValueError('Strict owner request denominator changed')
        for rid, request in requests.items():
            profile = request_plan(rid, request, trace, full, sql['path'])
            profile.update(family=context['residents'][alias], source_review=receipt(review_path),
                           original_execution_context_sha256=raw['execution_context_sha256'])
            if profile['cyclic'] or profile['unresolved']:
                raise ValueError('Unresolved or cyclic CUDA training topology')
            profiles.append(profile)
        sources.extend([owner['parsed'], sql])
    if len(profiles) != len(raw['request_records']):
        raise ValueError('Complete actual measured request denominator required')
    sources.extend(receipt(Path(__file__).with_name(n)) for n in ('request_cuda_dag_v5.py','request_cuda_dag_v2.py',
        'cuda_dependency_dag_v2.py','kernel_sequence_audit_v2.py','small_training_domain_bridge_review_v4.py'))
    if any(receipt(r['path']) != r for r in sources):
        raise ValueError('DAG conversion source changed during execution')
    result = dict(schema='gsdeploy.request_cuda_dag_profiles.v2', execution_context_sha256=raw['execution_context_sha256'],
        profiles=profiles, denominator=len(profiles), training_only=True, sources=sources,
        raw_schema_rewritten=False, original_full_execution_identity_retained=True,
        GPU_started=False, physical_accuracy_validated=False, unseen_input_supported=False)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out/'profiles.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = convert(args.review, args.output)
    print(json.dumps(dict(requests=result['denominator'], kernels=sum(p['kernel_count'] for p in result['profiles']))))


if __name__ == '__main__':
    main()
