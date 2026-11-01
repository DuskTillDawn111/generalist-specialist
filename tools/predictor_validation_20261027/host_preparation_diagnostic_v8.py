"""CPU forensic audit of isolated first-activity supply and natural phase bounds.

API timestamps belong to the light profiler domain. Natural Python boundaries
constrain preparation but do not observe the first CUDA activity directly.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics

from gsdeploy.resources import identity
from .dag_host_release_v5 import ACTIVITY, structure
from .dag_host_release_v7 import checked, verify_model
from .dag_phase_diagnostic_v8 import load_capture
from .native_collect import receipt, save
from .small_training_domain_bridge_review_v4 import request_kernels
from .trace_export import linux_pid


def summarize_capture(ref):
    strict, plan, raw = load_capture(ref, 'light-kernel')
    rows = []
    sources = [ref, strict['raw'], strict['plan'], strict['context']]
    for alias, owner in strict['owners'].items():
        parsed = checked(owner['parsed'])
        checked_ref = owner['sqlite']
        if receipt(checked_ref['path']) != checked_ref:
            raise ValueError('Changed solo CUDA source')
        sources.extend((owner['parsed'], checked_ref))
        with sqlite3.connect(Path(checked_ref['path']).as_uri()+'?mode=ro', uri=True) as conn:
            conn.row_factory = sqlite3.Row
            runtime = [dict(r) for r in conn.execute('select r.*,s.value as api_name from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id')]
            tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
            auxiliary = []
            for table, kind in (('CUPTI_ACTIVITY_KIND_MEMCPY', 'copy'), ('CUPTI_ACTIVITY_KIND_MEMSET', 'memset')):
                if table in tables:
                    auxiliary.extend((kind, dict(r)) for r in conn.execute('select * from '+table))
        for row in raw['request_records']:
            if row['model'] != alias:
                continue
            rid = row['request_id']
            phases = [p for p in parsed['phases'] if p['request_id'] == rid]
            pid = phases[0]['pid']
            mapping = parsed['clock_mappings'][str(pid)]
            if mapping.get('valid') is not True:
                raise ValueError('Valid independently observed handler/trace clock mapping required')
            handler = row['worker_response']['start_perf_ns'] - mapping['offset_ns']
            inside = lambda stamp: any(p['nsys_start_ns'] <= stamp <= p['nsys_end_ns'] for p in phases)
            apis = [a for a in runtime if linux_pid(a['globalTid']) == pid and inside(a['start'])]
            apis.sort(key=lambda a: (a['start'], a['end'], a['correlationId']))
            api_by_correlation = {a['correlationId']: a for a in apis}
            kernels = request_kernels(parsed, checked_ref['path'], row)
            activities = [('gpu', k['kernel']['correlation_id'], k['kernel']['nsys_start_ns']) for k in kernels]
            activities.extend((kind, a['correlationId'], a['start']) for kind, a in auxiliary
                              if linux_pid(a['globalPid']) == pid and a['correlationId'] in api_by_correlation)
            activities.sort(key=lambda a: (api_by_correlation[a[1]]['start'], a[2]))
            first = activities[0]
            gpu = next(a for a in activities if a[0] == 'gpu')
            first_api = api_by_correlation[first[1]]
            first_gpu_api = api_by_correlation[gpu[1]]
            before = [a for a in apis if a['start'] < first_api['start']]
            rows.append(dict(alias=alias, task=row['task'], pool_id=row['pool_id'], request_id=rid,
                source_raw=strict['raw'], source_sqlite=checked_ref, clock_mapping=mapping,
                first_runtime_API=dict(name=apis[0]['api_name'], offset_ns=apis[0]['start']-handler),
                first_activity=dict(kind=first[0], api_name=first_api['api_name'],
                    API_entry_offset_ns=first_api['start']-handler, API_return_offset_ns=first_api['end']-handler,
                    activity_start_offset_ns=first[2]-handler),
                first_gpu=dict(API_entry_offset_ns=first_gpu_api['start']-handler,
                    API_return_offset_ns=first_gpu_api['end']-handler, activity_start_offset_ns=gpu[2]-handler),
                preceding_runtime_API=[dict(name=a['api_name'], entry_offset_ns=a['start']-handler,
                    return_offset_ns=a['end']-handler) for a in before],
                complete_runtime_sequence_count=len(apis),
                complete_runtime_name_sequence_sha256=identity([a['api_name'] for a in apis]),
                complete_ordered_gpu_signature_sha256=identity([dict(group=k['group'], signature=k['signature']) for k in kernels]),
                complete_ordered_gpu_count=len(kernels),
                phase_offsets=[dict(id=p['id'], semantic_phase=p['semantic_phase'],
                    start_offset_ns=p['start_ns']-row['worker_response']['start_perf_ns'],
                    end_offset_ns=p['end_ns']-row['worker_response']['start_perf_ns'])
                    for p in row['worker_response'].get('semantic_forward_observations', {}).get('records', [])[:1]]))
    return rows, sources


def derive(parameters_path):
    parameters = checked(receipt(parameters_path))
    host = checked(parameters['host_model']); verify_model(host)
    dag = checked(host['dag'])
    plans = {(p['alias'], p['task'], p['pool_id']):p for p in dag['profiles']}
    groups = {(g['alias'], g['task'], g['pool_id']):g for g in host['groups']}
    by_key = defaultdict(list)
    sources = [receipt(__file__), receipt(parameters_path), parameters['host_model'], host['dag']]
    for ref in parameters['solo_light_reviews']:
        rows, refs = summarize_capture(ref); sources.extend(refs)
        for row in rows:
            by_key[row['alias'], row['task'], row['pool_id']].append(row)
    natural = defaultdict(list)
    for ref in parameters['solo_natural_reviews']:
        strict, _, raw = load_capture(ref, 'natural')
        sources.extend((ref, strict['raw'], strict['plan'], strict['context']))
        for row in raw['request_records']:
            w = row['worker_response']; start = w['start_perf_ns']
            stages = [dict(id=s['id'], start_offset_ns=s['start_ns']-start,
                end_offset_ns=s['end_ns']-start) for s in w.get('observed_stages', [])]
            semantic = w.get('semantic_forward_observations', {}).get('records', [])
            natural[row['model'], row['task'], row['pool_id']].append(dict(source_raw=strict['raw'],
                request_id=row['request_id'], stages=stages,
                first_semantic_start_offset_ns=semantic[0]['start_ns']-start if semantic else None,
                natural_first_CUDA_activity_observed=False))
    rows = []
    for key, profile in sorted(plans.items()):
        light = by_key[key]
        if len(light) != 2 or len({r['source_raw']['sha256'] for r in light}) != 2:
            raise ValueError('Two independent isolated light captures required per exact input')
        exact = len({r['complete_ordered_gpu_signature_sha256'] for r in light}) == 1
        if not exact:
            raise ValueError('Independent isolated ordered GPU signature mismatch')
        features = structure(profile); par = groups[key]['joint_mean']
        seq = sorted((n for n in profile['nodes'] if n['kind'] != 'handler_end'), key=lambda n:n['issue_start_ns'])
        first = next(n for n in seq if n['kind'] in ACTIVITY)
        gpu = next(n for n in seq if n['kind'] == 'gpu')
        anchors = dict(first_issue=dict(node_id=seq[0]['id'], kind=seq[0]['kind'], depth=features['issue_depth'][seq[0]['id']]),
            first_activity=dict(node_id=first['id'], kind=first['kind'], depth=features['issue_depth'][first['id']]),
            first_gpu=dict(node_id=gpu['id'], depth=features['issue_depth'][gpu['id']]))
        for name, node in (('first_issue',seq[0]),('first_activity',first),('first_gpu',gpu)):
            anchors[name].update(original_profile_API_entry_offset_ns=node['release_ns'],
                v7_issue_depth_predicted_release_ns=round(par['issue_depth_ns_per_step']*features['issue_depth'][node['id']]))
        rows.append(dict(alias=key[0], task=key[1], pool_id=key[2], template_anchors=anchors,
            independent_light_observations=light, independent_natural_observations=natural[key],
            exact_ordered_gpu_signature_match=exact,
            runtime_name_sequence_match=len({r['complete_runtime_name_sequence_sha256'] for r in light})==1,
            first_activity_API_entry_equal_capture_mean_ns=statistics.mean(r['first_activity']['API_entry_offset_ns'] for r in light),
            first_gpu_API_entry_equal_capture_mean_ns=statistics.mean(r['first_gpu']['API_entry_offset_ns'] for r in light)))
    sources = list({s['path']:s for s in sources}.values())
    if any(receipt(s['path']) != s for s in sources):
        raise ValueError('Source changed during forensic audit')
    return dict(schema='gsdeploy.isolated_host_preparation_forensic.v8', status='CPU_FORENSIC_COMPLETE',
        rows=rows, sources=sources, GPU_started=False, training_only=True,
        first_issue_depth_is_zero=False, independent_preparation_parameter_in_v7=False,
        profiler_to_natural_first_activity_transfer_validated=False,
        natural_semantic_start_is_first_CUDA_release=False,
        natural_endpoint_fit_is_API_supply_accuracy=False,
        conclusion='v7 spreads feasibility slack over issue depth and collapses measured preparation before first activity; ordinary first issue depth is 1, not 0')


def main():
    p=argparse.ArgumentParser();p.add_argument('--parameters',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    result=derive(a.parameters);save(out/'report.json',result)
    print(json.dumps(dict(report=receipt(out/'report.json'),rows=len(result['rows']))))

if __name__=='__main__':main()
