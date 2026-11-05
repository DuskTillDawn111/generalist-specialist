"""Retain host/device gaps with prior owner, idle duration and cache receipts.

The two clocks measure different scopes: host minus CUDA interval is only a
numerical contrast, not pure host work, GPU inactivity or a scheduler estimate.
No observations are removed or used as an after-the-fact prediction input.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.fit_activation_native import load_bundle
from experiments.analyze_activation_semantic import summarize
from gsdeploy.forecasting.native_training import validate_capture


def analyze(captures):
    observations=[];sources=[]
    for path in captures:
        bundle=load_bundle(path.resolve());raw=bundle['raw'];sources.append(bundle['source'])
        validate_capture(bundle,allowed_splits=('training','development_gc_diagnostic','development_host_diagnostic'))
        rows=sorted(raw['priming_request_records']+raw['request_records'],key=lambda r:r['worker_response']['start_perf_ns'])
        previous_owner={};previous_global=None
        for row in rows:
            w=row['worker_response'];alias=row['model'];prev=previous_owner.get(alias);pg=previous_global
            previous_owner[alias]=row;previous_global=row
            if row not in raw['request_records']:continue
            start=w['start_perf_ns']
            if pg and pg['previous_handler_next_request_ready']['perf_ns']>start:
                raise ValueError('This history contrast requires serial requests')
            trace=w.get('passive_gc_trace');usage=None
            if trace:usage={k:trace['usage_at_result_hook'][k]-v for k,v in trace['usage_at_first_phase_hook'].items()}
            for event in w['semantic_phase_events']:
                if not event['device_event_complete'] or event['device_elapsed_ms'] is None:raise ValueError('Complete semantic intervals required')
                host_ms=(event['host_end_ns']-event['host_start_ns'])/1e6
                observations.append(dict(capture=str(path.resolve()),request_id=row['request_id'],pool_id=row['pool_id'],
                    path=raw['resident_aliases'][alias]+'/'+row['task'],component=event['component'],ordinal=event['ordinal'],
                    host_ms=host_ms,device_interval_ms=event['device_elapsed_ms'],
                    host_minus_device_ms=host_ms-event['device_elapsed_ms'],
                    previous_owner_request_id=prev['request_id'] if prev else None,
                    owner_idle_since_ready_ms=(start-prev['previous_handler_next_request_ready']['perf_ns'])/1e6 if prev else None,
                    previous_owner_release=prev['worker_response'].get('cache_reclamation') if prev else None,
                    previous_owner_memory=prev['worker_response']['memory'] if prev else None,
                    previous_global_request_id=pg['request_id'] if pg else None,
                    previous_global_path=raw['resident_aliases'][pg['model']]+'/'+pg['task'] if pg else None,
                    previous_global_handler_ms=(pg['worker_response']['end_perf_ns']-pg['worker_response']['start_perf_ns'])/1e6 if pg else None,
                    current_memory=w['memory'],current_release=w.get('cache_reclamation'),
                    request_usage_delta=usage,host_bracket=event.get('passive_host_bracket')))
    groups=defaultdict(list)
    for row in observations:
        if row['component']=='prepare_and_media':groups[row['path']].append(row)
    summaries={p:dict(n=len(rows),host_minus_device_ms=summarize([r['host_minus_device_ms'] for r in rows]),
        previous_release_counts={str(v):sum((r['previous_owner_release'] or {}).get('performed') is v for r in rows) for v in (True,False,None)},
        diagnostic_gap_over_10ms=sum(r['host_minus_device_ms']>10 for r in rows)) for p,rows in groups.items()}
    return dict(schema='gsdeploy.host_history_diagnostic.v1',sources=sources,observations=observations,
        prepare_summaries=summaries,physical_predictor_qualified=False,formal_utility_qualified=False,
        limitations='All events retained. A 10ms contrast is a diagnostic display threshold, never an accuracy gate. Prior cache release is correlation only; host/CUDA clock scopes differ.',
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=analyze(a.capture)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result['prepare_summaries']))
