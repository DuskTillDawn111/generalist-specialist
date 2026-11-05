"""Audit paired observer pilots without claiming fine-phase predictor accuracy."""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.fit_activation_native import load_bundle
from gsdeploy.forecasting.native_training import digest


def summarize(values):
    values=sorted(values)
    return dict(n=len(values),mean=statistics.mean(values) if values else None,
                p95=values[min(len(values)-1,round(.95*(len(values)-1)))] if values else None,
                maximum=max(values) if values else None)


def output_identity(response):
    if response.get('output_sha256') is not None:return response['output_sha256']
    return digest({k:response.get(k) for k in ('prediction','raw_output','success','finish_reason')})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();base=args.base.resolve()
    criteria=json.loads((base/'observer-criteria.json').read_text());pairs=[];issues=[];deviations=defaultdict(list)
    for round_no in (1,2):
        control=load_bundle(base/f'control-round{round_no}');semantic=load_bundle(base/f'semantic-round{round_no}')
        if control['plan']['matched_job_content_sha256']!=semantic['plan']['matched_job_content_sha256']:
            raise ValueError('Observer comparison changed request jobs')
        for bundle in (control,semantic):
            raw=bundle['raw']
            if (raw['status']!='completed' or raw['split']!='development_observer_diagnostic'
                    or any(raw.get(k) for k in ('errors','runtime_errors','telemetry_errors'))
                    or raw['execution_context_sha256']!=bundle['plan']['execution_context_sha256']):
                raise ValueError('Complete clean observer diagnostic captures required')
            if any(r['worker_response'].get('success') is not True for r in raw.get('priming_request_records',[])):
                issues.append(raw['run_id']+':PRIMING_OUTPUT_UNVERIFIED')
        crows=control['raw']['request_records'];srows=semantic['raw']['request_records']
        if len(crows)!=len(srows):raise ValueError('Observer dropped requests')
        requests=[]
        for old,new in zip(crows,srows):
            for key in ('request_id','model','task','pool_id','profile_job_id'):
                if old[key]!=new[key]:raise ValueError('Observer comparison cohort mismatch')
            baseline,current=old['worker_response'],new['worker_response'];events=current.get('semantic_phase_events',[])
            label=f'round{round_no}:'+new['request_id'];path=new['model']+'/'+new['task']
            if baseline.get('success') is not True or current.get('success') is not True:
                issues.append(label+':OUTPUT_SUCCESS_UNVERIFIED')
            if output_identity(baseline)!=output_identity(current):issues.append(label+':OUTPUT_CHANGED')
            if not events:issues.append(label+':NO_SEMANTIC_EVENTS')
            for event in events:
                if (event['request_id']!=new['request_id'] or not event.get('device_event_complete')
                        or event.get('device_elapsed_ms') is None or event['device_elapsed_ms']<0
                        or event['stream_id']!=event['end_current_stream_id']
                        or not current['start_perf_ns']<=event['host_start_ns']<=event['host_end_ns']<=current['end_perf_ns']):
                    issues.append(label+':INVALID_SEMANTIC_EVENT')
            expected=('language' if new['model'].startswith('generalist_2') else
                      'image_features' if new['model'].startswith('image') else 'ocr_detection')
            if expected not in {e['component'] for e in events}:issues.append(label+':MISSING_EXPECTED_COMPONENT:'+expected)
            old_ms=(old['completion_time_ns']-old['scheduled_time_ns'])/1e6
            new_ms=(new['completion_time_ns']-new['scheduled_time_ns'])/1e6
            deviation=abs(new_ms-old_ms)/old_ms;deviations[path].append(deviation)
            components=defaultdict(list)
            for event in events:
                if event.get('device_elapsed_ms') is not None:components[event['component']].append(event['device_elapsed_ms'])
            requests.append(dict(request_id=new['request_id'],path=path,pool_id=new['pool_id'],job=new['profile_job_id'],
                control_e2e_ms=old_ms,semantic_e2e_ms=new_ms,absolute_relative_e2e_deviation=deviation,
                signed_e2e_ratio=new_ms/old_ms,semantic_event_count=len(events),
                device_elapsed_by_component={k:summarize(v) for k,v in components.items()}))
        pairs.append(dict(round=round_no,control_source=control['source'],semantic_source=semantic['source'],requests=requests))
    path_metrics={k:summarize(v) for k,v in deviations.items()}
    for path,metrics in path_metrics.items():
        if (metrics['mean']>criteria['maximum_mean_absolute_e2e_deviation']
                or metrics['p95']>criteria['maximum_p95_absolute_e2e_deviation']):issues.append(path+':OBSERVER_OR_REPEAT_VARIATION_TOO_LARGE')
    report=dict(schema='gsdeploy.semantic_observer_diagnostic.v1',criteria=criteria,pairs=pairs,
                analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                path_metrics=path_metrics,issues=issues,observer_repeat_check_passed=not issues,
                physical_predictor_qualified=False,phase_templates_qualified=False,formal_utility_qualified=False,
                scope='paired natural-run observer deviation; runtime variation and instrumentation effects not causally separated')
    with args.output.open('x') as out:json.dump(report,out,ensure_ascii=False,indent=2);out.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='pairs'},ensure_ascii=False))


if __name__=='__main__':main()
