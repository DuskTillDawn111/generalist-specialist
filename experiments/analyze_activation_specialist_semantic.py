"""Apply the unchanged observer thresholds to four-specialist matched pilots."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.fit_activation_native import load_bundle
from experiments.analyze_activation_semantic import output_identity, summarize
from experiments.diagnose_activation_semantic_repeats import keyed_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    criteria = json.loads((args.base/'observer-criteria.json').read_text())
    names = ['control-round1','semantic-round1','semantic-round2','control-round2']
    bundles = {name: load_bundle(args.base/name) for name in names}
    expected_aliases={family+'#0':family for family in ('text','image','ocr','asr')}
    keyed, issues = {}, []
    for name, bundle in bundles.items():
        raw, plan = bundle['raw'], bundle['plan']
        expected = Counter((job['id'], row['alias'], row['task'], row['pool_id'])
                           for job in plan['jobs'] for row in job['requests'])
        observed = Counter((row['profile_job_id'],row['model'],row['task'],row['pool_id'])
                           for row in raw['request_records'])
        if (raw['status']!='completed' or raw['split']!='development_observer_diagnostic'
                or any(raw.get(key) for key in ('errors','runtime_errors','telemetry_errors'))
                or raw['execution_context_sha256']!=plan['execution_context_sha256']
                or raw['resident_aliases']!=expected_aliases
                or expected!=observed):
            raise ValueError('Complete matched specialist observer cohort required')
        if any(sample.get('foreign_pids') for sample in raw['board_samples']):
            raise ValueError('Foreign GPU processes in observer cohort')
        if any(item['forced'] or item['exit_code']!=0 or not item['bye_received'] for item in raw['shutdown'].values()):
            raise ValueError('Unclean observer worker shutdown')
        if any(row['worker_response'].get('success') is not True
               for row in raw['request_records']+raw['priming_request_records']):
            issues.append(name+':OUTPUT_SUCCESS_UNVERIFIED')
        keyed[name] = keyed_rows(bundle)
    if any(set(rows)!=set(keyed[names[0]]) for rows in keyed.values()):
        raise ValueError('Different cross-round input/job cohorts')
    deviations = defaultdict(list)
    comparisons = []
    for index in (1,2):
        left,right=f'control-round{index}',f'semantic-round{index}'
        if bundles[left]['plan']['matched_job_content_sha256']!=bundles[right]['plan']['matched_job_content_sha256']:
            raise ValueError('Within-round observer job order changed')
        differences=[]
        for key, old in keyed[left].items():
            new=keyed[right][key]
            baseline,current=old['worker_response'],new['worker_response']
            label=right+':'+new['request_id']
            if output_identity(baseline)!=output_identity(current):
                differences.append(dict(job=key[0],family=key[1],task=key[2],pool_id=key[3]))
                issues.append(label+':OUTPUT_CHANGED')
            events=current.get('semantic_phase_events',[])
            expected=bundles[right]['plan']['expected_semantic_components'][new['task']]
            if not events or expected not in {event['component'] for event in events}:
                issues.append(label+':MISSING_EXPECTED_COMPONENT:'+expected)
            for event in events:
                if (event['request_id']!=new['request_id'] or not event.get('device_event_complete')
                        or event.get('device_elapsed_ms') is None or event['device_elapsed_ms']<0
                        or event['stream_id']!=event['end_current_stream_id']
                        or not current['start_perf_ns']<=event['host_start_ns']<=event['host_end_ns']<=current['end_perf_ns']):
                    issues.append(label+':INVALID_SEMANTIC_EVENT')
            old_ms=(old['completion_time_ns']-old['scheduled_time_ns'])/1e6
            new_ms=(new['completion_time_ns']-new['scheduled_time_ns'])/1e6
            deviations[new['task']].append(abs(new_ms-old_ms)/old_ms)
        comparisons.append(dict(left=left,right=right,output_differences=differences))
    repeat_changes={}
    for left,right in [('control-round1','control-round2'),('semantic-round1','semantic-round2')]:
        changes=[key for key,row in keyed[left].items()
                 if output_identity(row['worker_response'])!=output_identity(keyed[right][key]['worker_response'])]
        repeat_changes[left+' vs '+right]=changes
        if changes:issues.append(left+' vs '+right+':REPEAT_OUTPUT_CHANGED')
    path_metrics={path:summarize(values) for path,values in deviations.items()}
    for path,values in path_metrics.items():
        if (values['mean']>criteria['maximum_mean_absolute_e2e_deviation']
                or values['p95']>criteria['maximum_p95_absolute_e2e_deviation']):
            issues.append(path+':OBSERVER_OR_REPEAT_VARIATION_TOO_LARGE')
    report=dict(schema='gsdeploy.specialist_semantic_observer_diagnostic.v1',criteria=criteria,
        sources={name:bundle['source'] for name,bundle in bundles.items()},comparisons=comparisons,
        repeat_output_changes=repeat_changes,path_metrics=path_metrics,issues=issues,
        observer_repeat_check_passed=not issues,phase_templates_qualified=False,
        physical_predictor_qualified=False,formal_utility_qualified=False,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='new four-specialist observer diagnostic only; no future decode stopping oracle')
    with args.output.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
