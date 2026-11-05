"""Unchanged observer gates in the selected scoped five-model runtime."""
import argparse
import copy
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.fit_activation_native import load_bundle
from experiments.analyze_activation_semantic import output_identity,summarize
from experiments.diagnose_activation_semantic_repeats import keyed_rows
from experiments.activation_completion_bands import completion_band_layout as band_layout
from gsdeploy.forecasting.native_training import validate_capture, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    criteria=json.loads((args.base/'observer-criteria.json').read_text());names=['control-round1','semantic-round1','semantic-round2','control-round2']
    design=json.loads((args.base/'design.json').read_text())
    if (digest(criteria)!=design['original_criteria_sha256']
            or criteria['maximum_mean_absolute_e2e_deviation']!=.1
            or criteria['maximum_p95_absolute_e2e_deviation']!=.25):
        raise ValueError('Frozen original observer criteria changed')
    bundles={name:load_bundle(args.base/name) for name in names};keyed={};issues=[]
    expected={family+'#0':family for family in ('generalist_2','text','asr','image','ocr')}
    for name,bundle in bundles.items():
        raw,plan=bundle['raw'],bundle['plan']
        validate_capture(bundle,allowed_splits=('development_observer_diagnostic',))
        scope=plan['runtime_scope']
        if scope!=bundle['context']['runtime_policy']['scope'] or scope!=design['scope']:
            raise ValueError('Declared scope differs from context')
        for alias,family in raw['resident_aliases'].items():
            e=raw['ready'][alias].get('runtime_scope',{});strict=scope=='all_strict' or family=='generalist_2'
            if (e.get('scope')!=scope or e.get('slot')!=family or e.get('strict_requested') is not strict
                    or e.get('deterministic_algorithms') is not strict
                    or e.get('cublas_workspace_config')!=(':4096:8' if strict else None)):
                raise ValueError('Actual READY runtime settings differ from declared policy')
            if strict and (e.get('deterministic_warn_only') is not False or e.get('cudnn_benchmark') is not False
                           or e.get('cudnn_deterministic') is not True):
                raise ValueError('Incomplete strict runtime receipt')
        planned=Counter((job['id'],row['alias'],row['task'],row['pool_id']) for job in plan['jobs'] for row in job['requests'])
        observed=Counter((row['profile_job_id'],row['model'],row['task'],row['pool_id']) for row in raw['request_records'])
        if (raw['status']!='completed' or raw['split']!='development_observer_diagnostic' or planned!=observed
                or raw['resident_aliases']!=expected or raw['execution_context_sha256']!=plan['execution_context_sha256']
                or any(raw.get(key) for key in ('errors','runtime_errors','telemetry_errors'))
                or any(sample.get('foreign_pids') for sample in raw['board_samples'])
                or any(item['forced'] or item['exit_code']!=0 or not item['bye_received'] for item in raw['shutdown'].values())):
            raise ValueError('Complete clean, planned five-model observer capture required')
        for row in raw['priming_request_records']+raw['request_records']:
            response=row['worker_response']
            if response.get('success') is not True:issues.append(name+':'+row['request_id']+':OUTPUT_SUCCESS_UNVERIFIED')
            if row['model']=='generalist_2#0':
                modifier=response.get('memory_optimizations',{}).get('exllama_reconstruct_runtime',{})
                if (modifier.get('matmul_recons_thd')!=1 or modifier.get('configured_after_model_load') is not True
                        or modifier.get('quantized_weights_changed') is not False or modifier.get('quantized_layers',0)<=0):
                    issues.append(name+':'+row['request_id']+':RUNTIME_PATH_UNVERIFIED')
        keyed[name]=keyed_rows(bundle)
    if any(set(rows)!=set(keyed[names[0]]) for rows in keyed.values()):raise ValueError('Cross-round cohort differs')
    contexts=[]
    for name,bundle in bundles.items():
        ctx=copy.deepcopy(bundle['context'])
        mode='complete' if name.startswith('control-') else 'semantic'
        if ctx.pop('observer_diagnostic')!='five_model_scoped_'+mode:
            raise ValueError('Actual observer context differs from declared comparison arm')
        for binding in ctx['worker_bindings'].values():
            if binding['env'].pop('GS_SCOPED_OBSERVER')!=mode:
                raise ValueError('Worker observer mode differs')
        contexts.append(ctx)
    if any(ctx!=contexts[0] for ctx in contexts[1:]):
        raise ValueError('Observer comparison changed more than instrumentation mode')
    metrics=defaultdict(list);pairs=[]
    for index in (1,2):
        left,right=f'control-round{index}',f'semantic-round{index}';changes=[]
        if bundles[left]['plan']['matched_job_content_sha256']!=bundles[right]['plan']['matched_job_content_sha256']:
            raise ValueError('Within-round observer jobs differ')
        for key,old in keyed[left].items():
            row=keyed[right][key];family=bundles[right]['raw']['resident_aliases'][row['model']]
            label=right+':'+row['request_id'];response=row['worker_response'];path=family+'/'+row['task']
            if output_identity(old['worker_response'])!=output_identity(response):changes.append(key);issues.append(label+':OUTPUT_CHANGED')
            events=response.get('semantic_phase_events',[]);expected_component=bundles[right]['plan']['expected_semantic_components'][path]
            if expected_component not in {event['component'] for event in events}:issues.append(label+':MISSING_EXPECTED_COMPONENT:'+expected_component)
            try:band_layout(row,family)
            except Exception as exc:issues.append(label+':INVALID_DERIVED_BAND:'+type(exc).__name__+':'+str(exc))
            old_ms=old['completion_time_ns']-old['scheduled_time_ns'];new_ms=row['completion_time_ns']-row['scheduled_time_ns']
            metrics[path].append(abs(new_ms-old_ms)/old_ms)
        pairs.append(dict(left=left,right=right,output_changes=changes))
    repeats={}
    for left,right in [('control-round1','control-round2'),('semantic-round1','semantic-round2')]:
        changed=[key for key,row in keyed[left].items() if output_identity(row['worker_response'])!=output_identity(keyed[right][key]['worker_response'])]
        repeats[left+' vs '+right]=changed
        if changed:issues.append(left+' vs '+right+':REPEAT_OUTPUT_CHANGED')
    path_metrics={path:summarize(values) for path,values in sorted(metrics.items())}
    for path,value in path_metrics.items():
        if value['mean']>criteria['maximum_mean_absolute_e2e_deviation'] or value['p95']>criteria['maximum_p95_absolute_e2e_deviation']:
            issues.append(path+':OBSERVER_OR_REPEAT_VARIATION_TOO_LARGE')
    result=dict(schema='gsdeploy.five_model_scoped_observer_diagnostic.v1',criteria=criteria,
        sources={name:bundle['source'] for name,bundle in bundles.items()},pairs=pairs,repeat_output_changes=repeats,
        path_metrics=path_metrics,issues=issues,observer_repeat_check_passed=not issues,
        physical_predictor_qualified=False,phase_templates_qualified=False,formal_utility_qualified=False,
        runtime_scope=scope,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='selected runtime scope with actual READY receipts; natural variation plus instrumentation; no physical predictor qualification')
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ['pairs','sources']},ensure_ascii=False))


if __name__=='__main__':main()
