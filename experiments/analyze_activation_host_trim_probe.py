"""Feasibility only: CPU RSS receipts, complete trim lifecycle, unchanged outputs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.fit_activation_native import load_bundle
from experiments.analyze_activation_semantic import output_identity
from experiments.analyze_activation_host_diagnostic_v2 import validate_trace
from gsdeploy.forecasting.native_training import validate_capture


def analyze(directory):
    bundle=load_bundle(directory);raw=bundle['raw'];plan=bundle['plan']
    validate_capture(bundle,allowed_splits=('development_host_trim_feasibility',))
    reference=load_bundle(Path(plan['original_capture']));validate_capture(reference,allowed_splits=('development_host_diagnostic',))
    if reference['source']!=plan['original_source'] or plan.get('formal_latency_benefit_claimed') is not False:
        raise ValueError('Exact reference and feasibility-only scope required')
    ctx=copy.deepcopy(bundle['context']);old=reference['context']
    if (ctx.pop('observer_diagnostic')!='five_model_postprime_host_trim_feasibility'
            or ctx.pop('postprime_host_trim')!=dict(mode='on',api='glibc malloc_trim(0)',
                timing='after full declared priming before worker READY',gc_policy_changes=False,extra_cuda_sync=False)):
        raise ValueError('Heap trim policy mismatch')
    for alias,binding in ctx['worker_bindings'].items():
        if (binding['env'].pop('GS_POSTPRIME_HOST_TRIM')!='on'
                or Path(binding['command'][1]).name!='postprime_trim_scoped_worker.py'):raise ValueError('Wrong trim worker')
        binding['command'][1]=old['worker_bindings'][alias]['command'][1]
    ctx['source_content']=old['source_content'];ctx['observer_diagnostic']=old['observer_diagnostic']
    if ctx!=old:raise ValueError('Runtime changed beyond declared heap trim')
    receipts=[];issues=[]
    for alias,ready in raw['ready'].items():
        for key in ('runtime_scope','passive_gc_policy','passive_host_policy','spec_sha256','torch'):
            if ready.get(key)!=reference['raw']['ready'][alias].get(key):raise ValueError('Actual model/runtime policy drift')
        policy=ready['postprime_trim_policy'];primes=[r for r in raw['priming_request_records'] if r['model']==alias]
        if policy['mode']!='on' or policy['forced_gc'] or policy['gc_thresholds_changed'] or policy['extra_cuda_sync']:
            raise ValueError('Actual trim policy mismatch')
        if len(primes)!=policy['priming_requests']:raise ValueError('Incomplete matched priming')
        found=[]
        for row in primes:
            boundary=row['previous_handler_next_request_ready'];response=row['worker_response']
            validate_trace(boundary['passive_gc_lifecycle_trace'],response['start_perf_ns'],boundary['perf_ns'],ready['passive_gc_policy'])
            if 'postprime_trim_receipt' in boundary:found.append((row,boundary['postprime_trim_receipt']))
        if len(found)!=1 or found[0][0]['request_id']!=primes[-1]['request_id']:raise ValueError('Trim must occur only after final priming')
        row,receipt=found[0];boundary=row['previous_handler_next_request_ready'];response=row['worker_response']
        if (receipt['mode']!='on' or receipt['allocator_return'] not in (0,1)
                or receipt['gc_policy_unchanged'] is not True or receipt['extra_cuda_sync'] is not False
                or receipt['live_allocations_released'] is not False
                or not response['end_perf_ns']<=receipt['procedure_start_ns']<=receipt['allocator_call_start_ns']
                    <=receipt['allocator_call_end_ns']<=receipt['procedure_end_ns']<=boundary['perf_ns']):
            raise ValueError('Trim cost omitted from full worker lifecycle or invalid receipt')
        before,after=receipt['host_memory_before'],receipt['host_memory_after']
        receipts.append(dict(alias=alias,request_id=row['request_id'],receipt=receipt,
            rss_reduction_mib=(before['VmRSS_kib']-after['VmRSS_kib'])/1024,
            anonymous_rss_reduction_mib=(before['RssAnon_kib']-after['RssAnon_kib'])/1024,
            procedure_ms=(receipt['procedure_end_ns']-receipt['procedure_start_ns'])/1e6))
    lookup={(r['model'],r['task'],r['pool_id']):r for r in reference['raw']['request_records']}
    for row in raw['request_records']:
        oldrow=lookup[row['model'],row['task'],row['pool_id']]
        if output_identity(row['worker_response'])!=output_identity(oldrow['worker_response']):issues.append(row['request_id']+':OUTPUT_CHANGED')
        if 'postprime_trim_receipt' in row['previous_handler_next_request_ready']:raise ValueError('Undeclared formal trim')
    return dict(schema='gsdeploy.host_heap_trim_feasibility_result.v1',source=bundle['source'],reference=reference['source'],
        receipts=receipts,output_issues=issues,formal_requests=len(raw['request_records']),
        latency_benefit_established=False,physical_predictor_qualified=False,formal_utility_qualified=False,
        limitations='Single startup/priming feasibility observation with eight fixed output sentinels. RSS change does not identify freed allocation ownership or prove latency improvement. Complete trim cost remains inside startup/priming worker-ready lifecycle.',
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=analyze(a.capture.resolve())
    with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output_issues=result['output_issues'],rss_reduction_mib={r['alias']:r['rss_reduction_mib'] for r in result['receipts']},latency_benefit_established=False)))
