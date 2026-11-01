"""Endpoint-constrained host-launch supply hypothesis, never natural kernel truth.

Natural training handler time and light-domain kernel durations are separate
observations. Their nonnegative residual is partitioned by observed CUDA launch
API elapsed-time weights, not by profiler gaps. Kernel durations are unchanged.
This fits exact seen training inputs only; calibration identity is not accuracy.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sqlite3

from gsdeploy.resources import identity
from .native_collect import receipt
from .kernel_sequence_audit_v2 import verify
from .kernel_domain_bridge_review_v3 import load_report
from .ordered_kernel_profiles_v2 import SCHEMA as ORDERED_SCHEMA, OrderedKernelProfileProvider
from .trace_export import linux_pid

SCHEMA='gsdeploy.launch_supply_residual_hypothesis.v2'


def partition_nonnegative(total,weights):
    """Exact integer conservation; no negative clipping or kernel rescaling."""
    if type(total) is not int or total<0 or not weights or any(type(w) is not int or w<=0 for w in weights):
        raise ValueError('Nonnegative residual and positive integer API elapsed weights required')
    denominator=sum(weights)
    parts=[total*w//denominator for w in weights]
    remainder=total-sum(parts)
    order=sorted(range(len(weights)),key=lambda i:(-(total*weights[i]%denominator),i))
    for index in order[:remainder]:parts[index]+=1
    return parts


def fit_profile(profile,natural_handler_ns,api_elapsed_ns):
    kernels=[s for s in profile['segments'] if s['kind']=='gpu']
    durations=[round(s['duration_ms']['profiled']*1e6) for s in kernels]
    if len(api_elapsed_ns)!=len(kernels):raise ValueError('One exact launch API elapsed per kernel required')
    residual=natural_handler_ns-sum(durations)
    if residual<0:
        return dict(profile_id=profile['id'],known=False,reason='NATURAL_HANDLER_SHORTER_THAN_LIGHT_KERNEL_SUM',
                    natural_handler_ns=natural_handler_ns,light_kernel_sum_ns=sum(durations),residual_ns=residual)
    parts=partition_nonnegative(residual,api_elapsed_ns)
    return dict(profile_id=profile['id'],alias=profile['alias'],task=profile['task'],pool_id=profile['pool_id'],
        known=True,natural_handler_ns=natural_handler_ns,light_kernel_sum_ns=sum(durations),residual_ns=residual,
        supply_before_kernel_ns=[dict(segment_id=s['id'],duration_ns=p,launch_api_elapsed_ns=w)
                                 for s,p,w in zip(kernels,parts,api_elapsed_ns)],
        hypothesis='Independent host-launch supply shaped by profiled CUDA launch API elapsed, fitted to natural handler endpoint',
        weights_are_cpu_execution_measurements=False,kernel_durations_rescaled=False,profiled_gaps_scaled=False,
        in_sample_endpoint_identity=True,independent_prediction_error_evaluated=False,natural_timing_validated=False)


def fit(ordered_path,bridge_path):
    ordered_path=Path(ordered_path).resolve(); ordered=json.loads(ordered_path.read_text())
    bridge_path=Path(bridge_path).resolve();bridge=json.loads(bridge_path.read_text())
    if (ordered.get('schema')!=ORDERED_SCHEMA or ordered.get('provenance',{}).get('split')!='training'
            or bridge.get('schema')!='gsdeploy.kernel_three_domain_bridge_review.v3'
            or bridge.get('status')!='PASS' or bridge.get('training_only') is not True):
        raise ValueError('Training-only ordered profiles and passing exact work bridge required')
    verify(bridge['evaluator'])
    for entry in bridge['requests']:verify(entry['mapping'])
    reports={}
    for domain,key in [('light-kernel','light_review'),('natural','natural_review'),('metadata','metadata_review')]:
        verify(bridge[key]);reports[domain]=load_report(bridge[key]['path'],domain)
    lr,lp,lraw=reports['light-kernel']; nr,np,nraw=reports['natural']
    if ordered['execution_context_sha256']!=lraw['execution_context_sha256'] or lp['jobs']!=np['jobs']:
        raise ValueError('Supply fit domains do not bind ordered training execution/jobs')
    light_rows={r['request_id']:r for r in lraw['request_records']}
    natural_rows={(r['profile_job_id'],r['model'],r['pool_id']):r for r in nraw['request_records']}
    work={(r['trial_job_id'],r['alias'],r['pool_id']):r for r in bridge['requests']}
    launches={};fits=[]
    for profile in ordered['profiles']:
        row=light_rows[profile['id']];request_key=(row['profile_job_id'],row['model'],row['pool_id'])
        matched=work.get(request_key)
        if not matched or not matched['exact_work_matched'] or not matched['all_observed_kernel_groups_matched']:
            raise ValueError('Supply fit lacks exact observed work bridge')
        natural=natural_rows[request_key]['worker_response']
        elapsed=[]
        for segment in profile['segments']:
            if segment['kind']!='gpu':continue
            source=segment['timing_evidence']['source'];sha=source['sha256']
            if sha not in launches:
                verify(source)
                connection=sqlite3.connect(Path(source['path']).as_uri()+'?mode=ro',uri=True)
                connection.row_factory=sqlite3.Row
                try:
                    launches[sha]={(linux_pid(r['globalTid']),r['correlationId']):dict(r)
                        for r in connection.execute('select r.*,s.value as api_name from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id')}
                finally:connection.close()
            evidence=segment['timing_evidence']['identity']; launch=launches[sha].get((evidence[2],evidence[4]))
            if (not launch or not launch['api_name'].startswith(('cudaLaunchKernel','cuLaunchKernel')) or launch['returnValue']!=0
                    or launch['end']<=launch['start']):
                raise ValueError('Source-bound successful positive-duration kernel launch API required')
            elapsed.append(launch['end']-launch['start'])
        fits.append(fit_profile(profile,natural['end_perf_ns']-natural['start_perf_ns'],elapsed))
    result=dict(schema=SCHEMA,execution_context_sha256=ordered['execution_context_sha256'],
        natural_execution_context_sha256=nraw['execution_context_sha256'],training_only=True,
        sources=[receipt(ordered_path),receipt(bridge_path),receipt(__file__)],fits=fits,
        denominator=ordered['denominator'],independent_training_captures=bridge['independent_captures_per_domain'],
        minimum_training_capture_requirement_satisfied=bridge['minimum_training_capture_requirement_satisfied'],
        unmodeled=['Cross-owner host CPU contention','Memcpy/memset DRAM interference',
                  'Unseen input/kernel sequences','Clock placement uncertainty','Multi-stream scheduling'],
        independent_prediction_error_evaluated=False,physically_validated=False,deployment_qualified=False,
        natural_timing_validated=False,kernel_durations_rescaled=False,profiled_gaps_scaled=False)
    result['fit_sha256']=identity(result)
    return result


class LaunchSupplyResidualProvider(OrderedKernelProfileProvider):
    """FullHorizonModel prototype with explicit host/device independence opt-in.

    Supply progress one is a model assumption with no CPU measurement claim.
    Removing those rows from GPU packing assumes no unmodeled copy activity;
    the caller must opt in separately. GPU occupancy-to-progress and cross-SM
    capability assumptions remain separately controlled in the base provider.
    """
    def __init__(self,ordered,residual,execution_context_sha256,*,
                 allow_independent_host_supply_hypothesis=False,
                 device_progress_hypothesis='static_packing',context_switch_ms=0.,**options):
        if (residual.get('schema')!=SCHEMA or residual.get('execution_context_sha256')!=execution_context_sha256
                or residual.get('training_only') is not True):
            raise ValueError('Context-bound training supply residual required')
        if type(allow_independent_host_supply_hypothesis) is not bool:
            raise ValueError('Explicit boolean independent host supply hypothesis required')
        if device_progress_hypothesis not in ('static_packing','equal_context_time_slice'):
            raise ValueError('Explicit supported GPU scheduler hypothesis required')
        if type(context_switch_ms) not in (int,float) or not math.isfinite(context_switch_ms) or context_switch_ms<0:
            raise ValueError('Nonnegative finite hypothesized per-kernel context switch cost required')
        self.device_progress_hypothesis=device_progress_hypothesis
        self.context_switch_ms=context_switch_ms
        transformed=copy.deepcopy(ordered)
        fits={r['profile_id']:r for r in residual['fits'] if r['known']}
        supported=[]
        for profile in transformed['profiles']:
            fitrow=fits.get(profile['id'])
            if fitrow is None:continue
            if (fitrow['alias'],fitrow['task'],fitrow['pool_id'])!=(profile['alias'],profile['task'],profile['pool_id']):
                raise ValueError('Residual profile input/owner identity mismatch')
            kernels=[s for s in profile['segments'] if s['kind']=='gpu'];supplies=fitrow['supply_before_kernel_ns']
            if [s['id'] for s in kernels]!=[s['segment_id'] for s in supplies]:
                raise ValueError('Residual kernel sequence mismatch')
            kernel_sum=sum(round(s['duration_ms']['profiled']*1e6) for s in kernels)
            if (kernel_sum!=fitrow['light_kernel_sum_ns'] or
                    sum(s['duration_ns'] for s in supplies)!=fitrow['residual_ns'] or
                    fitrow['natural_handler_ns']!=kernel_sum+fitrow['residual_ns'] or
                    any(type(s['duration_ns']) is not int or s['duration_ns']<0 for s in supplies)):
                raise ValueError('Source-bound residual/kernel/endpoint conservation mismatch')
            profile['segments']=[]
            for kernel,supply in zip(kernels,supplies):
                if supply['duration_ns']:
                    profile['segments'].append(dict(id='supply-before-'+kernel['id'],kind='unclassified_gap',
                        duration_ms=dict(profiled=supply['duration_ns']/1e6),
                        evidence=dict(residual_fit_sha256=residual.get('fit_sha256'),
                                      launch_api_elapsed_ns=supply['launch_api_elapsed_ns']),
                        model_role='HYPOTHETICAL_HOST_LAUNCH_SUPPLY_NOT_MEASURED_CPU',natural_timing_validated=False))
                profile['segments'].append(kernel)
            supported.append(profile)
        transformed['profiles']=supported
        self.allow_independent_host_supply_hypothesis=allow_independent_host_supply_hypothesis
        options['allow_isolated_profiled_replay']=True
        super().__init__(transformed,execution_context_sha256,**options)

    def service(self,row,active,aliases,scenario):
        # Run base identity validation even when independence is opted in.
        self._validated_sources(row,active,aliases,scenario)
        if not self.allow_independent_host_supply_hypothesis:
            return self._unknown('INDEPENDENT_HOST_SUPPLY_HYPOTHESIS_OPT_IN_REQUIRED')
        source=self.segments.get((row.get('profile_id'),row.get('segment_id')))
        if source is None:raise ValueError('Unknown residual segment')
        if source['kind']=='unclassified_gap':
            result=super().service(row,{row['alias']:row},aliases,scenario)
        else:
            gpu_active={alias:r for alias,r in active.items() if r['segment_kind']=='gpu'}
            if self.device_progress_hypothesis=='equal_context_time_slice':
                count=len(gpu_active)
                result=dict(known=True,context_supported=True,
                    service_ms=row['segment_duration_ms']*count+self.context_switch_ms*(count-1),
                    assumed_relative_progress=1./count,
                    context_switch_ms=self.context_switch_ms,
                    switch_cost_scope='Per-active-kernel scenario assumption; quantum/preemption mechanics unvalidated')
            else:result=super().service(row,gpu_active,aliases,scenario)
        return dict(result,status='HOST_SUPPLY_RESIDUAL_HYPOTHESIS',
                    host_supply_independence_assumed=True,unmodeled_device_copy_interference_assumed_absent=True,
                    device_progress_hypothesis=self.device_progress_hypothesis,
                    natural_timing_validated=False,physically_validated=False,deployment_qualified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ordered',required=True);parser.add_argument('--bridge',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=fit(args.ordered,args.bridge)
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'fit.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(known=sum(r['known'] for r in result['fits']),unknown=sum(not r['known'] for r in result['fits']))))


if __name__=='__main__':main()
