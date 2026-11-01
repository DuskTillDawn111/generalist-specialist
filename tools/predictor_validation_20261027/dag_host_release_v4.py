"""Partially identified DAG host-supply scenarios from natural training bounds.

Natural handler duration alone cannot identify per-launch CPU supply. Under an
explicit transfer of isolated light-domain activity targets, this module fits
two different feasible release shapes; it never rescales GPU work or uses
profiled API issue offsets / observed handler anchors in a future prediction.
"""
from __future__ import annotations

import argparse
from collections import defaultdict,deque
import heapq
import json
import math
from pathlib import Path
import statistics

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.resources import identity
from .composite_execution_v2 import SCHEMA as COMPOSITE_SCHEMA
from .kernel_domain_bridge_review_v3 import load_report as load_v3
from .native_collect import receipt,save

SCHEMA='gsdeploy.dag_host_supply_training_hypothesis.v4'
SCENARIOS=('front_loaded_supply','host_issue_depth_supply')
STRUCTURAL={'event_record','event_wait','host_sync','handler_end'}
ACTIVITY={'gpu','copy','memset'}


def structure(profile):
    """Only source order and dependencies; target durations/releases are unused."""
    if profile.get('cyclic') or profile.get('unresolved') or not profile.get('nodes'):
        raise UnknownEvidence('RESOLVED_NONEMPTY_CUDA_DAG_REQUIRED')
    nodes={node['id']:node for node in profile['nodes']}
    if len(nodes)!=len(profile['nodes']):raise ValueError('Duplicate DAG node identity')
    successors={key:[] for key in nodes};incoming={key:0 for key in nodes}
    for edge in profile['edges']:
        before,after=edge['before'],edge['after']
        if before not in nodes or after not in nodes:raise ValueError('Foreign DAG dependency')
        successors[before].append(after);incoming[after]+=1
    queue=deque(key for key,count in incoming.items() if count==0);topological=[]
    while queue:
        key=queue.popleft();topological.append(key)
        for after in successors[key]:
            incoming[after]-=1
            if incoming[after]==0:queue.append(after)
    if len(topological)!=len(nodes):raise UnknownEvidence('CYCLIC_CUDA_DAG')
    terminal=[node['id'] for node in nodes.values() if node['kind']=='handler_end']
    if len(terminal)!=1 or successors[terminal[0]]:raise ValueError('One sink handler completion barrier required')
    if [key for key in nodes if not successors[key]]!=terminal:
        raise ValueError('Every CUDA branch must reach the handler completion barrier')
    threads=defaultdict(list);depths={terminal[0]:0}
    for node in nodes.values():
        if node['kind']=='handler_end':continue
        if node['kind'] not in ACTIVITY|STRUCTURAL:raise ValueError('Unsupported host-supply node kind')
        start,end,thread=(node.get(key) for key in ('issue_start_ns','issue_end_ns','issue_thread'))
        if any(type(value) is not int for value in (start,end,thread)) or end<=start:
            raise UnknownEvidence('ACTUAL_HOST_ISSUE_ORDER_OR_PREDICTED_TEMPLATE_ORDER_REQUIRED')
        threads[thread].append(node)
    for timeline in threads.values():
        timeline.sort(key=lambda node:(node['issue_start_ns'],node['issue_end_ns'],node['id']))
        pending=[];completed_depth=0
        for node in timeline:
            while pending and pending[0][0]<=node['issue_start_ns']:
                _,depth=heapq.heappop(pending);completed_depth=max(completed_depth,depth)
            depth=completed_depth+1;depths[node['id']]=depth
            heapq.heappush(pending,(node['issue_end_ns'],depth))
    maximum=max(depths.values(),default=0);count=sum(map(len,threads.values()))
    if maximum==0 or count==0:raise UnknownEvidence('NONEMPTY_HOST_ISSUE_STRUCTURE_REQUIRED')
    canonical=dict(nodes=sorted((key,node['kind'],depths[key]) for key,node in nodes.items()),
                   edges=sorted((e['before'],e['after']) for e in profile['edges']))
    return dict(nodes=nodes,successors=successors,topological=topological,terminal=terminal[0],
                issue_depth=depths,max_issue_depth=maximum,issue_count=count,
                structure_sha256=identity(canonical))


def training_critical_path(profile,features):
    """DAG lower-bound service with training activity targets, no release clock."""
    tails={}
    for key in reversed(features['topological']):
        node=features['nodes'][key];kind=node['kind']
        duration=node.get('duration_ns')
        if kind in STRUCTURAL:duration=0
        elif type(duration) not in (int,float) or not math.isfinite(duration) or duration<=0:
            raise UnknownEvidence('POSITIVE_TRAINING_ACTIVITY_DURATION_REQUIRED')
        tails[key]=duration+max((tails[after] for after in features['successors'][key]),default=0)
    return max(tails.values()),tails


def natural_boundaries(row):
    worker=row['worker_response'];start,end=worker.get('start_perf_ns'),worker.get('end_perf_ns')
    final=[stage for stage in worker.get('observed_stages',[]) if stage.get('id')=='final_sync']
    if (type(start) is not int or type(end) is not int or end<=start or len(final)!=1
            or type(final[0].get('end_ns')) is not int or not start<=final[0]['end_ns']<=end):
        raise UnknownEvidence('ACTUAL_NATURAL_FINAL_SYNC_AND_HANDLER_BOUNDARIES_REQUIRED')
    return dict(handler_ns=end-start,terminal_after_final_sync_ns=end-final[0]['end_ns'],
                final_sync_complete_offset_ns=final[0]['end_ns']-start)


def fit_observation(profile,row):
    features=structure(profile);critical,tails=training_critical_path(profile,features)
    bounds=natural_boundaries(row);budget=bounds['final_sync_complete_offset_ns']
    if budget<critical:raise UnknownEvidence('NATURAL_DEVICE_COMPLETION_PRECEDES_PROFILED_DAG_CRITICAL_PATH')
    slack=budget-critical
    # For every node, r_v + tail_v <= observed final-sync completion. Choosing
    # r_v=theta*depth_v/max_depth gives the largest feasible theta below.
    candidates=[(budget-tails[key])*features['max_issue_depth']/depth
                for key,depth in features['issue_depth'].items() if depth>0]
    depth_scale=min(candidates)
    if depth_scale<0:raise UnknownEvidence('NEGATIVE_HOST_SUPPLY_FEASIBILITY_SLACK')
    return dict(**bounds,profile_id=profile.get('request_id',profile.get('id')),alias=profile['alias'],
        family=profile['family'],task=profile['task'],pool_id=profile['pool_id'],
        profiled_dag_critical_path_ns=critical,activity_duration_sum_ns=sum(
            node['duration_ns'] for node in profile['nodes'] if node['kind'] in ACTIVITY),
        multistream_duration_sum_used_as_critical_path=False,issue_count=features['issue_count'],
        max_issue_depth=features['max_issue_depth'],structure_sha256=features['structure_sha256'],
        front_loaded_supply_ns=slack,host_issue_depth_scale_ns=depth_scale,
        front_loaded_ns_per_issue=slack/features['issue_count'],
        issue_depth_ns_per_step=depth_scale/features['max_issue_depth'],
        terminal_tail_ns=bounds['terminal_after_final_sync_ns'],
        identification='Conditional feasibility set, not a unique decomposition of CPU supply and GPU/resource/measurement effects',
        release_bounds_formula='0 <= release_v <= natural_final_sync_offset - profiled_DAG_tail_v',
        fitted_scenario_reproduces_training_endpoint_by_construction=True,
        training_endpoint_identity_is_prediction_accuracy=False)


def aggregate(observations):
    grouped=defaultdict(list)
    for row in observations:grouped[row['family'],row['task']].append(row)
    groups=[]
    for (family,task),rows in sorted(grouped.items()):
        captures={row['source'] for row in rows};pools={row['pool_id'] for row in rows}
        supported=len(captures)>=2 and len(pools)>=2
        fields=('front_loaded_ns_per_issue','issue_depth_ns_per_step','terminal_tail_ns')
        groups.append(dict(family=family,task=task,supported=supported,
            independent_capture_ids=sorted(captures),input_ids=sorted(pools),observation_count=len(rows),
            joint_mean={field:statistics.mean(row[field] for row in rows) for field in fields},
            joint_training_scenarios=[{field:row[field] for field in (*fields,'source','pool_id')} for row in rows],
            observed_parameter_ranges={field:dict(min=min(row[field] for row in rows),max=max(row[field] for row in rows)) for field in fields},
            ranges_are_confidence_or_prediction_bounds=False,
            reason=None if supported else 'TWO_NATURAL_CAPTURES_AND_TWO_INPUTS_REQUIRED'))
    return groups


def fit(dag_path,natural_reviews,composite_path,output):
    dag_ref=receipt(dag_path);dag=json.loads(Path(dag_path).read_text())
    contract_ref=receipt(composite_path);contract=json.loads(Path(composite_path).read_text())
    if (dag.get('schema')!='gsdeploy.request_cuda_dag_profiles.v2' or dag.get('training_only') is not True
            or contract.get('schema')!=COMPOSITE_SCHEMA or identity({k:v for k,v in contract.items()
                if k!='composite_execution_sha256'})!=contract.get('composite_execution_sha256')):
        raise ValueError('Training DAG and unchanged source-bound composite contract required')
    if dag['execution_context_sha256']!=contract['components']['service']['original_context_sha256']:
        raise ValueError('DAG template light domain differs from composite service source')
    context=contract['components']['lifecycle']['original_context_sha256']
    plans={(p['alias'],p['task'],p['pool_id']):p for p in dag['profiles']}
    if len(plans)!=len(dag['profiles']):raise ValueError('Unique training profile per exact input/owner required')
    observations=[];rejected=[];captures=[];seen=set();index=[]
    for path in natural_reviews:
        document=json.loads(Path(path).read_text());loader=load_v3
        if document.get('schema')=='gsdeploy.startup_training_collection_strict_review.v4':
            from .small_training_domain_bridge_review_v4 import load_report as loader
        report,plan,raw=loader(path,'natural');source='raw-sha256:'+report['raw']['sha256']
        if source in seen:raise ValueError('Repeated raw is not an independent natural capture')
        seen.add(source)
        if (raw.get('status')!='completed' or raw.get('execution_context_sha256')!=context
                or raw.get('kind')!='measured' or report.get('training_only') is not True):
            raise ValueError('Completed measured same-context natural training capture required')
        captures.append(dict(review=receipt(path),raw=report['raw'],context=report['context'],run_id=source))
        for row in raw['request_records']:
            if row.get('split')!='training' or row.get('status')!='completed':raise ValueError('Every row must be completed training')
            index.append(dict(run_id=source,request_id=row['request_id'],pool_id=row['pool_id'],task=row['task']))
            profile=plans.get((row['model'],row['task'],row['pool_id']))
            if profile is None:
                rejected.append(dict(source=source,request_id=row['request_id'],reason='NO_MATCHING_TRAINING_DAG_TEMPLATE'));continue
            try:observation=fit_observation(profile,row)
            except UnknownEvidence as exc:
                rejected.append(dict(source=source,request_id=row['request_id'],reason=str(exc)));continue
            observations.append(dict(observation,source=source,request_id=row['request_id']))
    if not captures:raise ValueError('Actual natural training captures required')
    groups=aggregate(observations);known=sum(group['supported'] for group in groups)
    result=dict(schema=SCHEMA,kind='partially_identified_dag_host_supply_hypothesis',
        status=('FITTED_CONDITIONAL_HYPOTHESES' if not rejected else 'FITTED_PARTIAL') if known else 'UNKNOWN',
        execution_context_sha256=context,light_execution_context_sha256=dag['execution_context_sha256'],
        dag=dag_ref,composite=contract_ref,captures=captures,groups=groups,observations=observations,rejected=rejected,
        supported_group_count=known,independent_natural_capture_count=len(captures),
        assumptions=['Profiled isolated activity durations transfer unchanged solely for conditional training-feasibility calculation',
            'Unlimited intra-owner ready activity progress gives a DAG critical-path lower bound, not measured GPU resource capacity',
            'Front-loaded and issue-depth releases are two different members of an otherwise non-identifiable feasibility set',
            'Pooled per-issue supply and post-final-sync tail exchangeable within family/task and identical execution context'],
        unidentifiable_effects=['Actual per-launch CPU execution and host stalls','Profiler/natural activity timing differences',
                              'Intra-owner GPU resource serialization beyond explicit dependencies','Cross-owner host/DRAM/driver contention'],
        fitted_scenarios=list(SCENARIOS),profiled_issue_times_used_as_future_release=False,
        measured_handler_end_used_as_future_anchor=False,kernel_durations_rescaled=False,
        observed_ranges_are_confidence_bounds=False,identification_complete=False,
        natural_timing_accuracy_validated=False,physical_gain_certified=False,GPU_started=False,
        sources=[receipt(__file__),receipt(Path(__file__).with_name('kernel_domain_bridge_review_v3.py')),
                 receipt(Path(__file__).with_name('small_training_domain_bridge_review_v4.py'))])
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    save(output/'training-index.json',dict(schema='gsdeploy.independent_prediction_gate_training_index.v2',split='training',records=index))
    result['training_index']=receipt(output/'training-index.json');save(output/'model.json',result)
    return result


class DagHostReleaseProvider:
    def __init__(self,document,context,plans,aliases,*,scenario,
                 allow_identification_hypothesis=False,allow_cross_input_pooling_hypothesis=False):
        if not allow_identification_hypothesis or not allow_cross_input_pooling_hypothesis:
            raise ValueError('Explicit partial-identification and cross-input supply pooling opt-ins required')
        if document.get('schema')!=SCHEMA or document.get('execution_context_sha256')!=context:
            raise ValueError('Source-bound natural host supply execution identity required')
        if scenario not in SCENARIOS:raise ValueError('Explicit supported host-supply scenario required')
        self.document=document;self.context=context;self.plans=plans;self.aliases=aliases;self.scenario=scenario
        self.groups={(g['family'],g['task']):g for g in document['groups'] if g['supported']};self.cache={}

    def __call__(self,node,row):
        key=row['alias'],row['task'],row['pool_id'];profile=self.plans.get(key)
        if profile is None:raise UnknownEvidence('NO_DAG_TEMPLATE_FOR_HOST_SUPPLY')
        group=self.groups.get((self.aliases[row['alias']]['family'],row['task']))
        if group is None:raise UnknownEvidence('NO_SUPPORTED_POOLED_DAG_HOST_SUPPLY')
        if key not in self.cache:self.cache[key]=structure(profile)
        features=self.cache[key];parameters=group['joint_mean']
        # No node duration, observed release offset, or measured handler endpoint
        # is read here. A changed fitted GPU duration changes predicted finishes.
        if node['id']==features['terminal']:
            tail=parameters.get('terminal_tail_ns')
            if type(tail) not in (int,float) or not math.isfinite(tail) or tail<0:
                raise UnknownEvidence('INVALID_TRAINED_POST_FINAL_SYNC_TAIL')
            return dict(not_before_offset_ns=0,after_predecessors_ns=round(tail))
        if node['id'] not in features['issue_depth']:raise UnknownEvidence('UNKNOWN_HOST_ISSUE_STRUCTURE_NODE')
        if self.scenario=='front_loaded_supply':
            release=parameters['front_loaded_ns_per_issue']*features['issue_count']
        else:release=parameters['issue_depth_ns_per_step']*features['issue_depth'][node['id']]
        if not math.isfinite(release) or release<0:raise UnknownEvidence('INVALID_TRAINED_HOST_SUPPLY_PARAMETER')
        return dict(not_before_offset_ns=round(release),after_predecessors_ns=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dag',required=True);parser.add_argument('--natural-reviews',nargs='+',required=True)
    parser.add_argument('--composite',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=fit(args.dag,args.natural_reviews,args.composite,args.output)
    print(json.dumps({key:result[key] for key in ('status','supported_group_count','independent_natural_capture_count')}))


if __name__=='__main__':main()
