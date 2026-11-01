"""Preparation-aware, exact-input host supply fitted only to isolated sources.

The initial launch prefix is anchored separately from subsequent issue depth.
Natural language-forward boundaries and independent light API offsets identify
a conditional cross-domain alignment, not the natural per-API execution clock.
OCR has no language anchor and explicitly retains the light transfer hypothesis.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sqlite3
import statistics

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.resources import identity
from .dag_host_release_v5 import ACTIVITY, natural_boundaries, structure, training_critical_path
from .dag_host_release_v7 import binding_identity, checked, verify_model as verify_v7
from .dag_phase_diagnostic_v8 import load_capture
from .host_preparation_diagnostic_v8 import derive as forensic_derive
from .native_collect import receipt, save
from .trace_export import linux_pid

SCHEMA='gsdeploy.preparation_conditioned_exact_input_host_supply.v8'
SCENARIO='preparation_conditioned_issue_depth_supply'


def api_captures(parameters):
    result={}
    for ref in parameters['solo_light_reviews']:
        strict,_,raw=load_capture(ref,'light-kernel')
        for alias,owner in strict['owners'].items():
            parsed=checked(owner['parsed'])
            with sqlite3.connect(Path(owner['sqlite']['path']).as_uri()+'?mode=ro',uri=True) as conn:
                conn.row_factory=sqlite3.Row
                runtime=[dict(r) for r in conn.execute('select r.*,s.value as api_name from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id')]
            for row in raw['request_records']:
                if row['model']!=alias:continue
                phases=[p for p in parsed['phases'] if p['request_id']==row['request_id']]
                pid=phases[0]['pid'];mapping=parsed['clock_mappings'][str(pid)]
                if mapping.get('valid') is not True:raise ValueError('Valid isolated API clock mapping required')
                start=row['worker_response']['start_perf_ns']-mapping['offset_ns']
                inside=lambda stamp:any(p['nsys_start_ns']<=stamp<=p['nsys_end_ns'] for p in phases)
                apis=sorted((a for a in runtime if linux_pid(a['globalTid'])==pid and inside(a['start'])),
                            key=lambda a:(a['start'],a['end'],a['correlationId']))
                roles={}
                for a in apis:
                    if a['globalTid'] not in roles:roles[a['globalTid']]=len(roles)
                    a['thread_role']=roles[a['globalTid']]
                semantic=row['worker_response'].get('semantic_forward_observations',{}).get('records',[])
                first_semantic=None
                if semantic:
                    candidates=[p for p in phases if p['segment_id']==semantic[0]['id']]
                    if len(candidates)!=1:raise ValueError('One source NVTX first semantic range required')
                    first_semantic=candidates[0]['nsys_start_ns']
                key=alias,row['task'],row['pool_id']
                result.setdefault(key,[]).append(dict(raw=strict['raw'],sqlite=owner['sqlite'],
                    apis=apis,handler_trace_start_ns=start,first_semantic_trace_start_ns=first_semantic,
                    clock_mapping=mapping))
    return result


def derive(host_path,forensic_path,parameters_path):
    refs=dict(source_host=receipt(host_path),forensic=receipt(forensic_path),parameters=receipt(parameters_path))
    host=checked(refs['source_host']);verify_v7(host)
    parameters=checked(refs['parameters'])
    if parameters['host_model']!=refs['source_host']:raise ValueError('Original host and isolated parameter sources differ')
    forensic=checked(refs['forensic'])
    if forensic_derive(parameters_path)!=forensic:raise ValueError('Forensic API evidence differs from independent original captures')
    dag=checked(host['dag']);plans={(p['alias'],p['task'],p['pool_id']):p for p in dag['profiles']}
    api_by_key=api_captures(parameters);forensic_rows={(r['alias'],r['task'],r['pool_id']):r for r in forensic['rows']}
    natural={}
    for ref in parameters['solo_natural_reviews']:
        strict,_,raw=load_capture(ref,'natural')
        for row in raw['request_records']:
            natural.setdefault((row['model'],row['task'],row['pool_id']),[]).append((strict['raw'],row))
    groups=[]
    for key,profile in sorted(plans.items()):
        captures=api_by_key[key]
        if len(captures)!=2 or len({c['raw']['sha256'] for c in captures})!=2:
            raise ValueError('Two independent isolated API captures required')
        signatures=[[dict(name=a['api_name'],thread_role=a['thread_role']) for a in c['apis']] for c in captures]
        if signatures[0]!=signatures[1]:raise UnknownEvidence('INDEPENDENT_RUNTIME_API_NAME_AND_THREAD_ORDER_MISMATCH')
        if not forensic_rows[key]['exact_ordered_gpu_signature_match']:
            raise UnknownEvidence('INDEPENDENT_FULL_GPU_ORDER_MISMATCH')
        original=[c for c in captures if c['sqlite']==profile['source_sqlite']]
        if len(original)!=1:raise ValueError('DAG must bind exactly one actual isolated API capture')
        source=original[0];source_index={(a['start'],a['end'],a['api_name']):i for i,a in enumerate(source['apis'])}
        features=structure(profile);critical,tails=training_critical_path(profile,features)
        seq=sorted((n for n in profile['nodes'] if n['kind']!='handler_end'),key=lambda n:(n['issue_start_ns'],n['issue_end_ns'],n['id']))
        first_activity=next(n for n in seq if n['kind'] in ACTIVITY)
        if key[1]=='ocr':
            anchor=next(n for n in seq if n['kind']=='gpu')
            anchor_kind='isolated_light_first_GPU_API_transfer_hypothesis'
        else:
            if any(c['first_semantic_trace_start_ns'] is None for c in captures):raise ValueError('Two actual light first semantic ranges required')
            anchor=next(n for n in seq if n['issue_start_ns']>=source['first_semantic_trace_start_ns'])
            anchor_kind='natural_first_semantic_plus_independent_light_API_prefix_alignment_hypothesis'
        anchor_depth=features['issue_depth'][anchor['id']]
        prefix=[n for n in seq if n['issue_start_ns']<=anchor['issue_start_ns']]
        if any(features['issue_depth'][n['id']]>anchor_depth for n in prefix):raise UnknownEvidence('MULTITHREAD_PREFIX_NOT_IDENTIFIED')
        if any(features['issue_depth'][n['id']]<anchor_depth for n in seq if n not in prefix):raise UnknownEvidence('NONMONOTONE_POST_PREPARATION_DEPTH')
        indexed={}
        for n in prefix:
            signature=n['issue_start_ns'],n['issue_end_ns'],n['api_name']
            if signature not in source_index:raise ValueError('Exact source DAG/API membership missing')
            indexed[n['id']]=source_index[signature]
        # Real pre-request device synchronization remains independent of input
        # preparation. It must not be shifted forward by OCR image preparation.
        preactivity_ids={n['id'] for n in prefix if n['issue_start_ns']<first_activity['issue_start_ns']}
        offsets={};details={}
        for node_id,index in indexed.items():
            values=[]
            for c in captures:
                api=c['apis'][index]
                base=(c['handler_trace_start_ns'] if key[1]=='ocr' or node_id in preactivity_ids
                      else c['first_semantic_trace_start_ns'])
                values.append(api['start']-base)
            offsets[node_id]=statistics.mean(values)
            details[node_id]=dict(source_runtime_ordinal=index,reference_offsets_ns=values,
                reference='handler_start' if key[1]=='ocr' or node_id in preactivity_ids else 'first_semantic_NVTX_start',
                kind=features['nodes'][node_id]['kind'],depth=features['issue_depth'][node_id])
        observations=[]
        for raw_ref,row in natural[key]:
            bounds=natural_boundaries(row);budget=bounds['completion_upper_bound_offset_ns']
            semantic=row['worker_response'].get('semantic_forward_observations',{}).get('records',[])
            natural_semantic=(semantic[0]['start_ns']-row['worker_response']['start_perf_ns']) if semantic else None
            if key[1]!='ocr' and natural_semantic is None:raise ValueError('Natural first semantic boundary required')
            releases={node_id:offset+(0 if key[1]=='ocr' or node_id in preactivity_ids else natural_semantic)
                      for node_id,offset in offsets.items()}
            if any(not math.isfinite(v) or v<0 for v in releases.values()):
                raise UnknownEvidence('NEGATIVE_PREPARATION_PREFIX_ALIGNMENT')
            if any(releases[n['id']]+tails[n['id']]>budget for n in prefix):
                raise UnknownEvidence('PREPARATION_PREFIX_INFEASIBLE_UNDER_NATURAL_COMPLETION_BOUND')
            baseline=releases[anchor['id']]
            candidates=[(budget-baseline-tails[n['id']])/(features['issue_depth'][n['id']]-anchor_depth)
                        for n in seq if features['issue_depth'][n['id']]>anchor_depth]
            if not candidates or min(candidates)<0:raise UnknownEvidence('NO_FEASIBLE_POST_PREPARATION_SUPPLY')
            step=min(candidates)
            observations.append(dict(source_raw=raw_ref,request_id=row['request_id'],
                first_semantic_start_offset_ns=natural_semantic,completion_upper_bound_offset_ns=budget,
                prefix_release_ns=releases,anchor_release_ns=baseline,post_preparation_ns_per_step=step,
                terminal_tail_ns=bounds['terminal_after_completion_bound_ns']))
        if len(observations)!=2 or len({r['source_raw']['sha256'] for r in observations})!=2:
            raise ValueError('Two independent isolated natural captures required')
        groups.append(dict(alias=key[0],task=key[1],pool_id=key[2],supported=True,
            structure_sha256=features['structure_sha256'],anchor_id=anchor['id'],anchor_depth=anchor_depth,
            anchor_kind=anchor_kind,first_activity_id=first_activity['id'],preactivity_ids=sorted(preactivity_ids),
            source_runtime_order_sha256=identity(signatures[0]),light_capture_refs=[c['raw'] for c in captures],
            prefix_offset_evidence=details,observations=observations,
            prefix_release_ns={node_id:statistics.mean(o['prefix_release_ns'][node_id] for o in observations) for node_id in offsets},
            joint_mean={field:statistics.mean(o[field] for o in observations)
                        for field in ('anchor_release_ns','post_preparation_ns_per_step','terminal_tail_ns')},
            profiled_DAG_critical_path_ns=critical,original_work_durations_rescaled=False,
            first_activity_transfer_physically_identified=False,post_preparation_shape_physically_identified=False))
    sources=[receipt(__file__),receipt(Path(__file__).with_name('host_preparation_diagnostic_v8.py')),
             receipt(Path(__file__).with_name('dag_host_release_v5.py')),receipt(Path(__file__).with_name('dag_host_release_v7.py'))]
    return dict(schema=SCHEMA,**refs,dag=host['dag'],public_bindings=copy.deepcopy(host['public_bindings']),
        execution_context_sha256=host['execution_context_sha256'],light_execution_context_sha256=host['light_execution_context_sha256'],
        groups=groups,sources=sources,scenario=SCENARIO,training_only=True,GPU_started=False,
        kind='preparation_conditioned_partial_identification_hypothesis',
        first_preparation_independently_anchored=True,post_preparation_budget_subtracted_before_fit=True,
        natural_first_CUDA_observed=False,profiler_natural_API_transfer_validated=False,
        SSG_endpoints_used_for_fit=False,validation_refit=False,physical_accuracy_validated=False,
        global_request_concurrency_cap_used=False,equal_context_fallback_used=False,
        assumptions=['Natural semantic Python start is a host boundary, not GPU busy or first CUDA activity',
            'Initial API-prefix position relative to first semantic NVTX transfers from two isolated light captures to natural',
            'OCR has no semantic anchor and first-GPU preparation remains an explicit isolated-light transfer hypothesis',
            'Post-preparation issue-depth shape remains a completion-feasible hypothesis, not measured natural API supply',
            'Exact content/work/DAG cells only; no co-run endpoint fitting or unseen-input extrapolation'])


def verify_model(document):
    if document.get('schema')!=SCHEMA:raise ValueError('Preparation-conditioned host model required')
    for field in ('source_host','forensic','parameters'):
        checked(document[field])
    rebuilt=derive(document['source_host']['path'],document['forensic']['path'],document['parameters']['path'])
    if rebuilt!=document:raise ValueError('Preparation host model differs from original isolated evidence')
    return document


class DagPreparationHostReleaseProvider:
    def __init__(self,document,context,plans,aliases,*,scenario,
                 allow_preparation_alignment_hypothesis=False,public_bindings=None):
        if allow_preparation_alignment_hypothesis is not True:raise ValueError('Explicit preparation alignment hypothesis opt-in required')
        if document.get('execution_context_sha256')!=context or scenario!=SCENARIO:
            raise ValueError('Exact original context and preparation-conditioned scenario required')
        verify_model(document)
        self.plans=plans;self.aliases=aliases;self.cache={};self.dag_ref=document['dag']
        self.groups={(g['alias'],g['task'],g['pool_id']):g for g in document['groups'] if g['supported']}
        self.bindings={(b['alias'],b['task'],b['pool_id']):binding_identity(b) for b in document['public_bindings']}
        if public_bindings is None or set(public_bindings)!=set(self.bindings):raise ValueError('Exact source-verified public input bindings required')
        for key,binding in self.bindings.items():
            if binding_identity(public_bindings[key])!=binding:raise ValueError('Changed exact public input binding')

    def __call__(self,node,row):
        key=row['alias'],row['task'],row['pool_id'];group=self.groups.get(key);plan=self.plans.get(key)
        if group is None or plan is None:raise UnknownEvidence('NO_EXACT_PREPARATION_HOST_SUPPLY')
        if key not in self.cache:
            binding=self.bindings[key]
            if (self.aliases[row['alias']]['family']!=binding['family'] or plan['source_sqlite']!=binding['source_sqlite']
                    or binding['template']!=dict(dag=self.dag_ref,alias=key[0],task=key[1],pool_id=key[2])):
                raise ValueError('Changed preparation host execution/template binding')
            features=structure(plan)
            if features['structure_sha256']!=group['structure_sha256']:raise ValueError('Changed preparation host DAG structure')
            self.cache[key]=features
        features=self.cache[key];parameters=group['joint_mean']
        if node['id']==features['terminal']:
            return dict(not_before_offset_ns=0,after_predecessors_ns=round(parameters['terminal_tail_ns']))
        if node['id'] not in features['issue_depth']:raise UnknownEvidence('UNKNOWN_PREPARATION_HOST_NODE')
        if node['id'] in group['prefix_release_ns']:release=group['prefix_release_ns'][node['id']]
        else:
            depth=features['issue_depth'][node['id']]-group['anchor_depth']
            if depth<0:raise UnknownEvidence('UNIDENTIFIED_PREPARATION_PREFIX_NODE')
            release=parameters['anchor_release_ns']+depth*parameters['post_preparation_ns_per_step']
        if not math.isfinite(release) or release<0:raise UnknownEvidence('INVALID_PREPARATION_HOST_RELEASE')
        return dict(not_before_offset_ns=round(release),after_predecessors_ns=0)


def main():
    p=argparse.ArgumentParser()
    for key in ('host','forensic','parameters','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    model=derive(a.host,a.forensic,a.parameters);save(out/'model.json',model)
    print(json.dumps(dict(model=receipt(out/'model.json'),groups=len(model['groups']))))

if __name__=='__main__':main()
