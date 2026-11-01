"""Exact-public-input host supply hypotheses fitted to isolated training bounds.

Two inputs do not identify a general shape law. Each owner/content/generation/
DAG cell needs two actual raw captures. This conditional endpoint decomposition
is neither a kernel interference model nor physical timing qualification.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import statistics

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.resources import identity
from .dag_host_release_v5 import fit_observation, structure, SCENARIOS
from .generation_work_v4 import actual_work, input_content_identity
from .native_collect import receipt, save
from .native_execution_projection_v5 import native_execution_projection

SCHEMA='gsdeploy.exact_input_conditioned_dag_host_supply_hypothesis.v7'
FIELDS=('front_loaded_ns_per_issue','issue_depth_ns_per_step','terminal_tail_ns')
BINDING_FIELDS=('alias','family','task','pool_id','input_content_sha256','input_tokens',
    'protocol_sha256','effective_generation','native_execution_projection_sha256',
    'original_light_context_sha256','original_natural_context_sha256','template','source_sqlite')


def checked(ref):
    if set(ref)!={'path','sha256'} or receipt(ref['path'])!=ref:
        raise ValueError('Changed exact-input host training source')
    return json.loads(Path(ref['path']).read_text())


def binding_identity(binding):
    """Canonical public conditioning; component dispatch identity is separate."""
    result={name:copy.deepcopy(binding[name]) for name in BINDING_FIELDS}
    if result['task']=='ocr':
        if result['input_tokens'] is not None or result['effective_generation'] is not None:
            raise ValueError('OCR remains explicitly non-autoregressive')
        result['work']={'kind':'NON_AUTOREGRESSIVE_OCR_SEPARATE'}
    else:
        work=binding['work']
        # Empirical scenario probability is not an extra conditioning feature.
        result['work']={k:copy.deepcopy(v) for k,v in work.items() if k not in ('weight','observation_count','independent_capture_count')}
    return result


def public_binding(profile,row,dag_ref,context,light_context,natural_context):
    catalogs=[s['frozen'] for s in context['execution_sources'] if s['original_path'].endswith('/runtime-inputs.jsonl')]
    if len(catalogs)!=1:raise ValueError('One immutable public runtime catalog required')
    catalog=catalogs[0]
    if receipt(catalog['path'])!=catalog:raise ValueError('Changed exact content catalog')
    pools={r['pool_id']:r for r in map(json.loads,Path(catalog['path']).read_text().splitlines())}
    pool=pools[row['pool_id']]
    if pool['task_type']!=row['task']:raise ValueError('Public input task differs from training row')
    worker=row['worker_response'];ocr=row['task']=='ocr'
    work={'kind':'NON_AUTOREGRESSIVE_OCR_SEPARATE'} if ocr else actual_work(worker,row['request_id'])
    if ocr and worker.get('finish_reason')!='ctc_complete':
        raise UnknownEvidence('EXPLICIT_TRAINING_OCR_COMPLETION_REQUIRED')
    binding=dict(alias=profile['alias'],family=profile['family'],task=profile['task'],pool_id=profile['pool_id'],
        input_content_sha256=input_content_identity(pool),input_tokens=None if ocr else work['actual_input_tokens'],
        protocol_sha256=worker.get('protocol_sha256'),effective_generation=None if ocr else work['effective_generation'],
        native_execution_projection_sha256=native_execution_projection(context)['projection_sha256'],
        original_light_context_sha256=light_context,original_natural_context_sha256=natural_context,
        template=dict(dag=dag_ref,alias=profile['alias'],task=profile['task'],pool_id=profile['pool_id']),
        source_sqlite=profile['source_sqlite'],work=work)
    return binding_identity(binding),catalog


def aggregate(observations):
    groups=defaultdict(list)
    for row in observations:groups[(row['alias'],row['task'],row['pool_id'],row['binding_sha256'],row['structure_sha256'])].append(row)
    result=[]
    for (alias,task,pool,binding,shape),rows in sorted(groups.items()):
        captures=sorted({r['source'] for r in rows});supported=len(captures)>=2
        # Equal capture weights; repeated rows in one raw cannot add independent support.
        means={field:statistics.mean(statistics.mean(r[field] for r in rows if r['source']==capture)
                                   for capture in captures) for field in FIELDS}
        result.append(dict(alias=alias,task=task,pool_id=pool,binding_sha256=binding,
            structure_sha256=shape,supported=supported,independent_capture_ids=captures,
            observation_count=len(rows),joint_mean=means,
            joint_training_scenarios=[{k:r[k] for k in (*FIELDS,'source','request_id')} for r in rows],
            observed_parameter_ranges={f:dict(min=min(r[f] for r in rows),max=max(r[f] for r in rows)) for f in FIELDS},
            ranges_are_confidence_or_prediction_bounds=False,
            reason=None if supported else 'TWO_INDEPENDENT_EXACT_INPUT_NATURAL_CAPTURES_REQUIRED'))
    return result


def derive(source_host_path):
    source_ref=receipt(source_host_path);source=checked(source_ref)
    if source.get('schema')!='gsdeploy.dag_host_supply_training_hypothesis.v5':
        raise ValueError('Original isolated v5 training evidence required')
    for ref in source['sources']:
        if receipt(ref['path'])!=ref:raise ValueError('Changed original isolated host producer source')
    dag=checked(source['dag']);contract=checked(source['composite'])
    if identity({k:v for k,v in contract.items() if k!='composite_execution_sha256'})!=contract.get('composite_execution_sha256'):
        raise ValueError('Changed original composite execution identity')
    if (dag['execution_context_sha256']!=source['light_execution_context_sha256']
            or contract['components']['lifecycle']['original_context_sha256']!=source['execution_context_sha256']):
        raise ValueError('Changed original isolated light/natural component context')
    if dag.get('training_only') is not True:raise ValueError('Training-only DAG required')
    if contract['target']['execution_context_sha256']!=source['execution_context_sha256']:
        raise ValueError('Original host/target context mismatch')
    plans={(p['alias'],p['task'],p['pool_id']):p for p in dag['profiles']}
    if len(plans)!=len(dag['profiles']):raise ValueError('Distinct owner/task/input DAG templates required')
    for p in plans.values():
        if receipt(p['source_sqlite']['path'])!=p['source_sqlite']:raise ValueError('Changed bound DAG SQLite')
    observations=[];bindings={};captures=[];seen=set();catalogs={};index=[];rejected=[]
    for capture in source['captures']:
        doc=checked(capture['review'])
        from .kernel_domain_bridge_review_v3 import load_report
        if doc.get('schema')=='gsdeploy.startup_training_collection_strict_review.v4':
            from .small_training_domain_bridge_review_v4 import load_report
        report,plan,raw=load_report(capture['review']['path'],'natural')
        if report['raw']!=capture['raw'] or report['context']!=capture['context']:
            raise ValueError('Original source capture receipts differ')
        context=checked(capture['context']);source_id='raw-sha256:'+capture['raw']['sha256']
        if source_id in seen:raise ValueError('Repeated raw is not an independent capture')
        seen.add(source_id)
        if (raw.get('status')!='completed' or raw.get('kind')!='measured' or report.get('training_only') is not True
                or raw.get('execution_context_sha256')!=source['execution_context_sha256']):
            raise ValueError('Completed measured original-context training capture required')
        # The source v5 captures are isolated fits, never SSG endpoint labels.
        jobs=plan.get('profile_jobs',plan.get('jobs',[]))
        if not jobs or any(len(job.get('inputs',{}))!=1 for job in jobs):
            raise ValueError('Only isolated capture endpoints may fit exact-input host supply')
        captures.append(capture)
        for row in raw['request_records']:
            if row.get('split')!='training' or row.get('status')!='completed':raise ValueError('Completed training rows required')
            key=row['model'],row['task'],row['pool_id'];profile=plans.get(key)
            index.append(dict(run_id=source_id,request_id=row['request_id'],task=row['task'],pool_id=row['pool_id']))
            if profile is None:rejected.append(dict(source=source_id,request_id=row['request_id'],reason='NO_EXACT_TRAINING_DAG'));continue
            binding,catalog=public_binding(profile,row,source['dag'],context,source['light_execution_context_sha256'],source['execution_context_sha256'])
            catalogs[catalog['path']]=catalog
            if key in bindings and bindings[key]!=binding:raise ValueError('Training content/generation/DAG identity differs within exact input')
            bindings[key]=binding
            try:fit=fit_observation(profile,row)
            except UnknownEvidence as exc:
                rejected.append(dict(source=source_id,request_id=row['request_id'],reason=str(exc)));continue
            observations.append(dict(fit,source=source_id,request_id=row['request_id'],binding_sha256=identity(binding)))
    expected_index=dict(schema='gsdeploy.independent_prediction_gate_training_index.v2',split='training',records=index)
    if checked(source['training_index'])!=expected_index:
        raise ValueError('Original isolated host training index differs from actual raw cohort')
    groups=aggregate(observations)
    model=dict(schema=SCHEMA,kind='exact_known_input_partial_identification_hypothesis',
        execution_context_sha256=source['execution_context_sha256'],light_execution_context_sha256=source['light_execution_context_sha256'],
        source_host=source_ref,dag=source['dag'],composite=source['composite'],captures=captures,
        public_catalogs=list(catalogs.values()),public_bindings=list(bindings.values()),groups=groups,
        observations=observations,rejected=rejected,supported_group_count=sum(g['supported'] for g in groups),
        exact_input_minimum_independent_captures=2,parameter_aggregation='equal raw-capture weights within exact owner/input/work/DAG cell',
        shape_features='Exact public content, tokenizer input_tokens, protocol, generation work, DAG source/order structure; no learned continuous shape extrapolation',
        unique_input_count_by_family={family:len({b['input_content_sha256'] for b in bindings.values() if b['family']==family}) for family in {b['family'] for b in bindings.values()}},
        fitted_scenarios=list(SCENARIOS),identification_complete=False,host_supply_physically_identified=False,
        strict_known=False,numeric_hypothesis_only=True,kernel_fit_gate_changed=False,
        physical_kernel_interference_model=False,training_endpoint_identity_is_prediction_accuracy=False,
        unseen_input_supported=False,cross_input_pooling_used=False,family_mean_fallback_used=False,
        observed_ranges_are_confidence_bounds=False,physical_accuracy_validated=False,physical_gain_certified=False,
        measured_handler_end_used_as_future_anchor=False,profiled_issue_times_used_as_future_release=False,
        kernel_durations_rescaled=False,SSG_endpoints_used_for_fit=False,validation_refit=False,GPU_started=False)
    return model,index


def fit(source_host_path,output):
    model,index=derive(source_host_path)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'training-index.json',dict(schema='gsdeploy.independent_prediction_gate_training_index.v2',split='training',records=index))
    model['training_index']=receipt(out/'training-index.json');model['sources']=[receipt(__file__)]
    save(out/'model.json',model);return model


def verify_model(document):
    """Rebuild every fitted parameter/support cell from the bound original raws.

    Refreshing an outer model receipt cannot legitimize changed fit cells,
    invented captures or stale nested training/index references.
    """
    if document.get('schema')!=SCHEMA:
        raise ValueError('Exact-input host v7 model required')
    checked(document['source_host'])
    rebuilt,index=derive(document['source_host']['path'])
    core={k:v for k,v in document.items() if k not in ('training_index','sources')}
    if core!=rebuilt:
        raise ValueError('Exact-input host model differs from rederived training evidence')
    expected_index=dict(schema='gsdeploy.independent_prediction_gate_training_index.v2',split='training',records=index)
    if checked(document['training_index'])!=expected_index:
        raise ValueError('Exact-input host training index differs from actual raw cohort')
    if document.get('sources')!=[receipt(__file__)]:
        raise ValueError('Exact-input host producer source changed')
    return document


class DagHostReleaseProvider:
    def __init__(self,document,context,plans,aliases,*,scenario,allow_identification_hypothesis=False,
                 allow_exact_input_conditioning_hypothesis=False,public_bindings=None):
        if allow_identification_hypothesis is not True or allow_exact_input_conditioning_hypothesis is not True:
            raise ValueError('Explicit partial-identification and exact-input conditioning opt-ins required')
        if document.get('schema')!=SCHEMA or document.get('execution_context_sha256')!=context:
            raise ValueError('Exact-input host original natural context required')
        if scenario not in SCENARIOS:raise ValueError('Explicit supported host-supply scenario required')
        verify_model(document)
        if public_bindings is None:raise ValueError('Independently source-verified public template bindings required')
        self.plans,self.aliases,self.scenario=plans,aliases,scenario;self.cache={}
        self.bindings={(b['alias'],b['task'],b['pool_id']):binding_identity(b) for b in document['public_bindings']}
        self.groups={(g['alias'],g['task'],g['pool_id']):g for g in document['groups'] if g['supported']}
        if len(self.bindings)!=len(document['public_bindings']) or len(self.groups)!=sum(g['supported'] for g in document['groups']):
            raise ValueError('Distinct exact-input host binding and group identities required')
        for key,binding in self.bindings.items():
            if key not in public_bindings or binding_identity(public_bindings[key])!=binding:
                raise ValueError('Exact public content/generation/DAG binding mismatch')
        self.dag_ref=document['dag']

    def __call__(self,node,row):
        key=row['alias'],row['task'],row['pool_id'];profile=self.plans.get(key);group=self.groups.get(key)
        if profile is None or group is None:raise UnknownEvidence('NO_SUPPORTED_EXACT_INPUT_DAG_HOST_SUPPLY')
        if key not in self.cache:
            binding=self.bindings[key]
            if (self.aliases[row['alias']]['family']!=binding['family'] or profile['source_sqlite']!=binding['source_sqlite']
                    or binding['template']!=dict(dag=self.dag_ref,alias=row['alias'],task=row['task'],pool_id=row['pool_id'])
                    or identity(binding)!=group['binding_sha256']):
                raise ValueError('Changed exact-input host execution/template binding')
            features=structure(profile)
            if features['structure_sha256']!=group['structure_sha256']:raise ValueError('Changed exact-input DAG host-order structure')
            if len(set(group['independent_capture_ids']))<2:raise UnknownEvidence('TWO_EXACT_INPUT_CAPTURE_SUPPORT_REQUIRED')
            self.cache[key]=features
        features=self.cache[key];parameters=group['joint_mean']
        if any(type(parameters.get(field)) not in (int,float) or not math.isfinite(parameters[field]) or parameters[field]<0 for field in FIELDS):
            raise UnknownEvidence('INVALID_EXACT_INPUT_HOST_PARAMETER')
        if node['id']==features['terminal']:
            return dict(not_before_offset_ns=0,after_predecessors_ns=round(parameters['terminal_tail_ns']))
        if node['id'] not in features['issue_depth']:raise UnknownEvidence('UNKNOWN_EXACT_INPUT_HOST_NODE')
        release=(parameters['front_loaded_ns_per_issue']*features['issue_count'] if self.scenario=='front_loaded_supply'
                 else parameters['issue_depth_ns_per_step']*features['issue_depth'][node['id']])
        if release<0:raise UnknownEvidence('INVALID_EXACT_INPUT_HOST_RELEASE')
        return dict(not_before_offset_ns=round(release),after_predecessors_ns=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-host',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();m=fit(a.source_host,a.output)
    print(json.dumps(dict(supported_groups=m['supported_group_count'],exact_input_only=True,physical_accuracy_validated=False)))

if __name__=='__main__':main()
