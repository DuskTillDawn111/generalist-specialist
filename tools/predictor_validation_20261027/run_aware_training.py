"""Run-aware primary-only OCR duration training and H600 whole-seed CV.

Profiles and H90 development flows may train the same conditional phase model.
Each H/drain/source condition stays separately labelled; no capacity evidence is
pooled. Identical-trace repeats are diagnostics only and never fitted. Prepare
reads prospective inputs, never runtime outputs. Fit/CV are explicit CPU steps.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
from . import flow_fit as original
from .background_profiles import background_semantic_traces,receipt
from .background_duration_variant import fit_traces,background_definition
from .duration_variant_profiles import variant_definition
from .duration_variant_fit import _verify_duration_definition
from .background_statistical import BackgroundStatisticalProvider
from .flow_profiles import map_flow_collection
from .flow_forensics import reconstruct_work
from .profiling_analysis import digest

read=original.read
save=original.save
_checked=original._checked
_unique_refs=original._unique_refs
_tools=original._tools
_training_trace=original._training_trace
TREE_SCHEMA=original.TREE_SCHEMA
PROTOCOL_SCHEMA=original.PROTOCOL_SCHEMA
EVIDENCE_SCHEMA=original.EVIDENCE_SCHEMA
RUN_SCHEMA='gsdeploy.run_aware_primary_training_protocol.v1'
NEW_TOOLS=('run_aware_training.py','duration_variant_profiles.py','background_duration_variant.py','duration_variant_fit.py')
POLICY={'estimator':variant_definition('ocr_handler'),'variant':'ocr_handler',
 'source_mapping_policy':original.POLICY,
 'fitting_cohort':'16 unique primary flows: old H90 eight plus new H600 eight, and old v7/v8 profile collections',
 'repeats':'Two identical-trace repeats and collection preflight never enter fit, solo baseline or coefficient rows; run-variation diagnostics only, not independent input realizations',
 'CV':'Two new H600 whole-seed folds, exclude all four primary routes of that seed; retain all H90 development flows and both old profile collections; CV tests only heldout H600 primary runs',
 'capacity':'H90/H600 and drain remain separate source/run/capacity-condition labels; pooling coefficients does not transfer capacity qualification or certify stability',
 'full_flow_count':16,'CV_training_flow_count':12,'CV_heldout_flow_count':4,
 'GPU_started':False,'independent_validation':False,'hyperparameter_sweep':False}


def _duration_tools():return [receipt(Path(__file__).with_name(name)) for name in NEW_TOOLS]


def _job_metadata(job,role):
    plan=read(_checked(job['plan']))
    if (plan.get('phase')!='predictor_training' or plan.get('split')!='training'
            or plan.get('plan_sha256')!=digest({k:v for k,v in plan.items() if k!='plan_sha256'})):
        raise ValueError('Only complete prospectively frozen training plans allowed')
    candidate=next((c for c in plan['candidates'] if c['id']==job['candidate_id']),None)
    if candidate is None:raise ValueError('Candidate absent from declared run plan')
    window=plan['window_seconds'];drain=plan['drain_seconds']
    if (window,drain)!=(90.,120.) and (window,drain)!=(600.,300.):
        raise ValueError('Only frozen H90/drain120 or H600/drain300 source conditions allowed')
    fid='seed-'+str(plan['seed'])+'__'+job['candidate_id']
    condition={'execution_context_sha256':plan['execution_context_sha256'],'window_seconds':window,
        'drain_seconds':drain,'candidate_id':job['candidate_id'],
        'resource_configuration':candidate['resource_configuration'],
        'workload_rps':plan['workload_rps'],'routing_weights':candidate.get('routing_weights')}
    return {'flow_id':fid,'run_id':job.get('run_id',fid+'__primary'),'campaign_role':job.get('campaign_role',role),
        'source_role':role,'seed':plan['seed'],'candidate_id':job['candidate_id'],
        'window_seconds':window,'drain_seconds':drain,'request_count':len(plan['trace']),
        'execution_context_sha256':plan['execution_context_sha256'],'capacity_condition':condition,
        'capacity_condition_sha256':digest(condition),'plan':job['plan'],
        'raw_path':str((Path(job['output'])/'raw.json').resolve()),
        'fitted':role!='repeat_diagnostic','not_independent_input_realization':role=='repeat_diagnostic'}


def prepare_protocol(old_protocol_path,campaign_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    old=read(old_protocol_path);campaign=read(campaign_path)
    if old.get('policy')!=original.POLICY:raise ValueError('Original fixed training source protocol required')
    _unique_refs(old['source_receipts'])
    if (campaign.get('schema')!='gsdeploy.h600_training_campaign.v1'
            or campaign.get('campaign_sha256')!=digest({k:v for k,v in campaign.items() if k!='campaign_sha256'})):
        raise ValueError('Frozen H600 training campaign required')
    _unique_refs(campaign['source_receipts'])
    primary=read(_checked(campaign['primary_suite']));repeats=read(_checked(campaign['repeat_suite']))
    old_suite=read(_checked(old['suite']))
    if (primary.get('schema')!='gsdeploy.continuous_training_suite.v1' or primary.get('no_validation_data') is not True
            or primary.get('repeats_excluded') is not True or len(primary['jobs'])!=8
            or repeats.get('schema')!='gsdeploy.training_run_repeat_suite.v1' or len(repeats['jobs'])!=2
            or len(old_suite['jobs'])!=8):raise ValueError('Eight old, eight new primary and two diagnostic repeat declarations required')
    metadata=[_job_metadata(j,'H90_development_primary') for j in old_suite['jobs']]
    metadata += [_job_metadata(j,'H600_primary') for j in primary['jobs']]
    repeat_metadata=[_job_metadata(j,'repeat_diagnostic') for j in repeats['jobs']]
    if (len({m['flow_id'] for m in metadata})!=16 or len({m['raw_path'] for m in metadata+repeat_metadata})!=18
            or len({m['run_id'] for m in metadata+repeat_metadata})!=18):raise ValueError('Unique primary flow and all run/raw identities required')
    if any(m['window_seconds']!=90 for m in metadata[:8]) or any(m['window_seconds']!=600 for m in metadata[8:]+repeat_metadata):
        raise ValueError('Source horizon roles do not match frozen H90/H600 conditions')
    seeds=sorted({m['seed'] for m in metadata[8:]})
    if len(seeds)!=2 or any(sum(m['seed']==seed for m in metadata[8:])!=4 for seed in seeds):raise ValueError('Two complete new H600 seeds required')
    if len({m['execution_context_sha256'] for m in metadata+repeat_metadata})!=1:raise ValueError('Shared runtime execution context required')
    for repeat in repeat_metadata:
        match=next((m for m in metadata[8:] if m['flow_id']==repeat['flow_id']),None)
        if match is None:raise ValueError('Repeat must bind a new H600 primary seed/candidate')
        rp=read(repeat['plan']['path']);pp=read(match['plan']['path'])
        if rp['trace']!=pp['trace'] or rp['candidates']!=pp['candidates']:
            raise ValueError('Diagnostic repeats must preserve exact primary inputs and routes')
    output.mkdir(parents=True,exist_ok=False)
    combined={'schema':'gsdeploy.continuous_training_suite.v1','created_ns':time.perf_counter_ns(),
        'source':receipt(__file__),'jobs':old_suite['jobs']+primary['jobs'],'no_validation_data':True,
        'repeats_excluded':True,'source_suites':[old['suite'],campaign['primary_suite']], 'GPU_started':False}
    save(output/'combined-primary-suite.json',combined)
    original.prepare_protocol([r['path'] for r in old['old_training_raws']],output/'combined-primary-suite.json',
        output/'primary16-mapping-protocol.json',expected_flow_count=16)
    refs=_unique_refs([receipt(old_protocol_path),receipt(campaign_path),campaign['primary_suite'],campaign['repeat_suite'],
        *campaign['source_receipts'],receipt(output/'combined-primary-suite.json'),receipt(output/'primary16-mapping-protocol.json'),*_duration_tools()])
    value={'schema':RUN_SCHEMA,'created_ns':time.perf_counter_ns(),'frozen_training_protocol':receipt(output/'primary16-mapping-protocol.json'),
        'old_training_protocol':receipt(old_protocol_path),'campaign':receipt(campaign_path),'policy':POLICY,
        'primary_runs':metadata,'diagnostic_repeat_runs':repeat_metadata,'H600_seeds':seeds,
        'source_receipts':refs,'GPU_started':False,'outcomes_read_during_prepare':False,'validation_sources_used':[]}
    value['protocol_sha256']=digest(value);save(output/'protocol.json',value);return value


def _verify_protocol(path):
    value=read(path)
    if (value.get('schema')!=RUN_SCHEMA or value.get('policy')!=POLICY
            or value.get('protocol_sha256')!=digest({k:v for k,v in value.items() if k!='protocol_sha256'})):
        raise ValueError('Frozen run-aware policy or source protocol changed')
    _unique_refs(value['source_receipts'])
    for ref in _duration_tools():
        if ref not in value['source_receipts']:raise ValueError('Run-aware tool source missing or changed')
    for label in value['primary_runs']+value['diagnostic_repeat_runs']:
        job={'plan':label['plan'],'candidate_id':label['candidate_id'],'output':str(Path(label['raw_path']).parent),
            'run_id':label['run_id'],'campaign_role':label['campaign_role']}
        if _job_metadata(job,label['source_role'])!=label:raise ValueError('Run horizon/source/capacity labels changed')
    return value


def load_training(protocol_path):
    protocol=_verify_protocol(protocol_path);bundle=original.load_training(protocol['frozen_training_protocol']['path'])
    labels={m['flow_id']:m for m in protocol['primary_runs']}
    for group in bundle['groups']:
        if group['role']=='continuous_training':group['run_metadata']=labels[group['group_id']]
    bundle.update(run_protocol=protocol,run_protocol_receipt=receipt(protocol_path));return bundle


def _fit_groups(bundle,groups,heldout_seed=None):
    refs=_unique_refs([bundle['protocol_receipt'],bundle['run_protocol_receipt'],*_tools(),*_duration_tools(),*[g['direct_source'] for g in groups]])
    model=fit_traces([t for g in groups for t in g['traces']],sources=refs,variant='ocr_handler');model['variant']='ocr_handler'
    omitted=sorted(m['flow_id'] for m in bundle['run_protocol']['primary_runs'] if m['source_role']=='H600_primary' and m['seed']==heldout_seed)
    model['training_evidence_tree']={'schema':TREE_SCHEMA,'protocol':bundle['protocol_receipt'],
        'duration_protocol':bundle['run_protocol_receipt'],'omitted_flow_id':None,'omitted_flow_ids':omitted,'omitted_seed':heldout_seed,
        'groups':[{k:copy.deepcopy(v) for k,v in g.items() if k in
            ('group_id','role','raw','plan','seed','candidate_id','direct_source','transitive_sources')} for g in groups],
        'transitive_source_receipts':_unique_refs([r for g in groups for r in g['transitive_sources']]),
        'continuous_training_flow_count':sum(g['role']=='continuous_training' for g in groups),'validation_sources_used':[]}
    model.update(training_policy=POLICY,source_mapping_policy=original.POLICY,duration_fit_protocol=bundle['run_protocol_receipt'],
        run_metadata=[g['run_metadata'] for g in groups if g['role']=='continuous_training'],
        training_observation_counts_by_source=[{'group_id':g['group_id'],'role':g['role'],
            'run_metadata':g.get('run_metadata'),'traces':len(g['traces']),
            'handler_rows':sum(len(t['requests']) for t in g['traces']),
            'semantic_rows':sum(len(r['segments']) for t in g['traces'] for r in t['requests'])} for g in groups],
        diagnostic_repeats_fitted=False,capacity_evidence_pooled=False,
        mapping_reports=[{'group_id':g['group_id'],'report':g['mapping_report']} for g in groups])
    model.pop('model_sha256');model['model_sha256']=digest(model);return model


def _verify_run_coverage(model,protocol,flows,omitted,omitted_ids,seed):
    labels={m['flow_id']:m for m in protocol['primary_runs']}
    if model.get('run_metadata')!=[labels[g['group_id']] for g in model['training_evidence_tree']['groups'] if g['role']=='continuous_training']:
        raise ValueError('Model source/run/capacity metadata omitted or changed')
    if model.get('diagnostic_repeats_fitted') is not False or model.get('capacity_evidence_pooled') is not False:
        raise ValueError('Repeated traces cannot fit or transfer capacity conditions')
    if seed is None:
        if omitted is not None or omitted_ids or len(flows)!=16:raise ValueError('Full model requires all16 primary flows')
    else:
        expected=sorted(m['flow_id'] for m in labels.values() if m['source_role']=='H600_primary' and m['seed']==seed)
        if seed not in protocol['H600_seeds'] or omitted is not None or sorted(omitted_ids)!=expected or len(expected)!=4 or len(flows)!=12:
            raise ValueError('H600 whole-seed CV requires12 primary training flows and four new heldout flows')


def verify_run_aware_model(path):
    """Independently re-walk direct and transitive original-raw evidence roles."""
    document=read(path);context=document['execution_context_sha256']
    _verify_duration_definition(document)
    if document.get('variant')!='ocr_handler':raise ValueError('Only the fixed OCR handler estimator is allowed')
    duration_protocol_ref=document['training_evidence_tree']['duration_protocol']
    duration_protocol=_verify_protocol(duration_protocol_ref['path'])
    _checked(duration_protocol_ref)
    if document.get('training_policy')!=POLICY or document.get('duration_fit_protocol')!=duration_protocol_ref:
        raise ValueError('Duration model policy or protocol binding mismatch')
    BackgroundStatisticalProvider(document,context)
    tree=document.get('training_evidence_tree',{})
    if tree.get('schema')!=TREE_SCHEMA or tree.get('validation_sources_used')!=[]:
        raise ValueError('Explicit validation-free training evidence tree required')
    protocol=read(_checked(tree['protocol']))
    if (protocol.get('schema')!=PROTOCOL_SCHEMA or protocol.get('policy')!=document.get('source_mapping_policy')
            or protocol.get('protocol_sha256')!=digest({k:v for k,v in protocol.items() if k!='protocol_sha256'})):
        raise ValueError('Model training protocol changed')
    if tree['protocol']!=duration_protocol['frozen_training_protocol']:
        raise ValueError('Duration and original evidence protocols disagree')
    _unique_refs(protocol['source_receipts']);_unique_refs(document['source_receipts'])
    _unique_refs(tree['transitive_source_receipts'])
    direct={r['path']:r for r in document['source_receipts']}
    declared={j['flow_id']:j for j in protocol['flow_jobs']};groups=tree['groups'];seen=set();flows=set();raws=[];trace_ids=[];expected_transitive=[]
    expected_direct=_unique_refs([tree['protocol'],duration_protocol_ref,*_tools(),*_duration_tools(),*[g['direct_source'] for g in groups]])
    if direct!={r['path']:r for r in expected_direct} or document['base_model']['source_receipts']!=document['source_receipts']:
        raise ValueError('Model has hidden, missing or inconsistent direct fit sources')
    for group in groups:
        gid=group['group_id']
        if gid in seen:raise ValueError('Duplicate evidence group')
        seen.add(gid);_unique_refs(group['transitive_sources']);_checked(group['raw']);raws.append(group['raw'])
        ref=group['direct_source']
        if direct.get(ref['path'])!=ref:raise ValueError('Evidence group missing from model direct sources')
        if group['role']=='old_profile_training':
            if ref!=group['raw'] or group['raw'] not in protocol['old_training_raws']:
                raise ValueError('Old profile source is not a declared original raw')
            raw=read(group['raw']['path'])
            if not raw.get('request_records') or any(r.get('split')!='training' for r in raw['request_records']):
                raise ValueError('Old original raw contains nontraining observations')
            traces,_=background_semantic_traces(group['raw']['path'])
            expected_refs=_unique_refs([group['raw'],*[r for t in traces for r in t.get('sources',[])]])
        elif group['role']=='continuous_training':
            if gid not in declared or group['raw']['path']!=declared[gid]['raw_path']:
                raise ValueError('Continuous original raw not in prospective suite')
            wrapper=read(_checked(ref));_unique_refs(wrapper['source_receipts'])
            if (wrapper.get('schema')!=EVIDENCE_SCHEMA or wrapper.get('kind')!='derived_training_evidence'
                    or wrapper.get('split')!='training' or wrapper.get('raw_source')!=group['raw']
                    or wrapper.get('plan_source')!=declared[gid]['plan'] or wrapper.get('flow_id')!=gid):
                raise ValueError('Derived training wrapper/source role mismatch')
            mapped=map_flow_collection(group['raw']['path']);traces=mapped['traces']
            if (read(group['raw']['path'])['candidate_id']!=declared[gid]['candidate_id']
                    or wrapper.get('trace_sha256')!=digest(wrapper['trace'])
                    or wrapper['trace_sha256']!=digest(traces[0])
                    or wrapper.get('mapping_report')!=mapped['mapping_report']):
                raise ValueError('Derived training trace does not match strictly remapped original raw')
            expected_refs=_unique_refs(mapped['mapping_report']['sources'])
            if wrapper['source_receipts']!=expected_refs:raise ValueError('Wrapper hides transitive original source receipts')
            flows.add(gid)
        else:raise ValueError('Unknown training evidence group role')
        if group['transitive_sources']!=expected_refs:raise ValueError('Evidence group omits original transitive receipts')
        expected_transitive.extend(expected_refs)
        for trace in traces:
            _training_trace(trace)
            if trace['execution_context_sha256']!=context:raise ValueError('Evidence/model execution context mismatch')
            trace_ids.append(trace['id'])
    if tree['transitive_source_receipts']!=_unique_refs(expected_transitive):raise ValueError('Incomplete global transitive training evidence')
    omitted=tree.get('omitted_flow_id');omitted_ids=tree.get('omitted_flow_ids',[]);omitted_seed=tree.get('omitted_seed');expected=set(declared)
    if len(omitted_ids)!=len(set(omitted_ids)) or not set(omitted_ids)<=expected:raise ValueError('Unknown or duplicate omitted flows')
    if omitted is not None and omitted_ids!=[omitted]:raise ValueError('Single-flow exclusion metadata mismatch')
    if omitted_seed is not None and set(omitted_ids)!={fid for fid,j in declared.items() if j['seed']==omitted_seed}:
        raise ValueError('Whole-seed exclusion metadata mismatch')
    if omitted is None and omitted_seed is None and omitted_ids:raise ValueError('Unspecified exclusion unit')
    expected.difference_update(omitted_ids)
    if flows!=expected or tree.get('continuous_training_flow_count')!=len(flows):
        raise ValueError('Wrong complete/crossfit flow evidence coverage')
    if {g['raw']['path'] for g in groups if g['role']=='old_profile_training'}!={r['path'] for r in protocol['old_training_raws']}:
        raise ValueError('Old training collection omitted or substituted')
    _verify_run_coverage(document,duration_protocol,flows,omitted,omitted_ids,omitted_seed)
    return {'verified':True,'run_aware_model_verified':True,'variant':document['variant'],'estimator':'duration_variants_WLS.v1','model':receipt(path),'execution_context_sha256':context,
        'run_aware_protocol':duration_protocol_ref,'capacity_evidence_pooled':False,'diagnostic_repeats_fitted':False,
        'training_flow_count':len(flows),'training_flow_ids':sorted(flows),'training_trace_ids':trace_ids,
        'source_group_count':len(groups),'omitted_flow_id':omitted,'omitted_flow_ids':omitted_ids,'omitted_seed':omitted_seed,
        'training_raw_receipts':raws,'direct_and_transitive_sources_verified':True,'validation_sources_used':[],
        'GPU_started':False}


def fit_protocol(protocol_path,output,*,full=False):
    """Fit two H600 seed folds by default; full16 requires explicit full=True."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    bundle=load_training(protocol_path);protocol=bundle['run_protocol']
    output.mkdir(parents=True,exist_ok=False);save(output/'started.json',{'protocol':receipt(protocol_path),'full':full,'GPU_started':False})
    original._materialize_evidence(bundle,output/'training-evidence')
    if full:
        model=_fit_groups(bundle,bundle['groups']);path=output/'training-run-aware-model.json';save(path,model)
        verification=verify_run_aware_model(path);save(output/'verification.json',verification)
        manifest={'schema':'gsdeploy.run_aware_full_fit_manifest.v1','model':receipt(path),
            'verification':receipt(output/'verification.json'),'protocol':receipt(protocol_path),
            'primary_training_flows':16,'diagnostic_repeats_fitted':False,'capacity_evidence_pooled':False,
            'independent_validation':False,'GPU_started':False}
    else:
        folder=output/'seed-crossfit';folder.mkdir();folds=[]
        for seed in protocol['H600_seeds']:
            labels=[m for m in protocol['primary_runs'] if m['source_role']=='H600_primary' and m['seed']==seed]
            excluded={m['flow_id'] for m in labels};groups=[g for g in bundle['groups'] if g['group_id'] not in excluded]
            dest=folder/('seed-'+str(seed));dest.mkdir()
            model=_fit_groups(bundle,groups,seed);path=dest/'model.json';save(path,model);verified=verify_run_aware_model(path)
            provider=BackgroundStatisticalProvider(model,bundle['execution_context_sha256']);heldout=[]
            for group in bundle['groups']:
                if group['group_id'] not in excluded:continue
                vectors=reconstruct_work(group['traces'],provider)
                vectors.update(crossfit_model=receipt(path),heldout_source=group['raw'],whole_seed_excluded=True,
                    retrospective_training_cross_validation=True,independent_validation=False)
                work=dest/(group['group_id']+'-heldout-work.json');save(work,vectors)
                heldout.append({'flow_id':group['group_id'],'run_id':group['run_metadata']['run_id'],
                    'run_metadata':group['run_metadata'],'heldout_raw':group['raw'],'heldout_plan':group['plan'],
                    'candidate_id':group['candidate_id'],'heldout_trace_id':group['traces'][0]['id'],'work_vectors':receipt(work)})
            diagnostic=dest/'verification.json';save(diagnostic,verified)
            folds.append({'fold_id':'seed-'+str(seed),'unit':'seed','heldout_seed':seed,'model':receipt(path),
                'heldout_flows':heldout,'training_flow_ids':verified['training_flow_ids'],'diagnostics':receipt(diagnostic)})
            print(json.dumps({'event':'run_aware_H600_seed_fit_complete','seed':seed,'training_flows':12,'heldout_runs':len(heldout)}),flush=True)
        manifest={'schema':'gsdeploy.continuous_seed_crossfit_manifest.v1','protocol':bundle['protocol_receipt'],
            'run_aware_protocol':receipt(protocol_path),'estimator':'duration_variants_WLS.v1','variant':'ocr_handler',
            'training_verifier':{'name':'verify_run_aware_model','source':receipt(__file__)},
            'folds':folds,'source_receipts':_unique_refs([*bundle['source_receipts'],receipt(protocol_path),*_duration_tools()]),
            'retrospective_training_cross_validation':True,'no_independent_validation':True,
            'CV_test_source_role':'H600_primary','diagnostic_repeats_fitted':False,'capacity_evidence_pooled':False,
            'source_scope':'H600 CV only; H90 development flows remain training and are not H600 confirmation',
            'GPU_started':False}
        save(output/'seed-crossfit-manifest.json',manifest)
    _verify_protocol(protocol_path);save(output/'manifest.json',manifest);return manifest


def repeat_diagnostics(protocol_path,model_path,output):
    """Read repeats only in a separate explicit diagnostics operation."""
    if Path(output).exists():raise FileExistsError(output)
    protocol=_verify_protocol(protocol_path);verified=verify_run_aware_model(model_path)
    if verified['omitted_seed'] is not None:raise ValueError('Repeat diagnostic requires full16 fitted model')
    model=read(model_path);provider=BackgroundStatisticalProvider(model,model['execution_context_sha256']);runs=[]
    for label in protocol['diagnostic_repeat_runs']:
        mapped=map_flow_collection(label['raw_path'])
        if mapped['traces'][0]['execution_context_sha256']!=model['execution_context_sha256']:
            raise ValueError('Repeat/model execution context mismatch')
        runs.append({'run_metadata':label,'raw':receipt(label['raw_path']),
            'work':reconstruct_work(mapped['traces'],provider),'fitted':False,'independent_input_realization':False})
    value={'schema':'gsdeploy.run_aware_repeat_diagnostics.v1','model':receipt(model_path),'protocol':receipt(protocol_path),
        'runs':runs,'fitting_observations_added':0,'independent_validation':False,'GPU_started':False}
    save(output,value);return value


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    a=sub.add_parser('prepare')
    for field in ('old-protocol','campaign','output'):a.add_argument('--'+field,required=True)
    a=sub.add_parser('fit');a.add_argument('--protocol',required=True);a.add_argument('--output',required=True);a.add_argument('--full',action='store_true')
    a=sub.add_parser('verify');a.add_argument('--model',required=True)
    a=sub.add_parser('repeat-diagnostics')
    for field in ('protocol','model','output'):a.add_argument('--'+field,required=True)
    args=p.parse_args()
    if args.op=='prepare':value=prepare_protocol(args.old_protocol,args.campaign,args.output)
    elif args.op=='fit':value=fit_protocol(args.protocol,args.output,full=args.full)
    elif args.op=='verify':value=verify_run_aware_model(args.model)
    else:value=repeat_diagnostics(args.protocol,args.model,args.output)
    print(json.dumps({'status':'VERIFIED' if args.op=='verify' else 'COMPLETE','schema':value.get('schema'),'GPU_started':False}))


if __name__=='__main__':main()


