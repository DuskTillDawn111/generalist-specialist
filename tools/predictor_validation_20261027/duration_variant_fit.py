"""Prospectively fixed OCR-handler / phase-ridge1 whole-seed training CV.

No old frozen file is modified. Uses frozen source mapping/materialization,
then an independent fitter and verifier with explicit new numerical provenance.
No full model or leave-one-flow-out model is fitted here, and no GPU is used.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
from . import flow_fit as original
from .background_profiles import background_semantic_traces, checked_receipt, receipt
from .background_statistical import BackgroundStatisticalProvider
from .background_duration_variant import fit_traces, background_definition
from .duration_variant_profiles import variant_definition, VARIANTS
from .flow_forensics import reconstruct_work
from .flow_profiles import map_flow_collection
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
DURATION_PROTOCOL_SCHEMA='gsdeploy.duration_variants_seed_fit_protocol.v1'
NEW_TOOLS=('duration_variant_profiles.py','background_duration_variant.py','duration_variant_fit.py')
POLICY={'numerical_engines':{v:variant_definition(v) for v in VARIANTS},'source_mapping_policy':original.POLICY,
 'fit_unit':'Two whole-seed refits for each of two fixed variants; baseline frozen models reused',
 'independent_validation':False,'GPU_started':False}


def _duration_tools():return [receipt(Path(__file__).with_name(name)) for name in NEW_TOOLS]


def _checked_selection(path,training_protocol):
    value=read(path)
    if (value.get('schema')!='gsdeploy.duration_variants_training_cv_selection_protocol.v1'
            or value.get('protocol_sha256')!=digest({k:v for k,v in value.items() if k!='protocol_sha256'})
            or value.get('original_training_protocol')!=receipt(training_protocol)
            or value.get('no_independent_results_read') is not True
            or [c['id'] for c in value['candidates']]!=['duration_v12',*VARIANTS]):
        raise ValueError('Frozen three-candidate training-only protocol required')
    for name in ('original_training_protocol','duration_fit_protocol','baseline_seed_manifest','prior_selection_protocol','original_threshold_addendum'):
        _checked(value[name])
    return value


def prepare_protocol(training_protocol,selection_path,output):
    if Path(output).exists():raise FileExistsError(output)
    _checked_selection(selection_path,training_protocol)
    old=read(training_protocol)
    if old.get('policy')!=original.POLICY:raise ValueError('Frozen original source policy required')
    _unique_refs(old['source_receipts'])
    value={'schema':DURATION_PROTOCOL_SCHEMA,'created_ns':time.perf_counter_ns(),
        'frozen_training_protocol':receipt(training_protocol),'selection_protocol':receipt(selection_path),
        'policy':POLICY,'policy_sha256':digest(POLICY),
        'source_receipts':_unique_refs([receipt(training_protocol),receipt(selection_path),*_duration_tools()]),
        'fit_seed_count':2,'variant_count':2,'baseline_refits':0,'GPU_started':False,'independent_validation':False,
        'no_independent_results_read':True}
    value['protocol_sha256']=digest(value);save(output,value);return value


def _verify_protocol(path):
    value=read(path)
    if (value.get('schema')!=DURATION_PROTOCOL_SCHEMA or value.get('policy')!=POLICY
            or value.get('protocol_sha256')!=digest({k:v for k,v in value.items() if k!='protocol_sha256'})):
        raise ValueError('Frozen variants fit policy changed')
    _unique_refs(value['source_receipts'])
    expected=_unique_refs([value['frozen_training_protocol'],value['selection_protocol'],*_duration_tools()])
    if value['source_receipts']!=expected:raise ValueError('Missing or changed variant tool sources')
    _checked_selection(value['selection_protocol']['path'],value['frozen_training_protocol']['path'])
    return value


def _verify_duration_definition(document):
    variant=document.get('variant');expected=background_definition(variant);base=variant_definition(variant)
    if (document.get('definition')!=expected or document.get('definition_sha256')!=digest(expected)
            or document['base_model'].get('definition')!=base
            or document['base_model'].get('definition_sha256')!=digest(base)):
        raise ValueError('Variant estimator mislabelled or numerical definition changed')
    if variant=='ocr_handler':
        fitted=document['base_model'];handlers={tuple(m['target']):m for m in fitted['handler_models']}
        for profile in fitted['profiles']:
            if profile['task']=='ocr' and [s['id'] for s in profile['segments']]!=['ocr_native']:
                raise ValueError('OCR handler variant must have one complete OCR stage')
        for model in fitted['semantic_models']:
            if model['target'][1]!='ocr':continue
            handler=handlers[tuple(model['target'][:3])]
            tied={key:handler['progress'][full] for key,full in fitted['phase_to_family'].items() if full in handler['progress']}
            if (model['target'][3]!='ocr_native' or model['progress']!=tied or model['priors']!=tied
                    or model['fit_diagnostics'].get('phase_parameters_fitted')!=0):
                raise ValueError('OCR phase speeds must be exactly tied to handler context')


def _fit_groups(bundle,groups,heldout_seed,variant):
    traces=[t for g in groups for t in g['traces']]
    for trace in traces:_training_trace(trace)
    sources=_unique_refs([bundle['protocol_receipt'],bundle['duration_protocol_receipt'],*_tools(),*_duration_tools(),*[g['direct_source'] for g in groups]])
    document=fit_traces(traces,sources=sources,variant=variant)
    document["variant"]=variant
    omitted=sorted(g['group_id'] for g in bundle['groups'] if g['role']=='continuous_training' and g['seed']==heldout_seed)
    document['training_evidence_tree']={'schema':TREE_SCHEMA,'protocol':bundle['protocol_receipt'],
        'duration_protocol':bundle['duration_protocol_receipt'],'omitted_flow_id':None,
        'omitted_flow_ids':omitted,'omitted_seed':heldout_seed,
        'groups':[{k:copy.deepcopy(v) for k,v in g.items() if k in
            ('group_id','role','raw','plan','seed','candidate_id','direct_source','transitive_sources')} for g in groups],
        'transitive_source_receipts':_unique_refs([r for g in groups for r in g['transitive_sources']]),
        'continuous_training_flow_count':sum(g['role']=='continuous_training' for g in groups),'validation_sources_used':[]}
    document['training_policy']=POLICY
    document['source_mapping_policy']=original.POLICY
    document['duration_fit_protocol']=bundle['duration_protocol_receipt']
    document['mapping_reports']=[{'group_id':g['group_id'],'report':g['mapping_report']} for g in groups]
    document.pop('model_sha256');document['model_sha256']=digest(document)
    return document


def verify_variant_model(path):
    """Independently re-walk direct and transitive original-raw evidence roles."""
    document=read(path);context=document['execution_context_sha256']
    _verify_duration_definition(document)
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
    if (omitted is not None or omitted_seed is None or len(omitted_ids)!=4 or len(flows)!=4
            or len({j['seed'] for j in declared.values()})!=2):
        raise ValueError('Duration CV models must exclude exactly one whole seed of four flows')
    return {'verified':True,'variant_model_verified':True,'variant':document['variant'],'estimator':'duration_variants_WLS.v1','model':receipt(path),'execution_context_sha256':context,
        'training_flow_count':len(flows),'training_flow_ids':sorted(flows),'training_trace_ids':trace_ids,
        'source_group_count':len(groups),'omitted_flow_id':omitted,'omitted_flow_ids':omitted_ids,'omitted_seed':omitted_seed,
        'training_raw_receipts':raws,'direct_and_transitive_sources_verified':True,'validation_sources_used':[],
        'GPU_started':False}


def _fit_variant(protocol_path,output,variant):
    """Exclusive two-whole-seed refit; no full or per-flow model fit."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    protocol=_verify_protocol(protocol_path)
    bundle=original.load_training(protocol['frozen_training_protocol']['path'])
    bundle['duration_protocol_receipt']=receipt(protocol_path)
    continuous=[g for g in bundle['groups'] if g['role']=='continuous_training']
    seeds=sorted({g['seed'] for g in continuous})
    if len(seeds)!=2 or any(sum(g['seed']==seed for g in continuous)!=4 for seed in seeds):
        raise ValueError('Exactly two seeds and four routes per seed required')
    output.mkdir(parents=True,exist_ok=False)
    save(output/'started.json',{'protocol':receipt(protocol_path),'created_ns':time.perf_counter_ns(),
        'once_only_output':True,'GPU_started':False})
    original._materialize_evidence(bundle,output/'training-evidence')
    folder=output/'seed-crossfit';folder.mkdir();folds=[]
    for seed in seeds:
        fid='seed-'+str(seed);dest=folder/fid;dest.mkdir()
        groups=[g for g in bundle['groups'] if g['role']!='continuous_training' or g['seed']!=seed]
        model=_fit_groups(bundle,groups,seed,variant);model_path=dest/'model.json';save(model_path,model)
        verified=verify_variant_model(model_path)
        provider=BackgroundStatisticalProvider(model,bundle['execution_context_sha256'])
        heldout=[g for g in continuous if g['seed']==seed];heldout_rows=[];all_vectors=[]
        for group in heldout:
            vectors=reconstruct_work(group['traces'],provider);all_vectors.extend(vectors['requests'])
            vectors.update(retrospective_training_cross_validation=True,independent_validation=False,
                crossfit_model=receipt(model_path),heldout_source=group['raw'],heldout_excluded_from_fit=True,
                whole_seed_excluded=True,estimator='duration_variants_WLS.v1',variant=variant)
            work_path=dest/(group['group_id']+'-heldout-work.json');save(work_path,vectors)
            heldout_rows.append({'flow_id':group['group_id'],'heldout_raw':group['raw'],'heldout_plan':group['plan'],
                'candidate_id':group['candidate_id'],'heldout_trace_id':group['traces'][0]['id'],'work_vectors':receipt(work_path)})
        diagnostics={'schema':'gsdeploy.duration_leave_one_seed_out_diagnostics.v1','fold_id':fid,
            'verification':verified,'heldout_work':original.reconstruction_summary({'requests':all_vectors}),
            'training_coverage':model['coverage'],'training_weight_and_baseline_audit':original.weight_audit(groups),
            'numerical_policy':POLICY,'independent_validation':False,'GPU_started':False}
        diag_path=dest/'diagnostics.json';save(diag_path,diagnostics)
        folds.append({'fold_id':fid,'unit':'seed','heldout_seed':seed,'model':receipt(model_path),
            'heldout_flows':heldout_rows,'training_flow_ids':[g['group_id'] for g in continuous if g['seed']!=seed],
            'diagnostics':receipt(diag_path)})
        print(json.dumps({'event':'variant_seed_crossfit_complete','variant':variant,'fold_id':fid,
            'known_joint_vectors':diagnostics['heldout_work']['known_joint_vectors']}),flush=True)
    manifest={'schema':'gsdeploy.continuous_seed_crossfit_manifest.v1','protocol':bundle['protocol_receipt'],
        'duration_protocol':receipt(protocol_path),'estimator':'duration_variants_WLS.v1','variant':variant,
        'variant_protocol':receipt(protocol_path),'selection_protocol':protocol['selection_protocol'],
        'training_verifier':{'name':'verify_variant_model','source':receipt(__file__)},
        'folds':folds,'source_receipts':_unique_refs([*bundle['source_receipts'],receipt(protocol_path),*_duration_tools()]),
        'retrospective_training_cross_validation':True,'no_independent_validation':True,
        'model_development_selection_unit':'whole seed','ensemble_enabled':False,'GPU_started':False}
    save(output/'seed-crossfit-manifest.json',manifest)
    final={'schema':'gsdeploy.duration_variant_seed_cv_fit_manifest.v1','protocol':receipt(protocol_path),
        'selection_protocol':protocol['selection_protocol'],'variant':variant,
        'output':receipt(output/'seed-crossfit-manifest.json'),'source_receipts':manifest['source_receipts'],
        'training_only':True,'independent_validation':False,'GPU_started':False,
        'full_model_fitted':False,'leave_one_flow_out_refits':0,'whole_seed_refits':2}
    _unique_refs(final['source_receipts']);save(output/'manifest.json',final);return final


def fit_protocol(protocol_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    protocol=_verify_protocol(protocol_path)
    # Audit all flows before exclusive parent output; each variant is also audited.
    original.load_training(protocol['frozen_training_protocol']['path'])
    output.mkdir(parents=True,exist_ok=False)
    save(output/'started.json',{'protocol':receipt(protocol_path),'GPU_started':False})
    variants=[]
    for variant in VARIANTS:
        result=_fit_variant(protocol_path,output/variant,variant)
        variants.append({'variant':variant,'manifest':result['output']})
    value={'schema':'gsdeploy.duration_variants_training_cv_suite_manifest.v1',
        'protocol':receipt(protocol_path),'variants':variants,'baseline_seed_manifest':read(protocol['selection_protocol']['path'])['baseline_seed_manifest'],
        'GPU_started':False,'independent_validation':False,'new_whole_seed_refits':4,'baseline_refits':0}
    save(output/'manifest.json',value);return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare')
    for field in ('training-protocol','selection-protocol','output'):p.add_argument('--'+field,required=True)
    p=sub.add_parser('fit');p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('verify');p.add_argument('--model',required=True)
    args=parser.parse_args()
    if args.op=='prepare':value=prepare_protocol(args.training_protocol,args.selection_protocol,args.output)
    elif args.op=='fit':value=fit_protocol(args.protocol,args.output)
    else:value=verify_variant_model(args.model)
    print(json.dumps({'status':'VERIFIED' if args.op=='verify' else 'COMPLETE','schema':value.get('schema'),'GPU_started':False}))


if __name__=='__main__':main()
