"""Fit the selected OCR duration variant on all training sources.

Preparation rejects an old-loss selection. Recursive receipts bind the prior
training-only development selection and unchanged numerical engine. A full
fit remains a development model and never implies CV or independent PASS.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
from . import duration_variant_fit as duration, flow_fit as original
from .background_duration_variant import fit_traces
from .variant_selection import select
import tempfile
from .background_statistical import BackgroundStatisticalProvider
from .background_profiles import background_semantic_traces, receipt
from .flow_profiles import map_flow_collection
from .profiling_analysis import digest

read=original.read
save=original.save
_checked=original._checked
_unique_refs=original._unique_refs
_tools=original._tools
_training_trace=original._training_trace
_duration_tools=duration._duration_tools
_verify_duration_definition=duration._verify_duration_definition
DURATION_DEFINITION=duration.variant_definition("ocr_handler")
TREE_SCHEMA=original.TREE_SCHEMA
PROTOCOL_SCHEMA=original.PROTOCOL_SCHEMA
EVIDENCE_SCHEMA=original.EVIDENCE_SCHEMA
FULL_SCHEMA='gsdeploy.selected_duration_variant_full_fit_protocol.v1'
FULL_POLICY={'numerical_engine':DURATION_DEFINITION,'source_mapping':original.POLICY,
 'fit_unit':'All eight intact continuous training flows plus both old training collections; no exclusions',
 'selection_required':'ocr_handler selected by frozen three-candidate training-only CV rule',
 'independent_validation':False,'CV_PASS_inferred':False,'GPU_started':False}


def verify_selection(path):
    selection=read(path)
    if (selection.get('schema')!='gsdeploy.duration_variants_training_cv_selection_result.v1'
            or selection.get('selected_variant')!='ocr_handler'
            or selection.get('independent_validation') is not False
            or selection.get('final_gain_eligible') is not False
            or type(selection.get('original_training_cv_thresholds_passed')) is not bool
            or not selection.get('source_receipts')):
        raise ValueError('Explicit training-only OCR variant winner required before full fit')
    visited={};pending=[receipt(path)]
    def refs_in(value):
        if isinstance(value,dict):
            if set(value)=={'path','sha256'}:yield value
            else:
                for v in value.values():yield from refs_in(v)
        elif isinstance(value,list):
            for v in value:yield from refs_in(v)
    while pending:
        ref=pending.pop();_checked(ref)
        if ref['path'] in visited:
            if visited[ref['path']]!=ref:raise ValueError('Conflicting recursive selection source')
            continue
        visited[ref['path']]=ref
        if Path(ref['path']).suffix=='.json':pending.extend(refs_in(read(ref['path'])))
    # Recompute the frozen training-only selection instead of trusting a winner label.
    reports={}
    for ref in selection['source_receipts']:
        if Path(ref['path']).suffix!='.json':continue
        candidate=read(ref['path'])
        if 'heldout_identity_audit' in candidate and 'candidates' in candidate:
            reports[candidate.get('variant','duration_v12')]=ref['path']
    if set(reports)!={'duration_v12','ocr_handler','phase_ridge1'}:
        raise ValueError('All three audited whole-seed reports required')
    with tempfile.TemporaryDirectory() as folder:
        computed=select(selection['selection_protocol']['path'],reports,Path(folder)/'selection.json')
    if computed!=selection:raise ValueError('Frozen selection result does not match its reports')
    return {'selection':receipt(path),'recursive_source_receipts':list(visited.values()),
        'selected_variant':selection['selected_variant'],
        'original_training_cv_thresholds_passed':selection['original_training_cv_thresholds_passed']}


def prepare_protocol(duration_cv_protocol,selection_path,output):
    if Path(output).exists():raise FileExistsError(output)
    numerical=duration._verify_protocol(duration_cv_protocol);verification=verify_selection(selection_path)
    threshold=read(numerical['selection_protocol']['path'])['original_threshold_addendum']
    required={numerical['selection_protocol']['path']:numerical['selection_protocol'],
        threshold['path']:threshold}
    actual={r['path']:r for r in verification['recursive_source_receipts']}
    if any(actual.get(k)!=v for k,v in required.items()):raise ValueError('Full fit selection omits frozen rule or thresholds')
    value={'schema':FULL_SCHEMA,'created_ns':time.perf_counter_ns(),'policy':FULL_POLICY,
        'duration_cv_protocol':receipt(duration_cv_protocol),'selection':receipt(selection_path),
        'selection_verification':verification,'source_receipts':_unique_refs([
            receipt(duration_cv_protocol),receipt(selection_path),receipt(__file__),*_duration_tools()]),
        'GPU_started':False,'independent_validation':False}
    value['protocol_sha256']=digest(value);save(output,value);return value


def _verify_full_protocol(path):
    value=read(path)
    if (value.get('schema')!=FULL_SCHEMA or value.get('policy')!=FULL_POLICY
            or value.get('protocol_sha256')!=digest({k:v for k,v in value.items() if k!='protocol_sha256'})):
        raise ValueError('Frozen selected-duration full-fit protocol changed')
    _unique_refs(value['source_receipts']);duration._verify_protocol(_checked(value['duration_cv_protocol']))
    if value['selection_verification']!=verify_selection(_checked(value['selection'])):
        raise ValueError('Recursive selection evidence changed')
    expected=_unique_refs([value['duration_cv_protocol'],value['selection'],receipt(__file__),*_duration_tools()])
    if value['source_receipts']!=expected:raise ValueError('Full fitter tool sources changed')
    return value


def verify_variant_full_model(path):
    """Independently re-walk direct and transitive original-raw evidence roles."""
    document=read(path);context=document['execution_context_sha256']
    _verify_duration_definition(document)
    duration_protocol_ref=document['training_evidence_tree']['duration_protocol']
    duration_protocol=duration._verify_protocol(duration_protocol_ref['path'])
    full_ref=document['training_evidence_tree']['full_fit_protocol']
    full_protocol=_verify_full_protocol(full_ref['path']);_checked(full_ref)
    if full_protocol['duration_cv_protocol']!=duration_protocol_ref:
        raise ValueError('Full and CV numerical protocol differ')
    _checked(duration_protocol_ref)
    if document.get('training_policy')!=FULL_POLICY or document.get('duration_fit_protocol')!=duration_protocol_ref or document.get('full_fit_protocol')!=full_ref:
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
    expected_direct=_unique_refs([tree['protocol'],duration_protocol_ref,full_ref,*_tools(),*_duration_tools(),receipt(__file__),*[g['direct_source'] for g in groups]])
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
    if omitted is not None or omitted_seed is not None or omitted_ids or len(flows)!=8:
        raise ValueError('Full duration model must include all eight flows without exclusions')
    return {'verified':True,'duration_model_verified':True,'variant_full_model_verified':True,'variant':document['variant'],'estimator':DURATION_DEFINITION['estimator'],'model':receipt(path),'execution_context_sha256':context,
        'training_flow_count':len(flows),'training_flow_ids':sorted(flows),'training_trace_ids':trace_ids,
        'source_group_count':len(groups),'omitted_flow_id':omitted,'omitted_flow_ids':omitted_ids,'omitted_seed':omitted_seed,
        'training_raw_receipts':raws,'direct_and_transitive_sources_verified':True,'validation_sources_used':[],
        'GPU_started':False}


def fit_full_protocol(protocol_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    full_protocol=_verify_full_protocol(protocol_path)
    cv_protocol=duration._verify_protocol(full_protocol['duration_cv_protocol']['path'])
    bundle=original.load_training(cv_protocol['frozen_training_protocol']['path'])
    continuous=[g for g in bundle['groups'] if g['role']=='continuous_training']
    if len(continuous)!=8:raise ValueError('All eight original training flows required')
    output.mkdir(parents=True,exist_ok=False)
    save(output/'started.json',{'protocol':receipt(protocol_path),'created_ns':time.perf_counter_ns(),'GPU_started':False})
    original._materialize_evidence(bundle,output/'training-evidence')
    sources=_unique_refs([bundle['protocol_receipt'],full_protocol['duration_cv_protocol'],receipt(protocol_path),
        *_tools(),*_duration_tools(),receipt(__file__),*[g['direct_source'] for g in bundle['groups']]])
    model=fit_traces([t for g in bundle['groups'] for t in g['traces']],sources=sources,variant='ocr_handler')
    model['variant']='ocr_handler'
    model['training_evidence_tree']={'schema':TREE_SCHEMA,'protocol':bundle['protocol_receipt'],
        'duration_protocol':full_protocol['duration_cv_protocol'],'full_fit_protocol':receipt(protocol_path),
        'omitted_flow_id':None,'omitted_flow_ids':[],'omitted_seed':None,
        'groups':[{k:copy.deepcopy(v) for k,v in g.items() if k in
            ('group_id','role','raw','plan','seed','candidate_id','direct_source','transitive_sources')} for g in bundle['groups']],
        'transitive_source_receipts':_unique_refs([r for g in bundle['groups'] for r in g['transitive_sources']]),
        'continuous_training_flow_count':8,'validation_sources_used':[]}
    model.update(training_policy=FULL_POLICY,source_mapping_policy=original.POLICY,
        duration_fit_protocol=full_protocol['duration_cv_protocol'],full_fit_protocol=receipt(protocol_path),
        mapping_reports=[{'group_id':g['group_id'],'report':g['mapping_report']} for g in bundle['groups']])
    model.pop('model_sha256');model['model_sha256']=digest(model)
    model_path=output/'training-variant-full-model.json';save(model_path,model)
    verification=verify_variant_full_model(model_path);save(output/'model-source-verification.json',verification)
    diagnostics=original._diagnostics(model,bundle['groups']);save(output/'training-variant-full-diagnostics.json',diagnostics)
    manifest={'schema':'gsdeploy.selected_duration_variant_full_fit_manifest.v1','protocol':receipt(protocol_path),
        'model':receipt(model_path),'verification':receipt(output/'model-source-verification.json'),
        'diagnostics':receipt(output/'training-variant-full-diagnostics.json'),
        'source_receipts':_unique_refs([*bundle['source_receipts'],*full_protocol['source_receipts']]),
        'selection':full_protocol['selection'],'all_training_flows':8,'validation_sources_used':[],
        'original_training_cv_thresholds_passed':full_protocol['selection_verification']['original_training_cv_thresholds_passed'],
        'independent_validation':False,'GPU_started':False,'core_enabled':False}
    _verify_full_protocol(protocol_path);save(output/'manifest.json',manifest);return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare')
    for name in ('duration-cv-protocol','selection','output'):p.add_argument('--'+name,required=True)
    p=sub.add_parser('fit');p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('verify');p.add_argument('--model',required=True)
    args=parser.parse_args()
    if args.op=='prepare':value=prepare_protocol(args.duration_cv_protocol,args.selection,args.output)
    elif args.op=='fit':value=fit_full_protocol(args.protocol,args.output)
    else:value=verify_variant_full_model(args.model)
    print(json.dumps({'status':'VERIFIED' if args.op=='verify' else 'COMPLETE','schema':value.get('schema'),'GPU_started':False}))


if __name__=='__main__':main()


