"""Strict cross-capture work, launch-order and within-request DAG equivalence.

Native identities stay in per-capture mappings. The common template excludes
response timestamps and receipt-specific static evidence only after proving
the complete observed launch/configuration/graph signatures equal. No fit,
new capture, context relaxation or OCR work extrapolation is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from gsdeploy.resources import identity
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .kernel_dag_training_mapping_audit_v1 import require_denominator, validate_mapping
from .kernel_domain_bridge_review_v3 import load_report, request_kernels, compare_sequences
from .kernel_sequence_audit_v2 import IDENTITY_FIELDS, verify
from .native_collect import receipt
from .ordered_occupancy_resolver_v2 import AuditedOccupancyResolver
from .paired_training_work_review_v4 import actual_work
from .request_cuda_dag_v2 import topological

SCHEMA='gsdeploy.kernel_dag_capture_alignment.v1'
STATIC_FIELDS=('name','hardware_id','block','grid','registers_per_thread','static_shared_bytes',
    'dynamic_shared_bytes','allocated_registers_per_block','allocated_shared_bytes',
    'architecture_blocks_per_sm_ceiling','cluster','cooperative_launch')


def need(value, reason):
    if not value:raise ValueError(reason)


def checked(ref):
    verify(ref);return json.loads(Path(ref['path']).read_text())


def require_native_training(strict, plan, raw):
    need(strict.get('training_only') is True and strict.get('validation_started') is False
         and plan.get('training_only') is True and raw['request_records']
         and all(r.get('split')=='training' for r in raw['request_records']),
         'Strict explicit training capture required')


def catalog_role(catalog, entry, strict, plan, raw):
    """Gate before profile access/copy; legacy absence needs a separate receipt."""
    require_native_training(strict,plan,raw)
    need(catalog.get('training_only') is True,'Explicit training-only catalog required')
    if 'validation_labels_read' not in catalog:
        need('catalog_role' in entry,'Missing explicit catalog validation role requires source-bound adapter')
        adapter=checked(entry['catalog_role'])
        need(adapter.get('schema')==SCHEMA+'.catalog_role'
             and adapter.get('training_only') is True and adapter.get('validation_labels_read') is False
             and adapter.get('catalog')==entry['profiles'] and adapter.get('strict_review')==entry['review']
             and adapter.get('mapping')==entry['mapping'] and adapter.get('raw')==strict['raw']
             and adapter.get('plan')==strict['plan']
             and adapter.get('basis')=='EXPLICIT_TRAINING_NATIVE_REVIEW_PLAN_AND_COMPLETE_REQUEST_SPLITS',
             'Exact source-bound training role adapter required')
        catalog=dict(catalog,validation_labels_read=False,catalog_role_adapter=entry['catalog_role'])
    need(catalog.get('validation_labels_read') is False,'Explicit unread-validation catalog required')
    return catalog


def write_catalog_role_adapter(entry, output):
    """Publish a new declaration; never mutate a legacy catalog or its receipts."""
    strict,plan,raw=load_report(entry['review']['path'],'light-kernel')
    need(receipt(entry['review']['path'])==entry['review'],'Exact strict review required')
    require_native_training(strict,plan,raw)
    catalog=checked(entry['profiles']);mapping=checked(entry['mapping'])
    need(catalog.get('training_only') is True and 'validation_labels_read' not in catalog,
         'Role adapter only applies to explicit training legacy catalogs with absent validation role')
    need(mapping.get('schema')=='gsdeploy.kernel_dag_training_mapping_audit.v1'
         and mapping.get('status')=='PASS_CPU_MAPPING_ONLY' and mapping.get('issues')==[]
         and entry['profiles'] in mapping['sources'] and mapping['raw']==strict['raw'],
         'Exact successful source-bound native mapping report required')
    adapter=dict(schema=SCHEMA+'.catalog_role',training_only=True,validation_labels_read=False,
        catalog=entry['profiles'],strict_review=entry['review'],mapping=entry['mapping'],
        raw=strict['raw'],plan=strict['plan'],
        basis='EXPLICIT_TRAINING_NATIVE_REVIEW_PLAN_AND_COMPLETE_REQUEST_SPLITS',
        legacy_catalog_modified=False,implementation=implementation_sources(__file__))
    output=Path(output).resolve()
    with output.open('x') as stream:json.dump(adapter,stream,indent=2);stream.write('\n')
    return receipt(output)


def request_binding(row, plan):
    """Actual output/work and explicit input bytes; pool/name equality is insufficient."""
    inputs=[item for item in plan['inputs'] if item['pool_id']==row['pool_id'] and item['task']==row['task']]
    need(len(inputs)==1 and type(inputs[0].get('exact_input_sha256')) is str
         and len(inputs[0]['exact_input_sha256'])==64,'One exact input-byte declaration required')
    response=row['worker_response']
    need(response.get('success') is True and type(response.get('raw_output')) is str,
         'Actual successful output required, including OCR')
    return dict(alias=row['model'],task=row['task'],pool_id=row['pool_id'],
        exact_input_sha256=inputs[0]['exact_input_sha256'],actual_work=actual_work(response))


def template(profile, items, binding):
    """A deliberately strict canonical labelling; ambiguous host order is rejected."""
    need(not profile['unresolved'] and profile['cyclic'] is False
         and topological(profile['nodes'],profile['edges']),'Complete acyclic request DAG required')
    nodes=profile['nodes'];ids=[n['id'] for n in nodes]
    need(len(ids)==len(set(ids)),'Unique native DAG nodes required')
    issued=[n for n in nodes if n['kind']!='handler_end']
    need(all(type(n.get('issue_start_ns')) is int and type(n.get('issue_end_ns')) is int
             and n['issue_start_ns']<n['issue_end_ns'] for n in issued),'Exact positive native API issue intervals required')
    issued.sort(key=lambda n:n['issue_start_ns'])
    need(len({n['issue_start_ns'] for n in issued})==len(issued),'Ambiguous tied native host issue order')
    # Cross-thread overlap does not establish a reproducible total launch order.
    last_end={}
    for node in issued:
        thread=node['issue_thread']
        need(all(end<=node['issue_start_ns'] for other,end in last_end.items() if other!=thread),
             'Concurrent host API issue order is not a proven template')
        last_end[thread]=node['issue_end_ns']
    terminal=[n for n in nodes if n['kind']=='handler_end']
    need(len(terminal)==1,'Exactly one terminal handler anchor required')
    ordered=issued+terminal;mapping={n['id']:'node:'+str(i) for i,n in enumerate(ordered)}
    by_kernel={tuple(item['kernel'].get(f) for f in IDENTITY_FIELDS):item for item in items}
    need(len(by_kernel)==len(items),'Unique complete ordered native kernel sequence required')
    streams={};threads={};labels=[];kernel_order=[];static={}
    for node in ordered:
        label=dict(id=mapping[node['id']],kind=node['kind'])
        if 'stream_key' in node:
            raw=tuple(node['stream_key']);label['stream_role']=streams.setdefault(raw,len(streams))
        if 'issue_thread' in node:
            label['thread_role']=threads.setdefault(node['issue_thread'],len(threads))
        if 'api_name' in node:label['api_name']=node['api_name']
        if node['kind']=='gpu':
            kernel=node['kernel'];item=by_kernel.get(tuple(kernel.get(f) for f in IDENTITY_FIELDS))
            need(item is not None,'Full GPU node/ordered launch bijection required')
            label.update(launch_signature=item['signature'],semantic_group=item['group'])
            scalar={field:kernel.get(field) for field in STATIC_FIELDS}
            need(kernel.get('static_evidence') and kernel.get('allocation_evidence')
                 and all(scalar.get(f) is not None for f in ('hardware_id','allocated_registers_per_block',
                     'allocated_shared_bytes','architecture_blocks_per_sm_ceiling')),
                 'Exact-source audited static allocation required before common static equivalence')
            label['static']=scalar;static[label['id']]=scalar;kernel_order.append(item)
        elif node['kind'] in ('copy','memset'):
            label['memory_metadata']=node.get('memory_metadata',{})
        elif node['kind'] not in ('event_record','event_wait','host_sync','handler_end'):
            raise ValueError('Unsupported native DAG node kind: '+node['kind'])
        labels.append(label)
    need([item['signature'] for item in kernel_order]==[item['signature'] for item in items]
         and len(kernel_order)==len(items),'Native DAG issue order differs from complete launch sequence')
    edges=[]
    for edge in profile['edges']:
        need(edge['before'] in mapping and edge['after'] in mapping,'DAG edge outside complete node cohort')
        edges.append(dict(before=mapping[edge['before']],after=mapping[edge['after']],kind=edge['kind']))
    edges.sort(key=lambda e:(e['before'],e['after'],e['kind']))
    need(len({(e['before'],e['after'],e['kind']) for e in edges})==len(edges),'Duplicate native DAG edge')
    document=dict(work=binding,nodes=labels,edges=edges,boundary_assumptions=dict(
        pre_request_external_dependencies_assumed_completed=profile['pre_request_external_dependencies_assumed_completed'],
        default_stream_semantics=profile['default_stream_semantics']),
        equivalence_scope='Within-request graph under unchanged published pre-request/default-stream assumptions; no initial-state or natural timing proof')
    return document,mapping,static


def load_capture(entry):
    strict,plan,raw=load_report(entry['review']['path'],'light-kernel')
    need(receipt(entry['review']['path'])==entry['review'],'Exact strict review required')
    require_native_training(strict,plan,raw)
    mapping_report=checked(entry['mapping']);catalog=checked(entry['profiles'])
    need(mapping_report.get('schema')=='gsdeploy.kernel_dag_training_mapping_audit.v1'
         and mapping_report.get('status')=='PASS_CPU_MAPPING_ONLY' and mapping_report.get('issues')==[]
         and entry['profiles'] in mapping_report['sources'] and mapping_report['raw']==strict['raw'],
         'Exact successful source-bound native mapping report required')
    catalog=catalog_role(catalog,entry,strict,plan,raw)
    need(catalog['execution_context_sha256']==raw['execution_context_sha256']==plan['execution_context_sha256'],
         'Unchanged training execution context required')
    profiles={p['request_id']:copy.deepcopy(p) for p in catalog['profiles']}
    require_denominator([r['request_id'] for r in raw['request_records']],list(profiles))
    resolver=AuditedOccupancyResolver(entry['occupancy_audit']['path'])
    context=checked(strict['context']);requests={};sources=strict_references([entry,strict,plan,context,
        mapping_report['sources'],catalog['sources'],checked(entry['occupancy_audit'])])
    for source in sources:verify(source)
    for row in raw['request_records']:
        profile=profiles[row['request_id']];owner=strict['owners'][row['model']]
        parsed=checked(owner['parsed']);items=request_kernels(parsed,owner['sqlite']['path'],row)
        request=dict(row=row,kernels=[dict(item['kernel'],source_sqlite=owner['sqlite']) for item in items])
        validate_mapping(row['request_id'],request,profile,raw['ready'][row['model']]['pid'])
        for node in profile['nodes']:
            if node['kind']=='gpu':
                patched=resolver(node['kernel']);need(patched is not None,'UNKNOWN exact static member cannot be aligned')
                for field in ('block','grid','registers_per_thread','static_shared_bytes','dynamic_shared_bytes','cluster'):
                    need(patched.get(field)==node['kernel'].get(field),'Exact static audit changes observed launch')
                node['kernel'].update(patched)
        binding=request_binding(row,plan);key=identity(binding)
        need(key not in requests,'Repeated exact work needs explicit request-instance mapping')
        requests[key]=dict(profile=profile,items=items,binding=binding)
    return dict(raw=strict['raw'],context=raw['execution_context_sha256'],profiler=context['profiler'],
        hardware=resolver.hardware,requests=requests,sources=sources)


def align(design_path,output):
    design_ref=receipt(design_path);design=checked(design_ref)
    need(design.get('schema')==SCHEMA+'.design' and design.get('training_only') is True
         and design.get('validation_labels_read') is False and len(design['captures'])>=2,
         'Explicit two-or-more training capture alignment design required')
    implementation=implementation_sources(__file__)
    captures=[load_capture(entry) for entry in design['captures']]
    need(len({c['raw']['sha256'] for c in captures})==len(captures),'Raw aliases cannot become independent capture replication')
    first=captures[0]
    need(all(c['context']==first['context'] and c['profiler']==first['profiler']
             and c['hardware']==first['hardware'] for c in captures),'Exact full execution context/profiler/hardware required; no nonce/context normalization')
    need(all(set(c['requests'])==set(first['requests']) for c in captures),'Complete exact-input/actual-work request cohort required across captures')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    matched=[];rejected=[];proofs=[]
    for key,request in first['requests'].items():
        try:
            observed=[c['requests'][key] for c in captures]
            sequences=[compare_sequences(observed[0]['items'],item['items']) for item in observed[1:]]
            need(all(all(g['status']=='EXACT_OBSERVED_SIGNATURE_ORDER_MATCH' for g in groups) for groups in sequences),
                 'Complete launch names/configuration/group order differs')
            normalized=[template(item['profile'],item['items'],item['binding']) for item in observed]
            common=normalized[0][0];need(all(t[0]==common for t in normalized),'Complete normalized DAG dependencies/static/order differ')
            template_id='exact-native-template:'+identity(common)
            proof=dict(schema=SCHEMA+'.equivalence',template_id=template_id,common=common,
                members=[dict(raw=c['raw'],native_profile_id=item['profile']['request_id'],native_to_template_node=mapping)
                         for c,item,(_,mapping,_) in zip(captures,observed,normalized)],
                original_native_evidence_preserved=True,response_timestamps_used_for_equivalence=False,
                numerical_qualification=False,natural_timing_validated=False)
            path=output/('template-'+key+'.json')
            with path.open('x') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
            proof_ref=receipt(path);proofs.append(proof_ref)
            static={node_id:dict(values,static_evidence=dict(exact_equivalence=proof_ref,template_node=node_id),
                allocation_evidence=dict(exact_equivalence=proof_ref,template_node=node_id))
                for node_id,values in normalized[0][2].items()}
            matched.append(dict(work_sha256=key,alias=request['binding']['alias'],pool_id=request['binding']['pool_id'],
                template_id=template_id,kernel_count=len(static),equivalence=proof_ref,
                common_static=static,baseline_identity_uses_common_template_not_native_request_id=True))
        except ValueError as exc:
            rejected.append(dict(work_sha256=key,alias=request['binding']['alias'],pool_id=request['binding']['pool_id'],reason=str(exc)))
    sources=strict_references([design_ref,design,implementation,[c['sources'] for c in captures]])
    for source in sources:verify(source)
    need(implementation_sources(__file__)==implementation,'Alignment implementation changed')
    exports={}
    if not rejected:
        catalog=dict(schema=SCHEMA+'.compiler_catalog',training_only=True,validation_labels_read=False,
            execution_context_sha256=first['context'],profiles=[],sources=sources+proofs,
            native_mapping_authentication_inherited=True)
        common_by_work={item['work_sha256']:item for item in matched}
        for key,item in common_by_work.items():
            common=checked(item['equivalence'])['common'];binding=common['work']
            catalog['profiles'].append(dict(alias=binding['alias'],family=first['requests'][key]['profile']['family'],
                request_id=item['template_id'],task=binding['task'],pool_id=binding['pool_id'],
                split='training',unresolved=[],cyclic=False,edges=common['edges'],
                nodes=[dict(id=n['id'],kind=n['kind'],**({'kernel':item['common_static'][n['id']]}
                       if n['kind']=='gpu' else {})) for n in common['nodes']]))
        compiler_captures=[]
        for capture in captures:
            cap=dict(schema='gsdeploy.kernel_dag_response_training.v1.capture',split='training',
                validation_labels_read=False,evidence_kind='declared_measured',
                capture_id='raw-sha256:'+capture['raw']['sha256'],raw_sha256=capture['raw']['sha256'],
                profiler_id=capture['profiler'],execution_context_sha256=capture['context'],
                hardware_id=capture['hardware']['hardware_id'],offered_request_ids=[],requests=[],kernels=[])
            for key,request in capture['requests'].items():
                item=common_by_work[key];profile=request['profile'];rid=profile['request_id']
                proof=checked(item['equivalence']);member=next(p for p in proof['members'] if p['raw']==capture['raw'])
                cap['offered_request_ids'].append(rid)
                cap['requests'].append(dict(request_id=rid,owner=profile['alias'],profile_id=item['template_id'],
                    task=profile['task'],pool_id=profile['pool_id']))
                for node in profile['nodes']:
                    if node['kind']!='gpu':continue
                    kernel=node['kernel'];template_node=member['native_to_template_node'][node['id']]
                    cap['kernels'].append(dict(request_id=rid,owner=profile['alias'],node_id=template_node,
                        kernel=item['common_static'][template_node],start_ns=kernel['nsys_start_ns'],
                        end_ns=kernel['nsys_end_ns'],clock_domain='native-session-sqlite:'+kernel['source_sqlite']['sha256'],
                        native_evidence=dict(raw=capture['raw'],source_sqlite=kernel['source_sqlite'],
                            native_node_id=node['id'],identity={f:kernel[f] for f in IDENTITY_FIELDS},
                            static_evidence=kernel['static_evidence'],allocation_evidence=kernel['allocation_evidence'])))
            compiler_captures.append(cap)
        for name,value in (('compiler-catalog.json',catalog),('compiler-captures.json',compiler_captures)):
            with (output/name).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
            exports[name]=receipt(output/name)
        # File output can be lengthy. Recheck every source after publication too.
        for source in sources:verify(source)
    report=dict(schema=SCHEMA,status='COMPLETE_EXACT_NATIVE_TEMPLATE_EQUIVALENCE_NO_FIT' if not rejected else 'PARTIAL_OR_UNKNOWN_FULL_COHORT_RETAINED',
        design=design_ref,independent_capture_count=len(captures),raws=[c['raw'] for c in captures],
        execution_context_sha256=first['context'],hardware_sha256=identity(first['hardware']),profiler_id=first['profiler'],
        offered_profiles=len(first['requests']),matched_profiles=len(matched),rejected_profiles=len(rejected),
        matched=matched,rejected=rejected,template_proofs=proofs,sources=sources,
        source_receipts_unchanged=True,native_contexts_relaxed=False,GPU_started=False,fit_started=False,
        numerical_qualification=False,independent_prediction_accuracy_evaluated=False,deployment_qualified=False,
        compiler_input_exported=bool(exports),compiler_inputs=exports,
        limitation='Common static evidence requires exact graph/work equality under original boundary assumptions. Original SQLite clock domains remain separate; mixed isolated sessions still fail the compiler clock gate, and this aligner never pools them.')
    with (output/'report.json').open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--design',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=align(args.design,args.output)
    print(json.dumps({k:report[k] for k in ('status','offered_profiles','matched_profiles','rejected_profiles')}))
