"""Strict repeated-instance template adapter for unified native training intakes.

Each native response remains a separate request observation. Shared baseline
identity is released only by exact work, full launch, dependency and audited
static equivalence across every measured instance. No collection or fit runs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from gsdeploy.resources import identity
from . import kernel_dag_capture_alignment_v1 as core
from . import kernel_dag_response_training_v1 as compiler
from .implementation_source_closure_v27 import implementation_sources, strict_references
from . import kernel_dag_response_replication_producer_v1 as producer
from .kernel_domain_bridge_review_v3 import request_kernels
from .kernel_sequence_audit_v2 import IDENTITY_FIELDS, verify
from .native_collect import receipt
from .ordered_occupancy_resolver_v2 import AuditedOccupancyResolver
from .unified_corun_review_v8_1 import load_review, LIGHT
from .unified_kernel_dag_intake_reauthentication_v1 import authenticate as reauthenticate_intake

SCHEMA='gsdeploy.kernel_dag_repeated_instance_alignment.v1'
PRODUCER_SCHEMA=producer.SCHEMA


def read(ref):
    verify(ref);return json.loads(Path(ref['path']).read_text())


def write(path,value):
    # These large immutable exports retain every field without indentation.
    with Path(path).open('x') as stream:json.dump(value,stream,separators=(',',':'));stream.write('\n')
    return receipt(path)


def load_member(entry):
    review=read(entry['producer_review'])
    core.need(review.get('schema')==PRODUCER_SCHEMA+'.review'
              and review.get('status')=='PASS_CPU_MATRIX_NATIVE_BINDING_ONLY'
              and review.get('training_only') is True and review.get('validation_labels_read') is False,
              'Explicit native matrix training review required before instance labels')
    core.need(producer.authenticate_review(review['campaign']['path'],review['capture_id'],
              review['native_review']['path'])==review,'Fresh exclusive claim/execution/after-matching review differs')
    authenticated=reauthenticate_intake(entry,review['native_review'])
    intake=authenticated['report']
    core.need(intake.get('schema')=='gsdeploy.unified_kernel_dag_intake.v1'
              and intake.get('status')=='COMPLETE_NATIVE_MAPPING_PATH_DIAGNOSTICS_NO_FIT'
              and intake.get('validation_labels_read') is False and intake.get('native_mapping_authenticated') is True
              and intake.get('common_native_session_clock') is True
              and intake.get('rejected_kernel_count')==0
              and intake.get('full_session_inventory',{}).get('outside_measured_overlap_wall_ns')==0,
              'Complete authenticated no-outside-peer unified intake required')
    core.need(intake['review']==review['native_review'] and intake['raw']==review['raw']
              and intake['sqlite']==review['sqlite'],'Matrix/native intake source binding differs')
    for ref in intake['sources']:verify(ref)
    strict,plan,raw=load_review(review['native_review']['path'],LIGHT)
    core.need(strict.get('training_only') is True and strict.get('validation_started') is False
              and strict.get('model_fitted') is False,'Explicit strict native training role required')
    campaign=producer.audit(review['campaign']['path'])
    core.need(receipt(review['campaign']['path'])==review['campaign'],'Exact campaign receipt required')
    entries=[e for e in campaign['runs'] if e['capture_id']==review['capture_id']]
    core.need(len(entries)==1 and strict['evidence']['plan.json']==entries[0]['plan']
              and strict['evidence']['context.json']==entries[0]['context'],
              'Native measured lifecycle differs from frozen campaign')
    core.need(producer.validate_records(plan,raw,campaign['exact_owner_inputs'],review['capture_id'])==review['instances'],
              'Fresh complete native instance mapping differs from producer review')
    catalog=authenticated['catalog'];capture=authenticated['capture']
    core.need(catalog.get('training_only') is True and catalog.get('validation_labels_read') is False
              and capture.get('split')=='training' and capture.get('validation_labels_read') is False,
              'Catalog/capture roles required before profile access')
    context=read(strict['evidence']['context.json']);parsed=read(strict['trace']['parsed'])
    profiles={p['request_id']:p for p in catalog['profiles']}
    rows={row['request_id']:row for row in raw['request_records']}
    instances={r['native_request_id']:r for r in review['instances']}
    core.need(len(profiles)==len(rows)==len(instances)==review['full_denominator']==20
              and set(profiles)==set(rows)==set(instances)==set(capture['offered_request_ids']),
              'Complete distinct measured instance denominator required')
    core.need(capture['raw_sha256']==intake['raw']['sha256']
              and capture['execution_context_sha256']==catalog['execution_context_sha256']==intake['execution_context_sha256'],
              'Native raw/full context differs')
    # Hardware was pinned by the exact-source occupancy receipt in the intake.
    # Require an explicit exact-source audit pin, never infer it from filenames.
    hardware=AuditedOccupancyResolver(entry['occupancy_audit']['path']).hardware
    core.need(entry['occupancy_audit'] in intake['sources'] and identity(hardware)==intake['hardware_sha256'],
              'Exact intake occupancy/full hardware binding required')
    observed=[]
    for rid,row in rows.items():
        core.need(instances[rid]['owner']==row['model'] and instances[rid]['job_id']==row['profile_job_id'],
                  'Request-instance provenance differs')
        binding=core.request_binding(row,plan)
        core.need(binding['exact_input_sha256']==instances[rid]['exact_input_sha256'],
                  'Exact measured input bytes differ')
        items=request_kernels(parsed,intake['sqlite']['path'],row)
        common,mapping,static=core.template(profiles[rid],items,binding)
        observed.append(dict(request_id=rid,instance=instances[rid],binding=binding,
            work_sha256=identity(binding),common=common,mapping=mapping,static=static,profile=profiles[rid]))
    sources=strict_references([entry,review,intake,context,implementation_sources(__file__)])
    for ref in sources:verify(ref)
    return dict(review=review,intake=intake,catalog=catalog,capture=capture,context=context,
                hardware=hardware,observed=observed,sources=sources,
                reauthenticated_from_sqlite=authenticated['reauthentication']['native_sqlite_reparsed'])


def align(design_path,output):
    design_ref=receipt(design_path);design=read(design_ref)
    core.need(design.get('schema')==SCHEMA+'.design' and design.get('training_only') is True
              and design.get('validation_labels_read') is False and len(design['captures'])==3,
              'Explicit complete three-capture training instance design required')
    implementation=implementation_sources(__file__);members=[load_member(e) for e in design['captures']]
    core.need(len({m['capture']['raw_sha256'] for m in members})==3
              and len({m['review']['capture_id'] for m in members})==3
              and len({m['review']['sqlite']['sha256'] for m in members})==3,
              'Independent native raw/lifecycle/session identities required')
    first=members[0]
    core.need(all(m['context']==first['context'] and m['hardware']==first['hardware'] for m in members),
              'Exact full frozen context/hardware required without normalization')
    core.need(all(m['review']['campaign']==first['review']['campaign'] for m in members),
              'All measured captures must bind one frozen campaign')
    cohorts=[{(o['instance']['job_id'],o['instance']['owner']) for o in m['observed']} for m in members]
    core.need(all(c==cohorts[0] and len(c)==20 for c in cohorts),'Complete owner/job instance cohort differs')
    grouped=defaultdict(list)
    for member in members:
        for observed in member['observed']:grouped[observed['work_sha256']].append((member,observed))
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    matched=[];rejected=[];common_by_work={}
    for key,observations in grouped.items():
        common=observations[0][1]['common']
        # One exact owner/input/work must cover five settings in each capture.
        complete=len(observations)==15 and all(sum(m is member for m,_ in observations)==5 for member in members)
        if not complete or not all(o['common']==common for _,o in observations):
            rejected.append(dict(work_sha256=key,offered_instances=len(observations),
                instance_ids=[o['instance']['instance_id'] for _,o in observations],
                reason='INCOMPLETE_EXACT_WORK_OR_FULL_DAG_STATIC_EQUIVALENCE'))
            continue
        template_id='exact-native-template:'+identity(common)
        proof=dict(schema=SCHEMA+'.equivalence',template_id=template_id,common=common,
            members=[dict(raw=m['intake']['raw'],sqlite=m['intake']['sqlite'],
                instance=o['instance'],native_to_template_node=o['mapping']) for m,o in observations],
            response_timestamps_used_for_equivalence=False,numerical_qualification=False)
        proof_ref=write(output/('template-'+key+'.json'),proof)
        static={node_id:dict(values,static_evidence=dict(exact_equivalence=proof_ref,template_node=node_id),
            allocation_evidence=dict(exact_equivalence=proof_ref,template_node=node_id))
            for node_id,values in observations[0][1]['static'].items()}
        common_by_work[key]=dict(template_id=template_id,proof=proof_ref,static=static,common=common,
                                family=observations[0][1]['profile']['family'])
        matched.append(dict(work_sha256=key,template_id=template_id,instances=len(observations),
                            kernel_nodes=len(static),equivalence=proof_ref))
    sources=strict_references([design_ref,design,implementation,[m['sources'] for m in members]])
    exports={};counts=None
    if not rejected and len(common_by_work)==4:
        catalog=dict(schema=SCHEMA+'.compiler_catalog',training_only=True,validation_labels_read=False,
            execution_context_sha256=first['capture']['execution_context_sha256'],profiles=[],sources=sources)
        for item in common_by_work.values():
            common=item['common'];binding=common['work']
            catalog['profiles'].append(dict(alias=binding['alias'],family=item['family'],request_id=item['template_id'],
                task=binding['task'],pool_id=binding['pool_id'],split='training',unresolved=[],cyclic=False,
                edges=common['edges'],nodes=[dict(id=n['id'],kind=n['kind'],
                    **({'kernel':item['static'][n['id']]} if n['kind']=='gpu' else {})) for n in common['nodes']]))
        captures=[]
        for member in members:
            old=member['capture'];capture={k:v for k,v in old.items() if k not in ('requests','kernels')}
            capture.update(capture_id=member['review']['capture_id'],requests=[],kernels=[])
            by_request={o['request_id']:o for o in member['observed']}
            for row in old['requests']:
                observed=by_request[row['request_id']];item=common_by_work[observed['work_sha256']]
                capture['requests'].append(dict(row,profile_id=item['template_id']))
            for kernel in old['kernels']:
                observed=by_request[kernel['request_id']];item=common_by_work[observed['work_sha256']]
                node=observed['mapping'][kernel['node_id']]
                capture['kernels'].append(dict(kernel,node_id=node,kernel=item['static'][node],
                    native_evidence=dict(intake=member['intake']['artifacts']['capture.json'],
                        native_request_id=kernel['request_id'],native_node_id=kernel['node_id'],
                        identity={f:kernel['kernel'][f] for f in IDENTITY_FIELDS})))
            captures.append(capture)
        prepared=compiler.compile_paths(catalog,first['hardware'],captures)
        counts={key:prepared[key] for key in ('offered_requests','offered_kernel_count','accepted_kernel_count')}
        counts['rejected_kernel_count']=len(prepared['rejected_rows'])
        exports['catalog.json']=write(output/'catalog.json',catalog)
        exports['captures.json']=write(output/'captures.json',captures)
        exports['paths.json']=write(output/'paths.json',dict(counts=counts,observations=prepared['observations'],
            rejected_rows=prepared['rejected_rows'],full_offered_request_registry=prepared['full_offered_request_registry']))
    for ref in sources:verify(ref)
    core.need(implementation_sources(__file__)==implementation,'Implementation changed')
    report=dict(schema=SCHEMA,status='COMPLETE_REPEATED_INSTANCE_TEMPLATE_PATHS_NO_FIT' if exports else 'UNKNOWN_FULL_INSTANCE_COHORT_RETAINED',
        design=design_ref,offered_captures=3,offered_instances=sum(len(m['observed']) for m in members),
        matched=matched,rejected=rejected,compiler_counts=counts,artifacts=exports,sources=sources,
        training_only=True,validation_labels_read=False,GPU_started=False,fit_started=False,
        baseline_identity_uses_exact_work_template=True,native_clock_domains_preserved=True,
        all_intakes_freshly_reauthenticated_from_sqlite=all(m.get('reauthenticated_from_sqlite') is True for m in members),
        numerical_rank_guaranteed=False,numerical_qualification=False)
    write(output/'report.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--design',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=align(args.design,args.output)
    print(json.dumps({k:report[k] for k in ('schema','status','offered_captures','offered_instances')}))
