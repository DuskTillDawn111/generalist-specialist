"""CPU-only exact-input macro training supplement; no execution or fit entry.

Merged pair/all-owner calendars cover every directed target mode. Existing
training evidence and the historical public validation cohort stay immutable.
Planned activity is never measured overlap, phase rank or resource support.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from itertools import combinations
import json
import math
from pathlib import Path
import uuid

from . import twoFIFO_prospective_contract_v1 as public
from .implementation_source_closure_v27 import implementation_sources

SCHEMA = 'gsdeploy.twoFIFO_matching_macro_supplement.v1'
OWNERS = ('text#0','text#1','generalist_2#0','ocr#0')
MEASUREMENT_NS, DRAIN_NS = 10_000_000_000, 60_000_000_000
MAX_CASES = 13
PAIR_GAPS_NS = (1_000_000, 20_000_000, 300_000_000)
ALL_FOUR_STEPS_NS = (1_000_000, 20_000_000, 50_000_000)
need, receipt, checked, digest, save = public.need, public.receipt, public.checked, public.digest, public.save


def required_groups(contract):
    groups = {owner:[] for owner in OWNERS}
    keys = sorted({(c['routes'][r['request_id']],r['task'],r['pool_id'])
        for c in contract['candidates'] for r in contract['trace']})
    need(set(contract['aliases']) == set(OWNERS), 'Original four resident owners required')
    for owner,task,pool in keys:
        need(owner in groups and task in contract['aliases'][owner]['supported_tasks'], 'Unsupported target key')
        groups[owner].append(dict(owner=owner,task=task,pool_id=pool))
    need(all(groups.values()), 'Every original owner needs an explicit target key')
    return groups


def definitions(groups, block_index=0):
    """Round-robin set coverage, fixed by public keys, independent of outcomes."""
    need(type(block_index) is int and 0 <= block_index < 3, 'One predeclared training block index required')
    result = []
    def add(mode, members, arrivals):
        ordered = [dict(member,arrival_ns=arrival) for member,arrival in zip(members,arrivals)]
        ordered.sort(key=lambda r:(r['arrival_ns'],OWNERS.index(r['owner'])))
        result.append(dict(case_id='case-'+str(len(result)+1).zfill(3),mode=mode,
            intended_active_owners=sorted({r['owner'] for r in members}),arrivals=ordered,
            actual_peer_overlap_known=False, actual_phase_coverage_known=False))
    for owner in OWNERS:
        for key in groups[owner]:
            add('solo_same_owner_burst',[key,key],[0,1_000_000])
    for left,right in combinations(OWNERS,2):
        for index in range(max(len(groups[left]),len(groups[right]))):
            members=[groups[left][index%len(groups[left])],groups[right][index%len(groups[right])]]
            if (block_index+index)%2:members.reverse()
            add('pair_activity',members,[0,PAIR_GAPS_NS[block_index]])
    for index in range(max(map(len,groups.values()))):
        rotation=(index+block_index)%4
        order=OWNERS[rotation:]+OWNERS[:rotation]
        if (block_index+index)%2:order=order[::-1]
        add('all_four_activity',[groups[o][index%len(groups[o])] for o in order],
            [ALL_FOUR_STEPS_NS[block_index]*i for i in range(4)])
    return result


def coverage(groups, cases):
    """Prove every key has solo, each other owner alone, and full roster intent."""
    required = [(key['owner'],key['task'],key['pool_id']) for values in groups.values() for key in values]
    rows=[]
    for owner,task,pool in required:
        modes={'solo':[],**{'pair:'+peer:[] for peer in OWNERS if peer!=owner},'all_four':[]}
        for case in cases:
            active={r['owner'] for r in case['arrivals']}
            if not any((r['owner'],r['task'],r['pool_id'])==(owner,task,pool) for r in case['arrivals']):continue
            if active=={owner}:modes['solo'].append(case['case_id'])
            if len(active)==2:modes['pair:'+next(iter(active-{owner}))].append(case['case_id'])
            if active==set(OWNERS):modes['all_four'].append(case['case_id'])
        need(all(modes.values()), 'A required directed target activity mode is missing: '+repr((owner,task,pool)))
        rows.append(dict(owner=owner,task=task,pool_id=pool,mode_case_ids=modes))
    return dict(required_keys=len(rows),directed_pair_modes=sum(len(OWNERS)-1 for _ in rows),
        target_mode_count=sum(len(r['mode_case_ids']) for r in rows),all_declared_modes_covered=True,
        measured_overlap_inferred_from_calendar=False,keys=rows)


def load_pool(ref):
    need(receipt(ref['path'])==ref, 'Input pool receipt changed')
    rows=[json.loads(line) for line in Path(ref['path']).read_text().splitlines()]
    need(len({r['pool_id'] for r in rows})==len(rows), 'Duplicate pool input identity')
    for row in rows:
        need(row['sample']['task']==row['task_type'], 'Prepared task/payload mismatch')
        if row['task_type']!='text':
            need(receipt(row['sample']['input'])['sha256']==row['sample']['input_sha256'], 'Input media bytes changed')
    need(receipt(ref['path'])==ref, 'Input pool changed during read')
    return rows


def union_inputs(context, old_rows, public_rows, contract):
    old={r['pool_id']:r for r in old_rows}; original={r['pool_id']:r for r in context['declared_inputs']}
    need(set(old)==set(original), 'Original pool and declared input domain differ')
    for pool,row in old.items():
        need(row['task_type']==original[pool]['task'] and public.input_identity(row)==original[pool]['exact_input_sha256'],
            'Original input declaration/content mismatch')
    available={r['pool_id']:r for r in public_rows}
    wanted={r['pool_id']:r['task'] for r in contract['trace']}
    added=[]
    for pool,task in sorted(wanted.items()):
        need(pool in available and available[pool]['task_type']==task, 'Missing original public input bytes')
        candidate=available[pool]
        if pool in old:
            need(public.input_identity(old[pool])==public.input_identity(candidate), 'Same public input ID changed content')
        else:
            old[pool]=copy.deepcopy(candidate);added.append(pool)
    declarations=copy.deepcopy(context['declared_inputs'])
    for pool in added:
        row=old[pool];sample=row['sample'];task=row['task_type']
        size=(len(sample['context'])+len(sample['question']) if task=='text' else
            sample['duration_s'] if task=='asr' else Path(sample['input']).stat().st_size)
        need(type(size) in (int,float) and math.isfinite(size) and size>0, 'Positive explicit input size required')
        declarations.append(dict(task=task,pool_id=pool,input_size=size,
            exact_input_sha256=public.input_identity(row)))
    return [old[p] for p in sorted(old)],declarations,added


def check_sources(context, old_plan):
    """Authenticate retained executable bytes, without asserting transfer."""
    need(context.get('execution_sources') and old_plan.get('source_freeze'), 'Nonempty retained executable source closure required')
    for item in context['execution_sources']:
        need(receipt(item['frozen']['path'])==item['frozen']
            and receipt(item['original_path'])['sha256']==item['frozen']['sha256'], 'Retained execution source drift')
    for item in old_plan['source_freeze']:
        need(receipt(item['frozen']['path'])==item['frozen']
            and receipt(item['original_path'])['sha256']==item['frozen']['sha256'], 'Retained collector source drift')


def build_contracts(preparation, namespace, output):
    """Pure deterministic reconstruction from public inputs; no raw/model reads."""
    need(len(namespace)==32 and all(c in '0123456789abcdef' for c in namespace), 'Explicit 128-bit hexadecimal namespace required')
    prepared=public.audit(preparation['path']);need(receipt(preparation['path'])==preparation, 'Prospective public preparation changed')
    contract=checked(prepared['artifacts']['public_contract']);context=checked(prepared['training_context'])
    old_plans=[checked(ref) for ref in prepared['training_plans']];old_plan=old_plans[0]
    check_sources(context,old_plan)
    need(all(p['prepared_pool']==old_plan['prepared_pool'] and p['priming']==old_plan['priming']
        and p['runtime_policy']==old_plan['runtime_policy'] for p in old_plans), 'One original runtime/priming/pool contract required')
    old_rows=load_pool(old_plan['prepared_pool']);public_rows=load_pool(prepared['inputs']['public_catalog'])
    pool,declarations,added=union_inputs(context,old_rows,public_rows,contract)
    out=Path(output).resolve();candidate=copy.deepcopy(context)
    old_digest=candidate.pop('source_rebound_context_sha256',None)
    candidate['declared_inputs']=declarations
    candidate['matching_macro_supplement']=dict(schema=SCHEMA+'.context',parent=prepared['training_context'],
        namespace=namespace,added_input_ids=added,only_declared_input_domain_extended=True,
        inherited_source_rebound_digest_is_parent_only=old_digest,
        statistical_context_transfer_qualified=False,physical_context_transfer_qualified=False)
    candidate['matching_context_sha256']=digest(candidate)
    groups=required_groups(contract)
    calendar_design=dict(schema=SCHEMA+'.predeclared_block_calendars',
        offset_selection_uses_outcomes=False,actual_overlap_guaranteed=False,
        blocks=[dict(block_index=i,pair_start_gap_ns=PAIR_GAPS_NS[i],
            all_four_step_ns=ALL_FOUR_STEPS_NS[i],cases=definitions(groups,i)) for i in range(3)])
    proof=coverage(groups,calendar_design['blocks'][0]['cases'])
    proof['by_block']=[dict(block_index=i,**coverage(groups,b['cases']))
        for i,b in enumerate(calendar_design['blocks'])]
    old_ids={a['request_id'] for p in old_plans for c in p['cases'] for a in c['arrivals']}
    old_ids|={r['request_id'] for r in contract['trace']}
    blocks=[]
    for repeat in range(3):
        cases=calendar_design['blocks'][repeat]['cases']
        block_id='matching-training-'+namespace+'-block-'+str(repeat+1)
        shift=repeat*17%len(cases);ordered=cases[shift:]+cases[:shift]
        shards=[]
        for start in range(0,len(ordered),MAX_CASES):
            shard_index=len(shards)+1;attempt_id=block_id+'-shard-'+str(shard_index);bound=[]
            for definition in ordered[start:start+MAX_CASES]:
                case=copy.deepcopy(definition)
                for ordinal,row in enumerate(case['arrivals']):
                    row.update(request_id=attempt_id+':'+case['case_id']+':req'+str(ordinal),arrival_index=ordinal)
                    need(row['request_id'] not in old_ids, 'Training and historical/validation request IDs overlap')
                case.update(offered=len(case['arrivals']),public_arrivals_sha256=digest(case['arrivals']))
                bound.append(case)
            shards.append(dict(schema=SCHEMA+'.attempt',split='training',attempt_id=attempt_id,capture_id=block_id,
                shard_index=shard_index,shards_in_block=math.ceil(len(cases)/MAX_CASES),
                execution_id=uuid.uuid5(uuid.NAMESPACE_URL,namespace+':'+attempt_id).hex,
                cases=bound,measurement_ns=MEASUREMENT_NS,drain_ns=DRAIN_NS,resident_owners=list(OWNERS),
                runtime_policy=copy.deepcopy(old_plan['runtime_policy']),runtime_config=copy.deepcopy(old_plan['runtime_config']),
                priming=copy.deepcopy(old_plan['priming']),parent_context=prepared['training_context'],
                context_path=str(out/'candidate-context.json'),prepared_pool_path=str(out/'runtime-inputs.jsonl'),
                inherited_collector_source_freeze=copy.deepcopy(old_plan['source_freeze']),
                collector_timing_policy=copy.deepcopy(old_plan['collector_timing_policy']),
                supervisor_budget=copy.deepcopy(old_plan['supervisor_budget']),
                preflight_free_bytes_required=old_plan['preflight_free_bytes_required'],
                no_retries=True,execute_entrypoint=None,GPU_started=False,
                training_role='complete_native_request_macro_template_support_not_kernel_or_resource_fit',
                old_producer_schema_not_claimed=True,all_block_shards_required_for_leaveout=True))
        blocks.append(dict(block_id=block_id,attempts=shards))
    return dict(candidate_context=candidate,pool=pool,added_input_ids=added,groups=groups,
        definitions=calendar_design,coverage=proof,blocks=blocks,
        old_training_campaign=prepared['inputs']['training_campaign'],public_preparation=preparation,
        original_public_plan=prepared['inputs']['public_plan'],original_context=prepared['training_context'],
        old_training_plans=prepared['training_plans'],original_pool=old_plan['prepared_pool'],
        public_catalog=prepared['inputs']['public_catalog'])


def summary(built):
    plans=[p for b in built['blocks'] for p in b['attempts']]
    definitions=built['definitions']['blocks'][0]['cases']
    return dict(training_blocks=len(built['blocks']),attempts=len(plans),
        cases=sum(len(p['cases']) for p in plans),
        measured_requests=sum(c['offered'] for p in plans for c in p['cases']),
        priming_history_requests=sum(len(p['priming']) for p in plans),
        old_pool_inputs=len(built['pool'])-len(built['added_input_ids']),union_pool_inputs=len(built['pool']),
        added_input_ids=built['added_input_ids'],cases_per_block=len(definitions),
        mode_case_counts_per_block=dict(Counter(c['mode'] for c in definitions)),
        pair_start_gap_ns_by_block=list(PAIR_GAPS_NS),all_four_step_ns_by_block=list(ALL_FOUR_STEPS_NS),
        required_owner_task_input_keys=built['coverage']['required_keys'],
        maximum_cases_per_attempt=max(len(p['cases']) for p in plans),
        maximum_fixed_case_seconds_per_attempt=max(len(p['cases']) for p in plans)*70,
        unmerged_five_modes_cases_per_block=5*built['coverage']['required_keys'])


def prepare(public_preparation_path, namespace, output):
    out=Path(output).resolve();need(not out.exists(), 'Exclusive new CPU preparation output required')
    built=build_contracts(receipt(public_preparation_path),namespace,out)
    out.mkdir(parents=True,exist_ok=False)
    artifacts={}
    for key,filename in (('candidate_context','candidate-context.json'),('coverage','coverage.json'),('definitions','case-definitions.json')):
        artifacts[key]=save(out/filename,built[key])
    pool_path=out/'runtime-inputs.jsonl'
    with pool_path.open('x') as stream:
        for row in built['pool']:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    artifacts['pool']=receipt(pool_path)
    blocks=[]
    for block in built['blocks']:
        refs=[]
        for plan in block['attempts']:
            folder=out/plan['attempt_id'];folder.mkdir();refs.append(save(folder/'plan.json',plan))
        blocks.append(dict(block_id=block['block_id'],attempts=refs))
    report=dict(schema=SCHEMA,status='CPU_PREPARED_MATCHING_MACRO_TRAINING_NO_EXECUTOR',namespace=namespace,
        public_preparation=built['public_preparation'],original_public_plan=built['original_public_plan'],
        old_training_campaign=built['old_training_campaign'],old_training_plans=built['old_training_plans'],
        original_context=built['original_context'],original_pool=built['original_pool'],public_catalog=built['public_catalog'],
        artifacts=artifacts,blocks=blocks,summary=summary(built),
        source_closure=implementation_sources(__file__),
        execute_entrypoint=None,numerical_fit_run=False,GPU_started=False,validation_labels_read=False,
        old_training_and_validation_artifacts_modified=False,qualified_fitted_model=False,
        resource_fit_eligible=False,kernel_resource_fit=False,main_queue_provider_eligible=False,final_gain_eligible=False,
        required_native_checks=['complete new-schema owned collection/review and every offered request retained',
            'all three whole blocks; full and each leaveout retain all macros and exact nuisance groups',
            'observed causal paths, positive OCR n0/npositive coverage and same/count contrast',
            'original twelve-feature residual rank, local root and second-order numerical gates',
            'all START/READY/RESULT/query and sender/reader/collector FIFO clocks authenticated'],
        unresolved=['New-schema owned executor/review/intake adapters are not implemented',
            'Calendar activity does not prove actual peer or phase overlap',
            'New exact input shapes and backend/context transfer are not qualified',
            'Macro supplement does not certify resource fullstates, history, startup or 90+120s support',
            'Fresh independent validation plan and predictions remain unsealed'])
    save(out/'manifest.json',report);audit(out/'manifest.json')
    return report


def audit(manifest_path):
    manifest=checked(receipt(manifest_path));out=Path(manifest_path).resolve().parent
    need(manifest.get('schema')==SCHEMA and manifest.get('status')=='CPU_PREPARED_MATCHING_MACRO_TRAINING_NO_EXECUTOR'
        and manifest.get('execute_entrypoint') is None and all(manifest.get(k) is False for k in
            ('numerical_fit_run','GPU_started','validation_labels_read','old_training_and_validation_artifacts_modified',
            'qualified_fitted_model','resource_fit_eligible','kernel_resource_fit','main_queue_provider_eligible','final_gain_eligible')),
        'CPU-only supplementary macro preparation required')
    need(manifest['source_closure']==implementation_sources(__file__), 'Supplement implementation changed')
    for ref in manifest['source_closure']:need(receipt(ref['path'])==ref, 'Supplement source receipt changed')
    built=build_contracts(manifest['public_preparation'],manifest['namespace'],out)
    for key in ('original_public_plan','old_training_campaign','old_training_plans','original_context','original_pool','public_catalog'):
        need(manifest[key]==built[key], 'Supplement parent lineage changed: '+key)
    for key in ('candidate_context','coverage','definitions'):
        need(checked(manifest['artifacts'][key])==built[key], 'Supplement content changed: '+key)
    need(load_pool(manifest['artifacts']['pool'])==built['pool'], 'Exact union input catalog changed')
    need(len(manifest['blocks'])==len(built['blocks']), 'Three complete supplementary blocks required')
    for saved,expected in zip(manifest['blocks'],built['blocks']):
        need(saved['block_id']==expected['block_id'] and len(saved['attempts'])==len(expected['attempts']), 'Complete whole-block shards required')
        for ref,plan in zip(saved['attempts'],expected['attempts']):
            need(Path(ref['path'])==out/plan['attempt_id']/'plan.json' and checked(ref)==plan, 'Exact supplementary plan identity/calendar changed')
    need(manifest['summary']==summary(built), 'Supplement summary changed')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--public-preparation',required=True);p.add_argument('--namespace',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('audit');p.add_argument('--manifest',required=True)
    args=parser.parse_args()
    value=(prepare(args.public_preparation,args.namespace,args.output) if args.command=='prepare' else audit(args.manifest))
    print(json.dumps(dict(status=value['status'],summary=value['summary'])))


if __name__=='__main__':main()
