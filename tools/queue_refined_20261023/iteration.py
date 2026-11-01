"""Frozen finite shortlist and serial efficacy workflow for the refined iteration.

Development may select workload and policy. Reserved paired runs compare the
same-residency calibrated spill policy with its same-risk no-spill controller;
the best measured alternative deployment is reported separately. Never refit or
reselect from reserved outcomes, and never interpret table lookup as GPU speedup.
"""
from __future__ import annotations
import argparse
import copy
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.queue_refined_20261023 import protocol,admission
from tools.queue_refined_20261023.analysis import analyze
from tools.queue_refined_20261023.search import measured_table_search
from tools.queue_refined_20261023.workflow import publish
from tools.queue_refined_20261023.native_runner import frozen_runtime

PACKAGE='tools.queue_refined_20261023'
HERE=Path(__file__).resolve().parent
CLAIMS={'development':'NEW_REFINED_DEVELOPMENT_NOT_INDEPENDENT_VALIDATION',
 'independent_validation':'INDEPENDENT_FIXED_SELECTION_NATIVE_VALIDATION',
 'final_confirmation':'FINAL_FIXED_SELECTION_NATIVE_CONFIRMATION'}
PRIMARY={'text':2,'generalist_2':1,'ocr':1}


def read(path):return json.loads(Path(path).read_text())
def symbolic(job):
    c=job['policy_contract']
    return dict(deployment=c['deployment'],strategy=c['strategy'],policy_contract_sha256=protocol.digest(c))
def cid(job):return protocol.digest(symbolic(job))


def supplemental_design(path):
    """Declare endpoints/counts before new native development is inspected."""
    value=dict(schema='queue_refined_efficacy_design_v1',status='FROZEN_BEFORE_NATIVE_DEVELOPMENT',
      development=dict(loads=[1.85,1.89],seeds=[203,204],jobs_per_load=8,total_jobs=16,
       roles=['calibrated_spill','risk_matched_no_spill','legacy_latency','strong_other_deployment'],
       shortlist='Per load: highest partial-QTLP exact-primary calibrated candidate with spill in both CPU seeds if available, else any spill, else highest zero-spill; its same-mode and same-risk calibrated specialist controller; old latency policy at primary; highest different-deployment candidate. Model-domain ties use one exact max then smallest ID within1e-12.',
       objective='Native original six-term U for ranking; CPU QTLP is screening only; absent M/E remain UNKNOWN.',
       order='Load1.85 then1.89; seed203 four declared roles, seed204 reverse order.'),
      screening=dict(seeds=[213,214],training=False,native_result_reads=False,candidates_per_load=51),
      selection=dict(primary_endpoint='Within-residency specialist-to-generalist routing benefit',
       reserved_pair='Exact primary calibrated_spill and its same-mode and same-risk risk_matched_no_spill, both eligible on both development seeds. Other deployment winner is retained and reported separately; no causal routing-benefit claim from changed residency.',
       workload='Highest paired mean delta_U(spill minus matched no-spill) among fully eligible loads; exact-max1e-12 ties pick smaller text rate. No cross-load raw-U ranking.',
       no_eligible_pair='Stop efficacy extension and report unresolved rather than replace ineligible/unknown observations or retune.',
       no_actual_spill='Retain and report zero-spill policy outcome; never force a route to manufacture benefit.',
       overall_winner='Report within each workload across all four actual conditions; reserved route endpoint does not imply this pair beats an unconfirmed different deployment.'),
      reserved=dict(independent_validation=[205,206],final_confirmation=[207,208,209],total_jobs=10,
       order='Alternating calibrated_spill/control and control/calibrated_spill by whole-pair index; freeze both phase packets before either phase.',
       no_reselection=True,no_refit=True,no_retry=True),
      original_constraints_unchanged=True,physical_full_pool_optimality=False,
      interpretation='This supplement resolves endpoint/shortlist details left open in preregistration-v1. It is written before native203/204 or reserved205-209 execution.')
    protocol.write(path,value);return value


def _source_receipts(original):
    sources=copy.deepcopy(original['sources'])
    for path in HERE.glob('*.py'):sources[str(path.resolve())]=protocol.receipt(path)
    serial_path=ROOT/'tools/queue_refined_calibration_20261023/serial.py'
    sources[str(serial_path)]=protocol.receipt(serial_path)
    sources[str(Path(__file__).resolve())]=protocol.receipt(__file__)
    return sources


def prepare(output,screening,old_packet,models,design):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    old=read(old_packet);manifest=read(models);design_doc=read(design)
    if design_doc['development']['total_jobs']!=16:raise ValueError('Unexpected declared count')
    if not protocol.validate_packet(old)['valid']:raise ValueError('Old immutable templates changed')
    screen=read(Path(screening)/'report.json')
    if any(protocol.receipt(path)['sha256']!=expected for path,expected in screen['bindings'].items()):
        raise ValueError('CPU screening source/input changed before native preparation')
    sources=_source_receipts(old);shared_inputs=copy.deepcopy(old['inputs'])
    shared_inputs += [protocol.receipt(p) for p in [old_packet,models,design,Path(screening)/'report.json']]
    efficiency_declaration=Path(design).parent/'efficiency-preregistration-v1.json'
    if efficiency_declaration.exists():shared_inputs.append(protocol.receipt(efficiency_declaration))
    shared_inputs += [protocol.receipt(m['model_path']) for m in manifest['models']]
    shared_inputs += [dict(path=path,sha256=expected) for path,expected in screen['bindings'].items()]
    models_by_deployment={protocol.digest(read(m['model_path'])['deployment']):m for m in manifest['models']}
    from tools.queue_refined_20261023.refined_search import ScreeningStudy
    config_module=importlib.import_module(frozen_runtime().__package__.split('.runtime')[0]+'.config')
    packet_paths=[];checks=[]
    for rate in design_doc['development']['loads']:
        folder=output/f'text{rate:g}';folder.mkdir()
        cpu_dir=Path(screening)/f'text{rate:g}';cpu_report=read(cpu_dir/'report.json')
        choices=cpu_report['comparison']['native_shortlist']
        role_rows=[(role,choices[role]) for role in design_doc['development']['roles']]
        if any(row is None for _,row in role_rows):raise ValueError('Incomplete CPU shortlist')
        domain=read(cpu_dir/'domain.json')
        declared={protocol.digest(c) for c in domain['candidates']}
        if len(declared)!=design_doc['screening']['candidates_per_load']:raise ValueError('CPU candidate domain count changed')
        for role,row in role_rows:
            if protocol.digest(row['candidate'])!=row['candidate_id'] or row['candidate_id'] not in declared:
                raise ValueError('CPU shortlist candidate identity outside frozen domain: '+role)
        spill,control=choices['calibrated_spill']['candidate'],choices['risk_matched_no_spill']['candidate']
        if (spill['deployment']!=PRIMARY or control['deployment']!=PRIMARY or not spill['allow_spill']
            or control['allow_spill'] or spill['mode']!=control['mode'] or spill['risk_quantile']!=control['risk_quantile']):
            raise ValueError('Primary mechanism comparison is not same-residency/mode/risk')
        if choices['legacy_latency']['candidate']['deployment']!=PRIMARY or choices['legacy_latency']['candidate']['mode']!='legacy':
            raise ValueError('Wrong primary legacy control')
        if choices['strong_other_deployment']['candidate']['deployment']==PRIMARY:raise ValueError('Alternative deployment is not distinct')
        base=read(old['base_packet']['path']);base['config']['workload']['text']=rate
        base_path=folder/'base-packet.json';protocol.write(base_path,base);base_receipt=protocol.receipt(base_path)
        config=copy.deepcopy(base['config']);config['catalog']['performance_mode']='task_replica_proxy';config['catalog']['memory_mode']='standalone_proxy'
        cpu_study=ScreeningStudy(base,manifest['models'],rate,seeds=design_doc['screening']['seeds'])
        inputs=shared_inputs+[base_receipt,protocol.receipt(cpu_dir/'report.json'),protocol.receipt(cpu_dir/'domain.json')]
        jobs=[]
        for index,seed in enumerate(design_doc['development']['seeds']):
            for role,row in (role_rows if index==0 else list(reversed(role_rows))):
                candidate=row['candidate'];x=candidate['deployment']
                template=next(j for j in old['jobs'] if j['policy_contract']['deployment']==x)
                previous=template['policy_contract'];legacy=candidate['mode']=='legacy'
                cpu_entry=cpu_study.bylabel[candidate['label']]
                if cpu_entry['reference_routing']!={'text':{'text':rate},**previous['background_routing']}:
                    raise ValueError('CPU/native background route mapping differs')
                if legacy and cpu_entry['legacy_service_ms']!=previous['service_ms']:
                    raise ValueError('CPU/native legacy estimates differ')
                if cpu_entry['quality_loss']!=previous['quality_loss']:
                    raise ValueError('CPU/native aggregate quality mapping differs')
                # Pure-specialist x has no non-specialist text route; its
                # allow_spill=True/False policies are route-equivalent.
                can_spill=candidate['allow_spill'] and 'generalist_2' in x
                if candidate['allow_spill'] and not can_spill:
                    text_families={spec['family'] for spec in cpu_entry['aliases'].values() if 'text' in spec['supported_tasks']}
                    if text_families!={'text'}:raise ValueError('Unproved no-generalist policy equivalence')
                strategy=('queue_latency' if legacy and can_spill else ('queue_quality' if can_spill else 'queue_specialist_only'))
                guard=dict(schema='queue_aware_token_bucket_v1',task='text',family='generalist_2',rate_rps=rate/8,burst=1.) if can_spill else None
                contract=protocol.make_policy_contract(x,previous['service_ms'],previous['quality_loss'],config['weights'],config['latency_reference_ms'],
                    strategy=strategy,models=config['catalog']['models'],text_rate=rate,
                    background_routing=previous['background_routing'],guard=guard,sources={p:r['sha256'] for p,r in sources.items()})
                if not legacy:
                    model_spec=models_by_deployment[protocol.digest(x)]
                    if protocol.receipt(model_spec['model_path'])['sha256']!=candidate['model_sha256']:raise ValueError('CPU/native model identity differs')
                    contract['refinement']=dict(schema='queue_refined_policy_v1',model=protocol.receipt(model_spec['model_path']),
                        mode=candidate['mode'],margin_ms=candidate['margin_ms'],risk_quantile=candidate['risk_quantile'],
                        concurrency_context=True,allow_spill=can_spill,horizon_seconds=600,energy_proxy_joules_per_service_ms=None)
                    contract['local_objective']='Calibrated causal conservative or fixed-denominator incremental QTLP proxy; full six-term native U evaluated separately.'
                protocol.validate_contract(contract)
                identity=dict(schema='queue_refined_native_job_v1',strategy=strategy,seed=seed,phase='development',
                    policy_contract_sha256=protocol.digest(contract),base_packet=base_receipt,
                    cpu_candidate_id=row['candidate_id'],cpu_candidate=candidate,role=role,workload=config['workload'],
                    native_semantic_mapping=('Only text family supports text; disabling absent spill routes is equivalent; legacy Q adds same family constant to both aliases.' if candidate['allow_spill'] and not can_spill else 'Exact mode, risk, model, margins and allowed routes.'),
                    horizon_seconds=600,warmup_seconds=30,arrival_process='poisson',measurement_claim=CLAIMS['development'])
                job=dict(id=protocol.digest(identity),label=f'{role}-seed{seed}',strategy=strategy,seed=seed,phase='development',role=role,
                    identity=identity,base_packet=base_receipt,policy_contract=contract,nominal_plan=protocol.nominal_plan(contract))
                seed_config=copy.deepcopy(config);seed_config['random_seed']=seed;seed_config['horizon_seconds']=600
                env=admission.capacity_envelope(config_module.SystemConfig(**seed_config),job['nominal_plan'],contract)
                ok,issues=admission.source_checks(contract);check=dict(label=job['label'],load=rate,capacity=env,sources_verified=ok,source_issues=issues)
                checks.append(check)
                if not env['constraints_verified'] or not ok:raise ValueError('Prospective shortlist admission failed: '+str(check))
                job['capacity_evidence']=env;jobs.append(job)
        packet=dict(schema='queue_refined_efficacy_packet_v1',status='PREPARED_NOT_EXECUTED',phase='development',
            source_package=PACKAGE,base_packet=base_receipt,inputs=list({r['path']:r for r in inputs}.values()),sources=sources,
            jobs=jobs,seeds=design_doc['development']['seeds'],worker_binding=old['worker_binding'],
            selection_design=protocol.receipt(design),screening=protocol.receipt(cpu_dir/'report.json'),GPU_started=False,no_retries=True)
        if not protocol.validate_packet(packet)['valid']:raise ValueError('Prepared packet invalid')
        path=folder/'experiment-packet.json';protocol.write(path,packet);packet_paths.append(protocol.receipt(path))
    plan=dict(schema='queue_refined_execution_plan_v1',development_packets=packet_paths,design=protocol.receipt(design),
        sources=sources,inputs=list({r['path']:r for r in shared_inputs}.values()),GPU_started=False,no_retries=True)
    protocol.write(output/'execution-plan.json',plan);protocol.write(output/'cpu-check.json',dict(valid=True,jobs=checks,GPU_started=False))
    return plan


def measured_audit(output,report):
    table=read(output/'analysis/measured-search-table.json')
    dfs=measured_table_search(table,'plain_dfs');bounded=measured_table_search(table,'bounded')
    if dfs['winner_id']!=bounded['winner_id'] or dfs['objective']!=bounded['objective']:raise ValueError('Measured DFS mismatch')
    winner=report['winner']
    if (winner['candidate_id'] if winner else None)!=dfs['winner_id']:raise ValueError('Native ranking/DFS mismatch')
    protocol.write(output/'analysis/measured-dfs-comparison.json',dict(plain_dfs=dfs,bounded=bounded,same_winner_and_value=True,
        scope='Existing native measurement table lookup; wall time is not GPU search efficiency.'))


def freeze_selection(packets,reports,output,design):
    choices=[]
    for packet,report in zip(packets,reports):
        templates={j['role']:j for j in packet['jobs']}
        rows={r['candidate_id']:r for r in report['ranking']}
        spill=rows[cid(templates['calibrated_spill'])];control=rows[cid(templates['risk_matched_no_spill'])]
        known=spill['complete_eligible_all_declared_seeds'] and control['complete_eligible_all_declared_seeds']
        choices.append(dict(load=templates['calibrated_spill']['policy_contract']['text_rate'],eligible_pair=known,
            mean_delta_U=spill['mean_U']-control['mean_U'] if known else None,
            spill_candidate=spill,control_candidate=control,overall_development_winner=report['winner']))
    eligible=[c for c in choices if c['eligible_pair']]
    if not eligible:
        result=dict(status='UNRESOLVED_NO_ELIGIBLE_PRIMARY_PAIR',choices=choices,no_reselection=True)
        protocol.write(output/'selection.json',result);return result,[]
    maximum=max(c['mean_delta_U'] for c in eligible)
    selected=min((c for c in eligible if maximum-c['mean_delta_U']<=1e-12),key=lambda c:c['load'])
    packet=next(p for p in packets if p['jobs'][0]['policy_contract']['text_rate']==selected['load'])
    templates={j['role']:j for j in packet['jobs']};phase_packets=[];pair_index=0
    for phase,seeds in [('independent_validation',design['reserved']['independent_validation']),('final_confirmation',design['reserved']['final_confirmation'])]:
        jobs=[]
        for seed in seeds:
            roles=['calibrated_spill','risk_matched_no_spill'];roles=roles if pair_index%2==0 else list(reversed(roles));pair_index+=1
            for role in roles:
                job=copy.deepcopy(templates[role]);job.update(seed=seed,phase=phase,label=f'{role}-seed{seed}')
                job['identity'].update(seed=seed,phase=phase,measurement_claim=CLAIMS[phase],
                    development_selection_digest=protocol.digest(selected),no_reselection=True)
                job['id']=protocol.digest(job['identity']);jobs.append(job)
        p=copy.deepcopy(packet);p.update(jobs=jobs,seeds=seeds,phase=phase,status='SELECTION_FROZEN_BEFORE_ANY_RESERVED_EXECUTION')
        path=output/(phase+'-packet.json');protocol.write(path,p);phase_packets.append(protocol.receipt(path))
    result=dict(status='FROZEN',choices=choices,selected=selected,exact_max_development_delta=maximum,
        phase_packets=phase_packets,selection_scope='Within exact primary residency and risk; overall deployment winners reported separately.',no_reselection=True)
    protocol.write(output/'selection.json',result);return result,phase_packets


def execute(plan_path,output):
    """Run all authorized phases to terminal; frozen sources fail closed."""
    from tools.queue_refined_calibration_20261023.serial import execute as serial_execute
    plan=read(plan_path);design=read(plan['design']['path']);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    bindings=[protocol.receipt(plan_path),plan['design'],*plan['sources'].values(),*plan['inputs'],*plan['development_packets']]
    state=dict(status='RUNNING',PID=os.getpid(),started_unix_ns=time.time_ns(),stages=[],bindings=bindings,no_retries=True)
    def pulse():state['heartbeat_unix_ns']=time.time_ns();publish(output/'progress.json',state)
    def verify():
        if any(protocol.receipt(r['path'])!=r for r in bindings):raise ValueError('Workflow bound input changed')
    def stage(bound,label):
        verify()
        if protocol.receipt(bound['path'])!=bound:raise ValueError('Phase packet changed')
        state['phase']=label;target=output/label;item=dict(label=label,packet=bound,progress=str(target/'progress.json'));state['stages'].append(item);pulse()
        code=serial_execute(bound['path'],target,PACKAGE)
        item['terminal_progress']=protocol.receipt(target/'progress.json');pulse()
        if code:raise RuntimeError('Native batch stopped: '+str(read(target/'progress.json').get('error')))
        report=read(target/'analysis/report.json');measured_audit(target,report);item.update(status='COMPLETE_AUDITED',report=protocol.receipt(target/'analysis/report.json'));pulse();return report
    pulse()
    try:
        packets=[read(b['path']) for b in plan['development_packets']]
        reports=[stage(b,f'development-text{p["jobs"][0]["policy_contract"]["text_rate"]:g}') for b,p in zip(plan['development_packets'],packets)]
        verify();selection,reserved=freeze_selection(packets,reports,output,design);state['selection']=protocol.receipt(output/'selection.json');pulse()
        if not reserved:
            state.update(status='COMPLETE_WITH_UNRESOLVED_EFFICACY',phase='COMPLETE',reason=selection['status']);return 0
        phase_reports=[stage(b,read(b['path'])['phase']) for b in reserved]
        raw_paths=[j['raw']['path'] for item in state['stages'][2:] for j in read(item['progress'])['jobs']]
        combined=analyze(raw_paths,design['reserved']['independent_validation']+design['reserved']['final_confirmation'],output/'all-independent-pairs-analysis')
        table=read(output/'all-independent-pairs-analysis/measured-search-table.json')
        dfs=measured_table_search(table,'plain_dfs');bounded=measured_table_search(table,'bounded')
        if dfs['winner_id']!=bounded['winner_id']:raise ValueError('Reserved combined DFS mismatch')
        protocol.write(output/'all-independent-pairs-analysis/measured-dfs-comparison.json',dict(plain_dfs=dfs,bounded=bounded,same_winner=True,table_lookup_only=True))
        result=dict(status='COMPLETE',selection=selection,development=[r['ranking'] for r in reports],
            independent_validation=phase_reports[0],confirmation=phase_reports[1],combined=combined,
            no_reselection=True,no_refit_on_reserved=True,physical_full_pool_optimality=False)
        protocol.write(output/'result.json',result);state.update(status='COMPLETE',phase='COMPLETE',result=protocol.receipt(output/'result.json'));return 0
    except (Exception,KeyboardInterrupt) as exc:
        state.update(status='STOPPED_FOR_DIAGNOSIS_NO_RETRY',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());return 1
    finally:state['finished_unix_ns']=time.time_ns();pulse()


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('design');d.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',type=Path,required=True);a.add_argument('--screening',type=Path,required=True);a.add_argument('--old-packet',type=Path,required=True);a.add_argument('--models',type=Path,required=True);a.add_argument('--design',type=Path,required=True)
    e=sub.add_parser('execute');e.add_argument('--plan',type=Path,required=True);e.add_argument('--output',type=Path,required=True);e.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.command=='design':print(json.dumps(supplemental_design(args.output),ensure_ascii=False))
    elif args.command=='prepare':print(json.dumps(prepare(args.output,args.screening,args.old_packet,args.models,args.design),ensure_ascii=False))
    else:
        if not args.execute:p.error('--execute required')
        raise SystemExit(execute(args.plan,args.output))
