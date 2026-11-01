"""Prospective five-term native experiment; sequential pairs, no old evidence edits."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import importlib
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

from tools.queue_five_term_20261023 import protocol
from tools.queue_five_term_20261023.native_runner import execute_job, frozen_runtime
from tools.queue_five_term_20261023.policy import FIXED_WEIGHTS, OBJECTIVE_SCHEMA
from tools.queue_refined_20261023 import admission

ROOT = Path(__file__).resolve().parents[2]
ROLES = ('coupled_spill', 'matched_no_spill', 'legacy_latency', 'legacy_no_spill')
PRIMARY = dict(text=2, generalist_2=1, ocr=1)


def read(path):
    return json.loads(Path(path).read_text())


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def design(path):
    value = dict(schema='gsdeploy.five_term_prospective_design.v1', objective_schema=OBJECTIVE_SCHEMA,
                 weights=dict(FIXED_WEIGHTS), P_evaluation_role='diagnostic_only',
                 model_pool=['generalist_1','generalist_2','generalist_3','text','image','ocr','asr'],
                 resident_deployment=PRIMARY, workload=dict(text=1.85,image=.01,ocr=.01,asr=.01),
                 horizon_seconds=600, warmup_seconds=30, drain_seconds=300, arrival_process='poisson',
                 development=dict(seeds=[401,402],roles=list(ROLES),runs=8,
                                  order=[['coupled_spill',401],['matched_no_spill',401],
                                         ['matched_no_spill',402],['coupled_spill',402],
                                         ['legacy_latency',401],['legacy_no_spill',401],
                                         ['legacy_no_spill',402],['legacy_latency',402]],
                                  primary='New coupled five-term online policy versus its exact allow_spill=False control',
                                  baseline='Same legacy latency scoring in both arms; only text-generalist candidate permission changes'),
                 online_params=dict(scenarios=['mean','p90'],utility_margin=0.,cooldown_ms=30000.),
                 cpu_selection='PRESPECIFIED_DEFAULT_WHEN_FULL_WORKLOAD_CPU_UNRESOLVED; text-only winner never selects native policy',
                 confirmation=dict(seeds=[421,422,423,424,425,426],roles=['coupled_spill','matched_no_spill'],runs=12,
                                   minimum_development_delta=.0001,
                                   gate='All eight development runs complete with valid evidence; each primary pair eligible, actual spill>0 and delta_U>0.0001',
                                   epsilon_scope='Exploratory practical lower bound; not a measurement-noise confidence limit',
                                   selection='Fixed coupled primary only; no substitution from legacy results',
                                   endpoint='All six fixed confirmation pairs eligible with actual spill>0 and delta_U>0; report each pair and uncertainty',
                                   no_refit=True,no_reselection=True),
                 no_retries=True, stop_on_infrastructure_failure=True, quality_admission='No absolute quality threshold',
                 original_stability_unchanged=True, original_rho_threshold=.8,
                 energy='Original window energy plus startup amortization; full-cohort drain energy remains diagnostic',
                 legacy_online_scope='Latency heuristic baseline, not complete five-term online optimizer',
                 old_reserved_seeds_untouched=[305,306,307,308,309,310],
                 scope='Fixed resident deployment and finite strategy domain from frozen seven-model source pool; no physical full-pool optimality claim')
    protocol.write(path,value)
    return value


def prepare(output, old_packet, resources, design_path):
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    old,plan = read(old_packet),read(design_path)
    base = read(old['base_packet']['path'])
    if base['config']['workload']!=plan['workload'] or plan['weights']!=FIXED_WEIGHTS:
        raise ValueError('Prespecified workload/objective changed')
    base['config']['weights']=dict(FIXED_WEIGHTS)
    base['config']['catalog'].update(performance_mode='task_replica_proxy',memory_mode='standalone_proxy')
    base['prospective_objective_schema']=OBJECTIVE_SCHEMA
    base['parent_base_receipt']=old['base_packet']
    protocol.write(output/'base-packet.json',base)
    base_ref=protocol.receipt(output/'base-packet.json')
    sources=copy.deepcopy(old['sources'])
    for path in Path(__file__).parent.glob('*.py'):
        sources[str(path.resolve())]=protocol.receipt(path)
    fingerprints={p:r['sha256'] for p,r in sources.items()}
    templates={j['role']:j['policy_contract'] for j in old['jobs']}
    contracts=dict(
        coupled_spill=protocol.make_contract(templates['coupled_spill'],resources,allow_spill=True,
                                              params=plan['online_params'],sources=fingerprints),
        matched_no_spill=protocol.make_contract(templates['coupled_spill'],resources,allow_spill=False,
                                               params=plan['online_params'],sources=fingerprints),
        legacy_latency=protocol.make_legacy_contract(templates['legacy_latency'],allow_spill=True,sources=fingerprints),
        legacy_no_spill=protocol.make_legacy_contract(templates['legacy_latency'],allow_spill=False,sources=fingerprints))
    configuration=importlib.import_module(frozen_runtime().__package__.split('.runtime')[0]+'.config')
    jobs=[]
    for role,seed in plan['development']['order']:
        contract=contracts[role]
        identity=dict(schema='gsdeploy.five_term_native_job.v1',phase='development',seed=seed,role=role,
                      objective_schema=OBJECTIVE_SCHEMA,weights=dict(FIXED_WEIGHTS),base_packet=base_ref,
                      design=protocol.receipt(design_path),policy_contract_sha256=protocol.digest(contract),
                      horizon_seconds=600,warmup_seconds=30,workload=plan['workload'],arrival_process='poisson',
                      measurement_claim='FRESH_FIVE_TERM_DEVELOPMENT_NOT_INDEPENDENT_CONFIRMATION')
        nominal=protocol.nominal_plan(contract)
        if nominal['deployment']!=PRIMARY:
            raise ValueError('Prespecified resident deployment changed')
        config=copy.deepcopy(base['config']);config['random_seed']=seed
        capacity=admission.capacity_envelope(configuration.SystemConfig(**config),nominal,contract)
        if not capacity['constraints_verified']:
            raise ValueError('Prior capacity envelope failed: '+repr(capacity))
        jobs.append(dict(id=protocol.digest(identity),label=f'{role}-seed{seed}',seed=seed,phase='development',
                         role=role,identity=identity,policy_contract=contract,nominal_plan=nominal,capacity_evidence=capacity))
    inputs=[*old['inputs'],protocol.receipt(old_packet),old['base_packet'],base_ref,
            protocol.receipt(resources),protocol.receipt(design_path)]
    packet=dict(schema='gsdeploy.five_term_native_packet.v1',phase='development',base_packet=base_ref,
                inputs=list({r['path']:r for r in inputs}.values()),sources=sources,jobs=jobs,seeds=[401,402],
                design=protocol.receipt(design_path),no_retries=True,GPU_started=False)
    check=protocol.validate_packet(packet)
    if not check['valid']: raise ValueError(check)
    protocol.write(output/'experiment-packet.json',packet)
    return packet


def summarize(job,raw,receipt):
    utility=raw.get('evaluation',{}).get('utility',{})
    routing={};gate_counts={}
    for row in raw.get('request_records',[]):
        family=raw.get('replica_to_family',{}).get(row['model'])
        key=row['task']+'->'+str(family);routing[key]=routing.get(key,0)+1
        for gate in row.get('routing_decision',{}).get('decision',{}).get('gates',[]):
            gate_counts[gate]=gate_counts.get(gate,0)+1
    return dict(label=job['label'],seed=job['seed'],role=job['role'],raw=receipt,status=raw.get('status'),
                U=utility.get('U'),eligible=utility.get('eligible',False),utility=utility,routing_counts=routing,
                actual_text_spills=routing.get('text->generalist_2',0),gates=gate_counts,
                causal_verified=raw.get('evaluation',{}).get('dynamic_audit',{}).get('verified'),
                stability=raw.get('evaluation',{}).get('stability'),
                cohort_energy=raw.get('cohort_energy_diagnostic'),system_metrics=raw.get('system_metrics'),
                per_task_metrics=raw.get('per_task_metrics'),config_weights=raw.get('config',{}).get('weights'),
                contract_weights=raw.get('policy_contract',{}).get('weights'))


def analyze(packet,observations):
    pairs=[]
    for on,off in [('coupled_spill','matched_no_spill'),('legacy_latency','legacy_no_spill')]:
        if not any(j['role']==on for j in packet['jobs']):continue
        for seed in packet['seeds']:
            a,b=observations.get(f'{on}-seed{seed}'),observations.get(f'{off}-seed{seed}')
            complete=bool(a and b and a['eligible'] and b['eligible'] and a['U'] is not None and b['U'] is not None)
            pairs.append(dict(strategy=on,seed=seed,complete_eligible=complete,
                              delta_U=a['U']-b['U'] if complete else None,
                              actual_text_spills=a['actual_text_spills'] if a else None))
    ranks=[]
    for role in sorted({j['role'] for j in packet['jobs']}):
        rows=[observations.get(f'{role}-seed{s}') for s in packet['seeds']]
        complete=all(r and r['eligible'] and r['U'] is not None for r in rows)
        ranks.append(dict(role=role,complete_eligible=complete,mean_U=statistics.fmean(r['U'] for r in rows) if complete else None))
    ranks.sort(key=lambda r:(r['mean_U'] is None,-(r['mean_U'] or 0),r['role']))
    primary=[p for p in pairs if p['strategy']=='coupled_spill']
    all_observed=len(observations)==len(packet['jobs'])
    all_valid=all_observed and all(r['status']=='completed' and r['eligible'] and r['causal_verified'] for r in observations.values())
    gate=packet['phase']=='development' and all_valid and len(primary)==2 and all(
        p['complete_eligible'] and p['delta_U']>.0001 and p['actual_text_spills']>0 for p in primary)
    confirmed=packet['phase']=='independent_confirmation' and all_valid and len(primary)==6 and all(
        p['complete_eligible'] and p['delta_U']>0 and p['actual_text_spills']>0 for p in primary)
    return dict(schema='gsdeploy.five_term_native_analysis.v1',phase=packet['phase'],weights=dict(FIXED_WEIGHTS),
                observations=list(observations.values()),paired=pairs,ranking=ranks,all_jobs_complete_valid=all_valid,
                confirmation_gate_passed=gate,all_six_confirmation_pairs_positive=confirmed,
                primary_fixed='coupled_spill',physical_seven_pool_optimality=False,
                scope='New-objective native finite strategy domain; primary causal effect requires own matched no-spill')


def prepare_confirmation(development_packet,development_report,output):
    packet,report=read(development_packet),read(development_report)
    if not protocol.validate_packet(packet)['valid'] or not report['confirmation_gate_passed']:
        raise ValueError('Frozen development gate not passed')
    if report!=load_analysis(packet,Path(development_report).parent):
        raise ValueError('Development report not reproducible from raw files')
    plan=read(packet['design']['path']);templates={j['role']:j for j in packet['jobs']};jobs=[]
    for index,seed in enumerate(plan['confirmation']['seeds']):
        roles=plan['confirmation']['roles'] if index%2==0 else list(reversed(plan['confirmation']['roles']))
        for role in roles:
            job=copy.deepcopy(templates[role]);identity=copy.deepcopy(job['identity'])
            identity.update(seed=seed,phase='independent_confirmation',selection=protocol.receipt(development_report),
                            measurement_claim='NEW_FIXED_FIVE_TERM_INDEPENDENT_CONFIRMATION')
            job.update(id=protocol.digest(identity),label=f'{role}-seed{seed}',seed=seed,
                       phase='independent_confirmation',identity=identity);jobs.append(job)
    packet.update(phase='independent_confirmation',seeds=plan['confirmation']['seeds'],jobs=jobs,
                  selection=protocol.receipt(development_report),
                  inputs=[*packet['inputs'],protocol.receipt(development_packet),protocol.receipt(development_report)])
    if not protocol.validate_packet(packet)['valid']:raise ValueError('Confirmation packet invalid')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    protocol.write(output/'experiment-packet.json',packet)
    return packet


def load_analysis(packet,output):
    observations={}
    for job in packet['jobs']:
        path=Path(output)/job['label']/'run.json'
        if path.exists():observations[job['label']]=summarize(job,read(path),protocol.receipt(path))
    return analyze(packet,observations)


def execute(packet_path,output):
    packet=read(packet_path)
    if not protocol.validate_packet(packet)['valid']:raise ValueError('Frozen sources/inputs changed')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);observations={}
    progress=dict(status='RUNNING',packet=protocol.receipt(packet_path),completed=[],current=None,started_at=timestamp())
    def status():
        (output/'progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
    status()
    for job in packet['jobs']:
        progress['current']=job['label'];status()
        print(json.dumps(dict(event='START',label=job['label'],at=timestamp())),flush=True)
        try:raw=execute_job(packet,job['id'],output/job['label'],execute=True)
        except Exception as exc:
            progress.update(status='STOPPED_FOR_DIAGNOSIS_NO_RETRY',reason=f'{type(exc).__name__}: {exc}');status();break
        row=summarize(job,raw,protocol.receipt(output/job['label']/'run.json'));observations[job['label']]=row
        progress['completed'].append({k:row[k] for k in ('label','raw','status','U','eligible','actual_text_spills')})
        print(json.dumps(dict(event='END',at=timestamp(),**progress['completed'][-1])),flush=True)
        protocol.write(output/(job['label']+'-interim-analysis.json'),analyze(packet,observations))
        status()
        if raw['status']!='completed' or not row['utility'].get('energy_verified') or not row['utility'].get('cohort_verified'):
            progress.update(status='STOPPED_FOR_DIAGNOSIS_NO_RETRY',reason='INCOMPLETE_RESOURCE_OR_CAUSAL_EVIDENCE');status();break
        del raw
    else:progress.update(status='COMPLETED',current=None);status()
    report=analyze(packet,observations)
    protocol.write(output/'analysis.json',report)
    protocol.write(output/'terminal.json',dict(progress,analysis=protocol.receipt(output/'analysis.json')))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    d=sub.add_parser('design');d.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('prepare')
    for name in ('output','old-packet','resources','design'):q.add_argument('--'+name,type=Path,required=True)
    c=sub.add_parser('confirmation')
    for name in ('development-packet','development-report','output'):c.add_argument('--'+name,type=Path,required=True)
    e=sub.add_parser('execute');e.add_argument('--packet',type=Path,required=True);e.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='design':result=design(a.output)
    elif a.mode=='prepare':result=prepare(a.output,a.old_packet,a.resources,a.design)
    elif a.mode=='confirmation':result=prepare_confirmation(a.development_packet,a.development_report,a.output)
    else:result=execute(a.packet,a.output)
    print(json.dumps(result if a.mode=='execute' else dict(status='PREPARED',mode=a.mode),ensure_ascii=False))
