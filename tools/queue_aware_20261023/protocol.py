"""CPU preparation and explicit identities for experimental native routing.

Policy tables contain frozen estimates/aggregate losses only. Native answers and
references remain in the evaluator. Every phase constructs a fresh dispatcher.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.spillover_capacity_20261022.exposure import frozen_api,SOURCE
from tools.queue_aware_20261023.policy import QueueAwarePolicy

STRATEGIES=['no_split_swrr','fixed_1of32_swrr','queue_latency','queue_quality','queue_specialist_only']


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def receipt(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def make_policy_contract(deployment,service_ms,quality_loss,weights,latency_refs_ms,*,strategy,
                         models,text_rate=1.85,background_routing=None,overdue_remaining_ms=5.,
                         sources=None,service_replica_states=None,guard=None):
    if strategy not in STRATEGIES:raise ValueError('Unknown declared strategy')
    if any(type(n) is not int or n<=0 for n in deployment.values()):raise ValueError('Positive resident replica counts required')
    states=service_replica_states or deployment
    if states!=deployment:raise ValueError('Service estimate replica-state identity differs from resident deployment')
    aliases={f'{m}#{i}':dict(family=m,supported_tasks=list(models[m]['supported_tasks'])) for m,n in sorted(deployment.items()) for i in range(n)}
    background_routing=background_routing if background_routing is not None else {'image':{'generalist_2':.01},'asr':{'generalist_2':.01},'ocr':{'ocr':.01}}
    mode='quality_aware' if strategy in {'queue_quality','queue_specialist_only'} else 'latency'
    contract=dict(schema='gsdeploy.causal_native_policy.v1',strategy=strategy,mode=mode,deployment=copy.deepcopy(deployment),
        aliases=aliases,service_ms=copy.deepcopy(service_ms),quality_loss=copy.deepcopy(quality_loss),
        weights=copy.deepcopy(weights),latency_refs_ms=copy.deepcopy(latency_refs_ms),
        text_rate=text_rate,background_routing=copy.deepcopy(background_routing),overdue_remaining_ms=overdue_remaining_ms,
        service_replica_states=copy.deepcopy(states),source_fingerprints=copy.deepcopy(sources or {}),guard=copy.deepcopy(guard),
        service_scope='Frozen pre-dispatch per-input difficulty estimates, not future worker observations.',
        quality_scope='Frozen family/task aggregate historical loss only; no input reference, answer or sample-specific quality.',
        phase_state='Fresh dispatcher/router/guard per warmup and measurement; no cross-phase outstanding or completion state.',
        static_control_guard_scope='Static SWRR baselines retain original routing; dynamic guard applies only to queue policies.',
        local_objective='L+Q surrogate only; original six-term measured evaluation is separate.',absolute_quality_gate=False)
    validate_contract(contract);return contract


def validate_contract(contract):
    if contract['strategy'] not in STRATEGIES:raise ValueError('Unknown strategy')
    if contract['service_replica_states']!=contract['deployment']:raise ValueError('Replica estimate identity mismatch')
    expected={f'{m}#{i}' for m,n in contract['deployment'].items() for i in range(n)}
    if set(contract['aliases'])!=expected:raise ValueError('Resident aliases mismatch')
    mode='quality_aware' if contract['strategy'] in {'queue_quality','queue_specialist_only'} else 'latency'
    if contract['mode']!=mode:raise ValueError('Strategy/mode mismatch')
    QueueAwarePolicy(contract['aliases'],contract['service_ms'],contract['quality_loss'],contract['weights'],
        contract['latency_refs_ms'],mode=mode,overdue_remaining_ms=contract['overdue_remaining_ms'])
    for task,row in contract['background_routing'].items():
        if task=='text':raise ValueError('Text may not be hidden in background routing')
        for family,q in row.items():
            if family not in contract['deployment'] or q<=0:raise ValueError('Invalid fixed background route')
            for alias,spec in contract['aliases'].items():
                if spec['family']==family and task not in spec['supported_tasks']:raise ValueError('Unsupported fixed background')
    for path,expected in contract.get('source_fingerprints',{}).items():
        if not isinstance(expected,str) or receipt(path)['sha256']!=expected:raise ValueError('Policy source changed')
    return True


def nominal_plan(contract):
    share=1/32 if contract['strategy']=='fixed_1of32_swrr' else 0.
    text={'text':contract['text_rate']*(1-share)}
    if share:text['generalist_2']=contract['text_rate']*share
    return dict(deployment=copy.deepcopy(contract['deployment']),routing={'text':text,**copy.deepcopy(contract['background_routing'])})


class Dispatcher:
    """Full resident snapshot, fixed backgrounds, independent text policy."""
    def __init__(self,contract):
        validate_contract(contract);self.contract=contract;self.aliases=contract['aliases'];api=frozen_api()
        plan=nominal_plan(contract);_,routes=api['trace'].expand_routes(plan['deployment'],plan['routing']);self.router=api['routing'].Router(routes)
        eligible={a:s for a,s in self.aliases.items() if s['family']=='text'} if contract['strategy']=='queue_specialist_only' else self.aliases
        self.policy=QueueAwarePolicy(eligible,contract['service_ms'],contract['quality_loss'],contract['weights'],contract['latency_refs_ms'],
            mode=contract['mode'],overdue_remaining_ms=contract['overdue_remaining_ms'])
        if contract.get('guard') and contract['strategy'].startswith('queue_'):
            # Root owns guard implementation; its constructor is explicitly
            # resolved here and bound into every job's contract sources.
            from tools.queue_aware_20261023.guarded_policy import RateLimitedPolicy
            if contract['strategy']!='queue_specialist_only':self.policy=RateLimitedPolicy(self.policy,contract['guard'])

    def choose(self,task,pool_id,now_ns,pending_snapshot):
        strategy=self.contract['strategy']
        if task!='text' or strategy in {'no_split_swrr','fixed_1of32_swrr'}:
            alias=self.router.choose(task)
            return dict(alias=alias,status='FROZEN_SWRR_BACKGROUND' if task!='text' else 'FROZEN_SWRR',
                trigger='STATIC_ROUTE',strategy=strategy,task=task,pool_id=pool_id,now_ns=now_ns,audit=[])
        result=self.policy.choose(task,pool_id,now_ns,pending_snapshot)
        result['strategy']=strategy;return result


def make_dispatcher(policy_contract):return Dispatcher(policy_contract)


def prepare(output,base_packet,simulation_contract,*,seeds=(113,114),guard=None,capacity_evidence=None):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    base=json.loads(Path(base_packet).read_text());sim=json.loads(Path(simulation_contract).read_text())
    if sim['deployment']!={'text':2,'generalist_2':1,'ocr':1}:raise ValueError('First native stage requires exact text2/G2/OCR resident estimate table')
    if list(seeds)!=[113,114]:raise ValueError('First native development seeds are113/114; reserved seeds untouched')
    source_files=[Path(__file__),Path(__file__).with_name('native_runner.py'),Path(__file__).with_name('runtime_window.py'),
        Path(__file__).with_name('policy.py'),Path(__file__).with_name('admission.py'),Path(__file__).with_name('evaluation.py'),
        ROOT/'tools/spillover_capacity_20261022/exposure.py']
    if guard:source_files.append(Path(__file__).with_name('guarded_policy.py'))
    source_files += list(SOURCE.rglob('*.py'))
    sources={str(p.resolve()):receipt(p) for p in source_files}
    inputs=[receipt(base_packet),receipt(simulation_contract),*base['bindings']]
    for bound in inputs:
        if receipt(bound['path'])!=bound:raise ValueError('Source packet binding changed')
    jobs=[]
    for seed in seeds:
        for strategy in STRATEGIES if seed==seeds[0] else reversed(STRATEGIES):
            contract=make_policy_contract(sim['deployment'],sim['service_ms'],sim['aggregate_quality_loss'],base['config']['weights'],
                base['config']['latency_reference_ms'],strategy=strategy,models=base['config']['catalog']['models'],
                sources={path:value['sha256'] for path,value in sources.items()},guard=guard)
            identity=dict(schema='gsdeploy.causal_native_job_identity.v1',strategy=strategy,seed=seed,policy_contract_sha256=digest(contract),
                base_packet=receipt(base_packet),simulation_contract=receipt(simulation_contract),horizon_seconds=600,warmup_seconds=30,
                arrival_process='poisson',deployment=sim['deployment'],measurement_claim='NEW_DEVELOPMENT_NATIVE_OBSERVATION_NOT_INDEPENDENT_CONFIRMATION')
            jobs.append(dict(id=digest(identity),label=f'{strategy}-seed{seed}',strategy=strategy,seed=seed,identity=identity,
                policy_contract=contract,nominal_plan=nominal_plan(contract),capacity_evidence=copy.deepcopy(capacity_evidence)))
    packet=dict(schema='gsdeploy.causal_native_packet.v1',status='PREPARED_NOT_EXECUTED',base_packet=receipt(base_packet),
        simulation_contract=receipt(simulation_contract),inputs=inputs,sources=sources,jobs=jobs,seeds=list(seeds),
        worker_binding='Frozen bind_native_job applied at explicit execute; all potential policy paths independently admitted.',
        no_retries=True,GPU_started=False)
    output.mkdir(parents=True,exist_ok=False);write(output/'experiment-packet.json',packet)
    print(json.dumps(dict(packet=receipt(output/'experiment-packet.json'),jobs=len(jobs),GPU_started=False)))
    return packet


def validate_packet(packet):
    changed=[]
    for source in [*packet['inputs'],*packet['sources'].values()]:
        if receipt(source['path'])!=source:changed.append(source['path'])
    for job in packet['jobs']:
        if digest(job['identity'])!=job['id'] or digest(job['policy_contract'])!=job['identity']['policy_contract_sha256']:
            changed.append(job['label'])
        if job['nominal_plan']!=nominal_plan(job['policy_contract']):changed.append(job['label']+':nominal_plan')
        validate_contract(job['policy_contract'])
    return dict(valid=not changed,changed=changed,GPU_started=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--output',type=Path,required=True);prep.add_argument('--base-packet',type=Path,required=True)
    prep.add_argument('--simulation-contract',type=Path,required=True);prep.add_argument('--guard',type=Path);prep.add_argument('--capacity-evidence',type=Path)
    check=sub.add_parser('check');check.add_argument('--packet',type=Path,required=True);a=p.parse_args()
    if a.mode=='prepare':prepare(a.output,a.base_packet,a.simulation_contract,guard=json.loads(a.guard.read_text()) if a.guard else None,
        capacity_evidence=json.loads(a.capacity_evidence.read_text()) if a.capacity_evidence else None)
    else:print(json.dumps(validate_packet(json.loads(a.packet.read_text()))))
