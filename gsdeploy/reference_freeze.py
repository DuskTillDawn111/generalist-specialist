"""Freeze measured references from native development receipts, CPU only."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from . import normalization as norm
from .config import SystemConfig
from .energy import energy_key
from .io import save, dumps
from .routing import Router
from .trace import expand_routes, make_trace, trace_digest


def freeze_references(config, manifest, base):
    """Derive denominators from raw records; never accept supplied scalar values."""
    if (manifest.get('schema') != 'gsdeploy.normalization_measurements.v1'
            or manifest.get('split') != 'normalization_development'
            or not manifest.get('confirmation_seeds')):
        raise ValueError('Explicit development references and reserved confirmation seeds required')
    spec = copy.deepcopy(norm.contract(config.catalog))
    if spec is None or spec['status'] != 'pending':
        raise ValueError('Only pending v2 references can be frozen; create a new version to revise')
    if set(manifest.get('task_runs', {})) != set(config.workload):
        raise ValueError('Reference runs required for every task')
    hardware = manifest.get('gpu_compatibility')
    if hardware is not None:
        if (set(hardware) != {'policy', 'name'} or hardware['policy'] != 'same_model'
                or not isinstance(hardware['name'], str) or not hardware['name'].strip()):
            raise ValueError('GPU compatibility requires an explicit same_model name')
        spec['gpu_compatibility'] = copy.deepcopy(hardware)
    receipts, seeds, used_paths = [], set(), set()
    protocol = None

    def read_run(name, role):
        path = (Path(base)/name).resolve()
        if path in used_paths:
            raise ValueError('Each reference receipt must be an independent run')
        used_paths.add(path)
        raw = json.loads(path.read_text())
        w = raw.get('measurement_window', {})
        tr = raw.get('transport_config', {})
        memory = raw.get('memory_evidence', {})
        preflight = raw.get('preflight') or {}
        gpu = preflight.get('gpu', {})
        # UUID records the physical source. Same-model campaigns explicitly bind
        # the model and retain the configured memory feasibility requirement.
        gpu_matches = (gpu.get('name') == hardware['name']
                       and norm.finite(gpu.get('memory_mib'), True)
                       and gpu['memory_mib'] >= config.gpu_memory_mib) if hardware else (
                           gpu.get('uuid') == spec['measurement_context']['gpu_uuid'])
        if (raw.get('backend') != 'native' or raw.get('status') != 'completed'
                or raw.get('errors') or w.get('aborted')
                or not preflight.get('ready') or not preflight.get('gpu_checked')
                or not gpu_matches
                or preflight.get('sample_pool_sha256') != spec['input_identity']['input_pool_sha256']
                or not preflight.get('admission_evidence', {}).get('verified')
                or not memory.get('measured') or not memory.get('resource_valid')
                or w.get('drain_end_time_ns', 0) < w.get('end_time_ns', 1)):
            raise ValueError('Complete native run with matching GPU/input/admission/memory evidence required')
        seed = tr['random_seed']
        if seed in manifest['confirmation_seeds']:
            raise ValueError('Reference seed overlaps reserved confirmation seeds')
        rates = {t:q for t,q in tr['workload_rps'].items() if q>0}
        records = raw['request_records']
        serial=raw.get('isolated_reference_plan')
        if serial is not None:
            if (not role.startswith('isolated_e2e:') or tr['arrival_process']!='isolated_serial'
                    or set(serial)!={'schema','task','input_pool','pool_ids'}
                    or serial['schema']!='gsdeploy.isolated_reference_plan.v1'
                    or serial['task']!=role.split(':',1)[1] or set(rates)!={serial['task']}):
                raise ValueError('Serial reference is restricted to predeclared isolated task runs')
            pool_path=Path(serial['input_pool']['path'])
            pool_bytes=pool_path.read_bytes()
            if (hashlib.sha256(pool_bytes).hexdigest()!=spec['input_identity']['input_pool_sha256']
                    or serial['input_pool']['sha256']!=spec['input_identity']['input_pool_sha256']):
                raise ValueError('Serial reference input pool fingerprint mismatch')
            pool=[json.loads(line) for line in pool_bytes.decode().splitlines()]
            ids=[row['pool_id'] for row in pool if row['task_type']==serial['task']]
            if (not ids or len(ids)!=len(set(ids)) or serial['pool_ids']!=ids
                    or [r.get('pool_id') for r in records]!=ids):
                raise ValueError('Serial reference must cover the entire frozen task input order')
            for a,b in zip(records,records[1:]):
                ready=a.get('previous_handler_next_request_ready',{})
                if (ready.get('clock')!='host_perf_counter_ns'
                        or ready.get('boundary')!='previous_handler_next_request_ready'
                        or type(ready.get('perf_ns')) is not int
                        or not a['completion_time_ns']<=ready.get('received_time_ns',-1)<=b['scheduled_time_ns']
                        or not a['worker_response']['end_perf_ns']<=ready['perf_ns']<=ready['received_time_ns']):
                    raise ValueError('Serial reference requires observed predecessor completion and READY')
            trace=[((r['scheduled_time_ns']-w['start_time_ns'])/1e9,serial['task']) for r in records]
        else:
            trace = make_trace(rates,w['duration_seconds'],seed,tr['arrival_process'])
        aliases, routes = expand_routes(raw['plan']['deployment'],raw['plan']['routing'])
        router = Router(routes)
        if (not trace or len(records) != len(trace) or w.get('offered_requests') != len(trace)
                or w.get('trace_sha256') != trace_digest(trace)):
            raise ValueError('Reference must contain the entire declared offered cohort')
        for i, ((at, task), r) in enumerate(zip(trace,records)):
            if (r.get('request_id') != f'measurement:{i}' or r.get('task') != task
                    or r.get('model') != router.choose(task) or r.get('status') != 'completed'
                    or r.get('scheduled_time_ns') != w['start_time_ns']+round(at*1e9)
                    or not r['scheduled_time_ns'] <= r['actual_submit_time_ns'] <= r['completion_time_ns'] <= w['drain_end_time_ns']):
                raise ValueError('Invalid reference request identity/timing or unsuccessful delivery')
        key = energy_key(raw['plan']['deployment'],raw['plan']['routing'],rates,
                         w['duration_seconds'],seed,tr['arrival_process'],catalog=config.catalog,
                         resource_configuration=raw.get('resource_configuration'))
        if raw.get('energy_evidence',{}).get('profile_key') != key:
            raise ValueError('Reference checkpoint/input/protocol profile key mismatch')
        receipts.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             role=role,seed=seed,gpu=copy.deepcopy(gpu)))
        seeds.add(seed)
        return raw,rates

    for task, names in manifest['task_runs'].items():
        if not names:
            raise ValueError('Empty isolated reference runs: '+task)
        latencies=[]
        for name in names:
            raw,rates=read_run(name,'isolated_e2e:'+task)
            model=spec['tasks'][task]['reference_path']
            if (set(rates) != {task} or raw['plan']['deployment'] != {model:1}
                    or config.catalog['models'][model]['role'] != 'specialist'):
                raise ValueError('Isolated task reference must use its predeclared single specialist')
            previous=None
            for r in raw['request_records']:
                # No outstanding predecessor: this is an isolated end-to-end reference.
                if previous is not None and r['scheduled_time_ns'] < previous:
                    raise ValueError('Queued/overlapping reference requests cannot define isolated latency')
                previous=r['completion_time_ns']
                latencies.append((previous-r['scheduled_time_ns'])/1e6)
        mean=math.fsum(latencies)/len(latencies)
        spec['tasks'][task].update(reference_mean_e2e_ms=mean,deadline_ms=spec['deadline_multiplier']*mean)
        spec['tasks'][task]['provenance']['latency']='Native independent isolated development receipts; arithmetic mean e2e multiplied by fixed k.'
    names=manifest.get('episode_runs',[])
    if len(names)<2:
        raise ValueError('Repeated baseline episode runs required')
    energies=[]; episode_seeds=set()
    for name in names:
        raw,rates=read_run(name,'reference_episode')
        tr=raw['transport_config']; w=raw['measurement_window']
        if (rates != config.workload or w['duration_seconds'] != config.horizon_seconds
                or tr['arrival_process'] != config.arrival_process
                or raw['plan']['deployment'] != spec['reference_deployment']
                or any(raw['plan']['routing'][t] != {spec['tasks'][t]['reference_path']:q} for t,q in rates.items())
                or raw.get('resource_configuration') is not None):
            raise ValueError('Baseline episode differs from frozen workload, horizon, deployment or routing')
        seed=tr['random_seed']
        if seed in episode_seeds:
            raise ValueError('Independent baseline seeds required')
        episode_seeds.add(seed)
        current=dict(warmup_seconds=tr['warmup_seconds'],priming_policy=raw.get('priming_policy'),
                     drain_seconds=tr['drain_timeout_seconds'],arrival_process=tr['arrival_process'])
        if protocol is not None and current != protocol:
            raise ValueError('Reference episodes must share warmup, priming and drain protocol')
        protocol=current
        e=norm.measured_episode(raw['energy_evidence'],w)
        if (e.get('kind') != 'measured' or e.get('scope') != 'whole_gpu_board'
                or not e['episode_coverage_valid'] or not norm.finite(e['episode_joules'],True)):
            raise ValueError('Complete startup-through-drain board power telemetry required')
        from .evaluation import evaluate_run
        evaluated=evaluate_run(config,raw)
        if (evaluated['stability'].get('stability_state_v2') != 'steady'
                or not evaluated['utility'].get('constraints_verified')
                or not evaluated['utility'].get('quality_identity_verified')):
            raise ValueError('Reference baseline must be steady, feasible and identity-verified')
        energies.append(e['episode_joules'])
    spec.update(status='frozen',episode_reference_joules=math.fsum(energies)/len(energies),
        episode_protocol=protocol,reference_seeds=sorted(seeds),confirmation_seeds=manifest['confirmation_seeds'],
        reference_receipts=[*spec.get('reference_receipts',[]),*receipts])
    refs={t:r['deadline_ms'] for t,r in spec['tasks'].items()}
    data={**config.catalog,norm.KEY:spec}
    norm.validate(data,config.workload,refs,config.horizon_seconds,config.gpu_memory_mib)
    return spec


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--measurements',help='Native development receipt manifest; omit to inspect status')
    p.add_argument('--output',help='New reference JSON; never overwrites an existing version')
    args=p.parse_args(argv)
    c=SystemConfig.load(args.config)
    if not args.measurements:
        s=norm.contract(c.catalog)
        print(dumps(dict(utility_schema=c.utility_schema,normalization_sha256=norm.identity(c.catalog),
                         ready=norm.ready(c.catalog),references=s)))
        return 0
    if not args.output: p.error('--output is required when freezing references')
    path=Path(args.measurements).resolve()
    result=freeze_references(c,json.loads(path.read_text()),path.parent)
    save(args.output,result)
    print(dumps(dict(reference_file=str(Path(args.output).resolve()),normalization_sha256=norm.identity({norm.KEY:result}),
                     latency_reference_ms={t:r['deadline_ms'] for t,r in result['tasks'].items()})))
    return 0


if __name__=='__main__': raise SystemExit(main())
