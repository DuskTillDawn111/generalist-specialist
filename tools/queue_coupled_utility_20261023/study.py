"""Append-only prospective native development and conditional confirmation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import time

from tools.queue_coupled_utility_20261023 import protocol
from tools.queue_coupled_utility_20261023.native_runner import execute_job, frozen_runtime
from tools.queue_refined_20261023 import admission

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = {'text': 2, 'generalist_2': 1, 'ocr': 1}
ROLES = ('coupled_spill', 'matched_no_spill', 'legacy_latency', 'other_mixed')


def read(path):
    return json.loads(Path(path).read_text())


def design(path):
    value = dict(schema='gsdeploy.coupled_prospective_design.v1',
                 status='DECLARED_BEFORE_CONTROLLED_CALIBRATION_RESULTS',
                 model_pool=['generalist_1', 'generalist_2', 'generalist_3', 'text', 'image', 'ocr', 'asr'],
                 workload={'text': 1.85, 'image': .01, 'ocr': .01, 'asr': .01},
                 objective='Original measured equal-weight Q/T/L/P/M/E; no absolute quality gate',
                 calibration=dict(seeds=[301, 302], role='Isolated controlled text context interventions; no native total-U claim'),
                 cpu=dict(seeds=[313, 314], role='New causal shared-rate finite-domain conditional surrogate; UNKNOWN retained',
                          selection='Use highest complete full-workload conditional CPU score if available; otherwise prespecified mean+p90, margin0, cooldown30s default. Do not treat text-only diagnostic as full-workload selection.'),
                 development=dict(seeds=[303, 304], roles=list(ROLES), runs=8,
                                  order='Four roles in declared order at303, reverse at304; same Poisson/input seed within each set',
                                  endpoints='Measured six-term U and same-residency spill-control paired delta, exact causal replay, route exposure, stability, full-cohort energy diagnostic'),
                 confirmation=dict(seeds=[305, 306, 307, 308, 309, 310], roles=['coupled_spill', 'matched_no_spill'], runs=12,
                                   gate='Both development pairs fully eligible, actual new-policy spill on each seed, and delta_U>0 on each seed. Otherwise stop with explicit negative/no-mechanism evidence; do not force spill or tune weights.',
                                   order='Alternate spill/control and control/spill by whole-pair index',
                                   primary='All six predeclared pairs complete and positive; report individual deltas and uncertainty, no seven-pool physical-optimum claim',
                                   no_reselection=True, no_refit=True),
                 no_retries=True, stop_on_infrastructure_failure=True,
                 original_stability_unchanged=True, prior_rho_threshold=.8,
                 formal_measurement_seconds=600, warmup_seconds=30, drain_seconds=300,
                 energy='Original measurement-window E primary; drain/cohort diagnostic does not redefine it',
                 scope='Seven-model source pool remains frozen; new controlled resources support only exact PRIMARY text contexts. Comparisons with another explicit resident deployment are empirical controls, not exhaustive physical coverage.',
                 online_limitations=['M physical delta unknown', 'Observed-prefix conditional future cost excludes future arrivals',
                                     'Controlled dynamic-rate model remains a numerical hypothesis requiring prospective validation'])
    protocol.write(path, value)
    return value


def _sources(old):
    result = copy.deepcopy(old['sources'])
    for path in Path(__file__).parent.glob('*.py'):
        result[str(path.resolve())] = protocol.receipt(path)
    return result


def prepare(output, old_packet, resources, design_path, *, cpu_report=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    old = read(old_packet)
    plan = read(design_path)
    if plan['workload'] != {'text': 1.85, 'image': .01, 'ocr': .01, 'asr': .01}:
        raise ValueError('Frozen workload changed')
    templates = {j['role']: j for j in old['jobs']}
    sources = _sources(old)
    fingerprints = {p: r['sha256'] for p, r in sources.items()}
    base_receipt = old['base_packet']
    base = read(base_receipt['path'])
    if base['config']['workload'] != plan['workload']:
        raise ValueError('Base workload changed')
    params = {}
    allow_spill = True
    cpu_selection = 'PRESPECIFIED_DEFAULT_FULL_WORKLOAD_CPU_UNRESOLVED'
    if cpu_report:
        cpu = read(cpu_report)
        selected = cpu.get('selected_full_workload_candidate')
        full = cpu if cpu.get('domain') == 'full_workload' else cpu.get('studies', {}).get('full_workload', {})
        if selected is None and full.get('whole_declared_cpu_domain_certified') and full.get('winner_id'):
            cpu_root = Path(cpu_report).parent if cpu.get('domain') == 'full_workload' else Path(cpu_report).parent / 'full_workload'
            candidate_rows = read(cpu_root / 'dfs.json')['records']
            selected = next(row['candidate'] for row in candidate_rows if row['candidate_id'] == full['winner_id'])
        if selected:
            params = {k: selected[k] for k in ('scenarios', 'utility_margin', 'cooldown_ms')}
            allow_spill = selected.get('allow_spill', True)
            cpu_selection = 'COMPLETE_FULL_WORKLOAD_CONDITIONAL_CPU_SELECTION'
    primary = templates['calibrated_spill']['policy_contract']
    spill = protocol.make_contract(primary, resources, allow_spill=allow_spill, params=params, sources=fingerprints)
    control = protocol.make_contract(primary, resources, allow_spill=False, params=params, sources=fingerprints)
    legacy = copy.deepcopy(templates['legacy_latency']['policy_contract'])
    other = copy.deepcopy(templates['strong_other_deployment']['policy_contract'])
    for contract in (legacy, other):
        contract['source_fingerprints'] = fingerprints
        protocol.validate_contract(contract)
    contracts = dict(coupled_spill=spill, matched_no_spill=control, legacy_latency=legacy, other_mixed=other)
    inputs = list(old['inputs']) + [protocol.receipt(p) for p in (old_packet, resources, design_path)]
    if cpu_report:
        inputs.append(protocol.receipt(cpu_report))
    jobs = []
    import importlib
    configuration = importlib.import_module(frozen_runtime().__package__.split('.runtime')[0] + '.config')
    for index, seed in enumerate(plan['development']['seeds']):
        for role in (ROLES if index == 0 else tuple(reversed(ROLES))):
            contract = contracts[role]
            identity = dict(schema='gsdeploy.coupled_native_job.v1', phase='development', seed=seed, role=role,
                            policy_contract_sha256=protocol.digest(contract), base_packet=base_receipt,
                            design=protocol.receipt(design_path), workload=plan['workload'], horizon_seconds=600,
                            warmup_seconds=30, arrival_process='poisson', cpu_selection=cpu_selection,
                            measurement_claim='NEW_COUPLED_DEVELOPMENT_NOT_INDEPENDENT_CONFIRMATION')
            job = dict(id=protocol.digest(identity), label=f'{role}-seed{seed}', seed=seed, phase='development',
                       role=role, identity=identity, policy_contract=contract, nominal_plan=protocol.nominal_plan(contract))
            config = copy.deepcopy(base['config'])
            config['random_seed'] = seed
            config['catalog']['performance_mode'] = 'task_replica_proxy'
            config['catalog']['memory_mode'] = 'standalone_proxy'
            capacity = admission.capacity_envelope(configuration.SystemConfig(**config), job['nominal_plan'], contract)
            if not capacity['constraints_verified']:
                raise ValueError('Prior envelope failed: ' + str(capacity))
            job['capacity_evidence'] = capacity
            jobs.append(job)
    packet = dict(schema='gsdeploy.coupled_native_packet.v1', phase='development', base_packet=base_receipt,
                  inputs=list({r['path']: r for r in inputs}.values()), sources=sources, jobs=jobs,
                  seeds=plan['development']['seeds'], design=protocol.receipt(design_path), no_retries=True,
                  GPU_started=False, cpu_selection=cpu_selection)
    if not protocol.validate_packet(packet)['valid']:
        raise ValueError('New packet invalid')
    protocol.write(output / 'experiment-packet.json', packet)
    return packet


def analyze(packet, observations):
    groups = {}
    paired = []
    for job in packet['jobs']:
        raw = observations.get(job['label'])
        utility = raw['evaluation']['utility'] if raw else {}
        records = raw.get('request_records', []) if raw else []
        row = dict(label=job['label'], role=job['role'], seed=job['seed'], deployment=job['nominal_plan']['deployment'],
                   status=raw.get('status') if raw else 'NOT_EXECUTED', U=utility.get('U'), terms={t: utility.get(t + '_norm') for t in 'QTLPME'},
                   eligible=utility.get('eligible', False), stability=raw.get('evaluation', {}).get('stability', {}) if raw else {},
                   declared_spill_allowed=job['policy_contract'].get('coupled', {}).get('allow_spill'),
                   routing_counts={}, gates={}, request_count=len(records),
                   cohort_energy=raw.get('cohort_energy_diagnostic') if raw else None,
                   causal_verified=raw.get('evaluation', {}).get('dynamic_audit', {}).get('verified') if raw else None)
        for r in records:
            family = raw['replica_to_family'].get(r['model'])
            key = r['task'] + '->' + str(family)
            row['routing_counts'][key] = row['routing_counts'].get(key, 0) + 1
            for gate in r.get('routing_decision', {}).get('decision', {}).get('gates', []):
                row['gates'][gate] = row['gates'].get(gate, 0) + 1
        groups[job['label']] = row
    for seed in packet['seeds']:
        spill = groups.get(f'coupled_spill-seed{seed}')
        control = groups.get(f'matched_no_spill-seed{seed}')
        complete = bool(spill and control and spill['eligible'] and control['eligible'] and spill['U'] is not None and control['U'] is not None)
        paired.append(dict(seed=seed, complete_eligible=complete,
                           delta_U=spill['U'] - control['U'] if complete else None,
                           spilled_text=spill['routing_counts'].get('text->generalist_2', 0) if spill else None))
    ranking = []
    for role in sorted({j['role'] for j in packet['jobs']}):
        rows = [r for r in groups.values() if r['role'] == role]
        complete = len(rows) == len(packet['seeds']) and all(r['eligible'] and r['U'] is not None for r in rows)
        ranking.append(dict(role=role, complete_eligible=complete,
                            mean_U=statistics.fmean(r['U'] for r in rows) if complete else None))
    ranking.sort(key=lambda r: (r['mean_U'] is None, -(r['mean_U'] or 0), r['role']))
    gate = len(paired) == 2 and all(p['complete_eligible'] and p['delta_U'] > 0 and p['spilled_text'] > 0 for p in paired)
    return dict(schema='gsdeploy.coupled_native_analysis.v1', observations=list(groups.values()), paired=paired,
                ranking=ranking, confirmation_gate_passed=gate, physical_seven_pool_optimality=False,
                scope='Declared finite native candidate table only; route effect from exact matched-residency pair')


def prepare_confirmation(development_packet, development_report, output):
    packet = read(development_packet)
    report = read(development_report)
    design_doc = read(packet['design']['path'])
    if not protocol.validate_packet(packet)['valid']:
        raise ValueError('Development source packet changed')
    if not report.get('confirmation_gate_passed'):
        raise ValueError('Predeclared development gate failed; do not manufacture a confirmation')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    templates = {j['role']: j for j in packet['jobs']}
    jobs = []
    for index, seed in enumerate(design_doc['confirmation']['seeds']):
        roles = ['coupled_spill', 'matched_no_spill']
        if index % 2:
            roles.reverse()
        for role in roles:
            job = copy.deepcopy(templates[role])
            identity = copy.deepcopy(job['identity'])
            identity.update(phase='independent_confirmation', seed=seed,
                            selection=protocol.receipt(development_report),
                            measurement_claim='NEW_FIXED_SELECTION_INDEPENDENT_COUPLED_CONFIRMATION')
            job.update(id=protocol.digest(identity), label=f'{role}-seed{seed}', seed=seed,
                       phase='independent_confirmation', identity=identity)
            jobs.append(job)
    packet.update(phase='independent_confirmation', seeds=design_doc['confirmation']['seeds'], jobs=jobs,
                  inputs=[*packet['inputs'], protocol.receipt(development_packet), protocol.receipt(development_report)],
                  selection=protocol.receipt(development_report))
    if not protocol.validate_packet(packet)['valid']:
        raise ValueError('Confirmation packet invalid')
    protocol.write(output / 'experiment-packet.json', packet)
    return packet


def execute(packet_path, output):
    packet = read(packet_path)
    if not protocol.validate_packet(packet)['valid']:
        raise ValueError('Frozen packet changed')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    observations = {}
    progress = dict(status='RUNNING', packet=protocol.receipt(packet_path), completed=[], current=None, no_retries=True)
    # One mutable status view; original run files and terminal are append-only.
    def status():
        (output / 'progress.json').write_text(json.dumps(progress, indent=2, ensure_ascii=False) + '\n')
    status()
    for job in packet['jobs']:
        progress['current'] = job['label']
        status()
        print(json.dumps(dict(event='START', label=job['label'], at=time.time())), flush=True)
        try:
            raw = execute_job(packet, job['id'], output / job['label'], execute=True)
        except Exception as exc:
            progress.update(status='STOPPED_FOR_DIAGNOSIS_NO_RETRY', reason=f'{type(exc).__name__}: {exc}')
            status()
            break
        observations[job['label']] = raw
        value = raw['evaluation']['utility']
        progress['completed'].append(dict(label=job['label'], raw=protocol.receipt(output / job['label'] / 'run.json'),
                                           status=raw['status'], U=value.get('U'), eligible=value.get('eligible')))
        print(json.dumps(dict(event='END', **progress['completed'][-1])), flush=True)
        status()
        if raw['status'] != 'completed' or not value.get('energy_verified') or not value.get('cohort_verified'):
            progress.update(status='STOPPED_FOR_DIAGNOSIS_NO_RETRY', reason='INCOMPLETE_NATIVE_OR_RESOURCE_OR_CAUSAL_EVIDENCE')
            status()
            break
    else:
        progress.update(status='COMPLETED', current=None)
        status()
    report = analyze(packet, observations)
    protocol.write(output / 'analysis.json', report)
    protocol.write(output / 'terminal.json', dict(progress, analysis=protocol.receipt(output / 'analysis.json')))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('design'); d.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--output', type=Path, required=True); q.add_argument('--old-packet', type=Path, required=True)
    q.add_argument('--resources', type=Path, required=True); q.add_argument('--design', type=Path, required=True)
    q.add_argument('--cpu-report', type=Path)
    c = sub.add_parser('confirmation')
    c.add_argument('--development-packet', type=Path, required=True)
    c.add_argument('--development-report', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    e = sub.add_parser('execute'); e.add_argument('--packet', type=Path, required=True); e.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'design':
        result = design(a.output)
    elif a.command == 'prepare':
        result = prepare(a.output, a.old_packet, a.resources, a.design, cpu_report=a.cpu_report)
    elif a.command == 'confirmation':
        result = prepare_confirmation(a.development_packet, a.development_report, a.output)
    else:
        result = execute(a.packet, a.output)
    print(json.dumps(result if a.command == 'execute' else dict(status='PREPARED', command=a.command), ensure_ascii=False))
