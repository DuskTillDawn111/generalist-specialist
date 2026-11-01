"""Finite CPU search in an explicitly conditional shared-rate environment.

Every evaluator call runs its own arrival/start/finish simulation. DFS is an
independent exhaustive reference. Symbolic equivalence reduces only branches
whose decisions are provably identical; no stored objective table is searched.
Physical six-term utility and physical optimality remain UNKNOWN.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.queue_coupled_utility_20261023.model import UnknownEvidence, finite
from tools.queue_coupled_utility_20261023.policy import CoupledUtilityPolicy, percentile
from tools.queue_coupled_utility_20261023.resources import load_provider
from tools.spillover_capacity_20261022.exposure import frozen_api

CPU_SEEDS = (313, 314)
TOLERANCE = 1e-12


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(value):
    return copy.deepcopy(value) if isinstance(value, dict) else json.loads(Path(value).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def candidates(deployment):
    for spill, scenarios, margin, cooldown in itertools.product(
        (False, True), (('mean',), ('mean', 'p90')), (0., .0005, .002), (0., 30000.)):
        yield dict(deployment=copy.deepcopy(deployment), allow_spill=spill, scenarios=list(scenarios),
                   utility_margin=margin, cooldown_ms=cooldown)


def structural_groups(items):
    """With spill disabled, margin and cooldown cannot affect any chosen alias.

    Specialist scoring and deterministic ties still depend on scenarios, so
    scenarios are retained in the equivalence key. Induction on common arrivals
    preserves all active work, completion feedback and every next snapshot.
    """
    groups = {}
    for item in items:
        key = identity(item) if item['allow_spill'] else identity(dict(
            deployment=item['deployment'], allow_spill=False, scenarios=item['scenarios']))
        groups.setdefault(key, []).append(item)
    result = []
    for members in groups.values():
        members.sort(key=identity)
        result.append(dict(representative=members[0], members=members,
            proof=None if len(members) == 1 else dict(
                rule='SPILL_DISABLED_MARGIN_COOLDOWN_ALIAS_EQUIVALENCE', objective_used=False,
                conditions=['Identical deployment, scenarios, arrivals, input cohort and shared-rate provider',
                            'allow_spill=False for every member'],
                argument='Margin applies only to generalist acceptance; cooldown gates only spill. Both cannot change specialist scoring or deterministic alias ties. Identical aliases inductively give identical shared progress, completions and objective.',
                scope='Deterministic conditional CPU proxy only; no physical repeated-run equivalence')))
    return sorted(result, key=lambda row: identity(row['representative']))


class SharedRateEnvironment:
    """FIFO work-conserving conditional process; reprice every active head.

    Future arrival events advance latent progress but are never exposed to the
    policy. A service_ms estimate denotes the duration of one unit of work in
    that current active context. Missing service or whole-board power aborts
    the run as UNKNOWN, including unsupported background paths.
    """
    def __init__(self, aliases, provider, deadline_ns):
        self.aliases = copy.deepcopy(aliases)
        self.provider = provider
        self.deadline_ns = deadline_ns
        self.clock = 0.
        self.pending = {a: deque() for a in aliases}
        self.remaining = {}
        self.last = {a: None for a in aliases}
        self.results = {}
        self.newly_completed = []
        self.energy_joules = 0.
        self.event_count = 0
        self.rate_changes = 0
        self.active_contexts = Counter()

    def active(self):
        return {a: rows[0] for a, rows in self.pending.items() if rows}

    def snapshot(self):
        return {a: dict(outstanding=[dict(request_id=r['request_id'], task=r['task'], pool_id=r['pool_id'],
                    arrival_ns=r['arrival_ns'], enqueued_ns=r['arrival_ns'], actual_submit_ns=r['arrival_ns'])
                    for r in rows], last_completed_ns=self.last[a]) for a, rows in self.pending.items()}

    def _evidence(self):
        active = self.active()
        power = self.provider.power(active, self.aliases, 'mean')
        if power.get('known') is not True or not finite(power.get('watts')) or power['watts'] < 0:
            raise UnknownEvidence('UNKNOWN_ENVIRONMENT_BOARD_POWER:' + str(power.get('status')))
        rates = {}
        for alias, row in active.items():
            value = self.provider.service(row, active, self.aliases, 'mean')
            if (value.get('known') is not True or value.get('context_supported') is not True
                    or not finite(value.get('service_ms')) or value['service_ms'] <= 0):
                raise UnknownEvidence('UNKNOWN_ENVIRONMENT_SERVICE:' + alias + ':' + row['task'] + ':' + str(value.get('status')))
            rates[alias] = value['service_ms']
        self.rate_changes += len(rates)
        self.active_contexts[json.dumps(sorted((a, r['task']) for a, r in active.items()))] += 1
        return rates, power['watts']

    def _interval(self, until, rates, watts):
        delta = until - self.clock
        if delta < -1e-6:
            raise ValueError('Environment time moved backward')
        delta = max(0., delta)
        self.energy_joules += watts * max(0., min(until, self.deadline_ns) - min(self.clock, self.deadline_ns)) / 1e9
        for alias, service_ms in rates.items():
            self.remaining[alias] = max(0., self.remaining[alias] - delta / (service_ms * 1e6))
        self.clock = until

    def advance(self, until_ns):
        if until_ns < self.clock - 1e-6:
            raise ValueError('Arrival clock is not monotone')
        while self.clock < until_ns:
            rates, watts = self._evidence()
            if not rates:
                self._interval(float(until_ns), rates, watts)
                break
            durations = {a: self.remaining[a] * ms * 1e6 for a, ms in rates.items()}
            delta = min(durations.values())
            target = min(float(until_ns), self.clock + delta)
            finishes = self.clock + delta <= until_ns + 1e-6
            self._interval(target, rates, watts)
            if not finishes:
                break
            completed = [a for a, d in durations.items() if abs(d - delta) <= max(1e-6, abs(delta) * 1e-12)]
            self.event_count += 1
            if self.event_count > 100000:
                raise UnknownEvidence('ENVIRONMENT_EVENT_LIMIT')
            for alias in completed:
                row = self.pending[alias].popleft()
                finish = round(self.clock)
                record = self.results[row['request_id']]
                record.update(completion_ns=finish, finish_ns=self.clock,
                              service_ms=(self.clock-record['start_ns']) / 1e6)
                self.newly_completed.append(dict(record))
                self.last[alias] = finish
                if self.pending[alias]:
                    self.remaining[alias] = 1.
                    self.results[self.pending[alias][0]['request_id']]['start_ns'] = self.clock
                else:
                    self.remaining.pop(alias)

    def enqueue(self, request, alias):
        if request['task'] not in self.aliases[alias]['supported_tasks']:
            raise ValueError('Unsupported environment route')
        row = dict(request, alias=alias)
        if row['request_id'] in self.results:
            raise ValueError('Duplicate environment request')
        self.results[row['request_id']] = dict(row, family=self.aliases[alias]['family'])
        self.pending[alias].append(row)
        if len(self.pending[alias]) == 1:
            self.remaining[alias] = 1.
            self.results[row['request_id']]['start_ns'] = self.clock
        # Validate the entire newly active shared state immediately.
        self._evidence()

    def drain(self):
        while self.active():
            rates, _ = self._evidence()
            until = self.clock + min(self.remaining[a] * ms * 1e6 for a, ms in rates.items())
            self.advance(until)


def evaluate_cohort(contract, resource_document, cohort, candidate, *, text_only=False, provider=None, memory_norm=None):
    aliases = contract['aliases']
    provider = provider or load_provider(resource_document, candidate['deployment'])
    workload = {'text': contract['text_rate'], **{t: sum(v.values()) for t, v in contract.get('background_routing', {}).items()}}
    if text_only:
        workload = {'text': workload['text']}
    horizon_ns = round(contract['horizon_seconds'] * 1e9)
    environment = SharedRateEnvironment(aliases, provider, horizon_ns)
    policy = CoupledUtilityPolicy(aliases, provider, contract['quality_loss'], contract['weights'],
        contract['latency_refs_ms'], workload, scenarios=candidate['scenarios'], allow_spill=candidate['allow_spill'],
        utility_margin=candidate['utility_margin'], cooldown_ms=candidate['cooldown_ms'],
        exposure_rate_rps=contract['text_rate']/8, exposure_burst=1., energy_reference_watts=450., drift_ratio_limit=2.)
    policy.start_phase(0, horizon_ns)
    api = frozen_api()
    _, routes = api['trace'].expand_routes(candidate['deployment'], contract['background_routing'])
    router = api['routing'].Router(routes)
    decisions = []
    try:
        for request in cohort:
            if text_only and request['task'] != 'text':
                continue
            environment.advance(request['arrival_ns'])
            policy.observe_completed(environment.newly_completed)
            environment.newly_completed.clear()
            if request['task'] == 'text':
                policy.set_arrival(request['request_id'], request['arrival_ns'])
                result = policy.choose('text', request['pool_id'], request['arrival_ns'], environment.snapshot())
                alias = result['alias']
                decisions.append(dict(request_id=request['request_id'], alias=alias,
                    spill=result['actual_spill_selected'], status=result['status'], gates=result['gates']))
            else:
                alias = router.choose(request['task'])
            policy.observe_admission(dict(request, alias=alias))
            environment.enqueue(request, alias)
        environment.advance(horizon_ns)
        environment.drain()
        rows = list(environment.results.values())
        if not rows:
            raise UnknownEvidence('EMPTY_COHORT')
        terms = dict(Q=0., T=1-sum(r['finish_ns'] <= horizon_ns for r in rows)/len(rows), L=0., P=0.,
                     M=memory_norm, E=environment.energy_joules/(450.*contract['horizon_seconds']))
        metrics = {}
        for task, rate in workload.items():
            task_rows = [r for r in rows if r['task'] == task]
            if not task_rows:
                raise UnknownEvidence('MISSING_TASK_COHORT:' + task)
            latencies = [(r['finish_ns']-r['arrival_ns'])/1e6 for r in task_rows]
            losses = [contract['quality_loss'][r['family']][task] for r in task_rows]
            if any(not finite(v) or v < 0 for v in losses):
                raise UnknownEvidence('UNKNOWN_QUALITY:' + task)
            alpha = rate/sum(workload.values())
            terms['Q'] += alpha*statistics.mean(losses)
            terms['L'] += alpha*statistics.mean(latencies)/contract['latency_refs_ms'][task]
            terms['P'] += alpha*percentile(latencies)/contract['latency_refs_ms'][task]
            metrics[task] = dict(count=len(task_rows), mean_ms=statistics.mean(latencies), p95_ms=percentile(latencies))
        objective = -sum(contract['weights'][t]*v for t, v in terms.items() if v is not None)
        return dict(status='COMPLETE_CONDITIONAL_CPU_PROXY', U=objective, eligible=True, terms=terms,
            formal_U=None, physical_U=None, original_stability='UNKNOWN_NOT_A_NATIVE_MEASUREMENT',
            memory_scope='Calibration empirical resident peak constant, not actual route-dependent peak' if memory_norm is not None else 'M_UNKNOWN_OMITTED_FROM_PARTIAL_OBJECTIVE',
            energy_scope='Shared whole-board measurement window integral only; startup amortization omitted',
            objective_terms=[t for t, v in terms.items() if v is not None], task_metrics=metrics,
            request_count=len(rows), spill_count=sum(r['spill'] for r in decisions),
            routes=dict(Counter(r['alias'] for r in rows)), route_trace_sha256=identity([(r['request_id'], r['alias']) for r in rows]),
            completion_trace_sha256=identity([(r['request_id'], r['finish_ns']) for r in rows]),
            decision_statuses=dict(Counter(r['status'] for r in decisions)), events=environment.event_count,
            active_rate_recomputations=environment.rate_changes, drain_finish_ns=environment.clock,
            conditional_proxy_only=True, future_arrivals_visible_to_policy=False)
    except UnknownEvidence as exc:
        return dict(status='UNKNOWN', reason=str(exc), U=None, eligible=None, formal_U=None, physical_U=None,
            observed_requests=len(environment.results), events=environment.event_count,
            active_rate_recomputations=environment.rate_changes, unsupported_not_infeasible=True,
            conditional_proxy_only=True, future_arrivals_visible_to_policy=False)


def execute_search(items, seeds, evaluate, *, optimized=False):
    """Real evaluations per visited representative; strict nonnegative-cost bound.

    U<=0 is valid because all retained terms/weights are nonnegative. Strict
    bound pruning preserves equal optima. Typically this weak bound prunes no
    candidates; symbolic branch equivalence is reported separately.
    """
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Distinct nonempty evaluation seeds required')
    if len(items) != len({identity(c) for c in items}):
        raise ValueError('Duplicate candidate identity')
    started = time.perf_counter()
    groups = structural_groups(items) if optimized else [dict(representative=c, members=[c], proof=None) for c in items]
    records = []
    values = {}
    calls = 0
    pruned = 0
    best = None
    for group in groups:
        if optimized and best is not None and 0. < best-TOLERANCE:
            pruned += len(group['members'])
            continue
        runs = []
        for seed in seeds:
            result = evaluate(group['representative'], seed)
            calls += 1
            runs.append(dict(result, seed=seed))
        known = all(r.get('eligible') is True and finite(r.get('U')) for r in runs)
        value = statistics.mean(r['U'] for r in runs) if known else None
        if value is not None and value > TOLERANCE:
            raise ValueError('Nonnegative-cost upper-bound proof violated')
        for candidate in group['members']:
            cid = identity(candidate)
            records.append(dict(candidate_id=cid, candidate=candidate, representative_id=identity(group['representative']),
                objective=value, status='COMPLETE' if known else 'UNKNOWN', runs=copy.deepcopy(runs),
                actual_evaluator_called_for_identity=cid == identity(group['representative']), equivalence_proof=group['proof']))
            if known:
                values[cid] = value
        if known:
            best = value if best is None else max(best, value)
    ties = sorted(cid for cid, value in values.items() if best-value <= TOLERANCE) if best is not None else []
    return dict(algorithm='symbolic_quotient_strict_zero_bound' if optimized else 'plain_dfs',
        winner_id=ties[0] if ties else None, tie_ids=ties, exact_max_objective=best,
        candidate_count=len(items), representative_count=len(groups), oracle_calls=calls,
        proved_equivalent_candidates_skipped=len(items)-len(groups), strict_bound_candidates_pruned=pruned,
        unknown_ids=sorted(r['candidate_id'] for r in records if r['status']=='UNKNOWN'),
        whole_declared_cpu_domain_certified=bool(ties) and len(values)==len(items), physical_optimality=False,
        elapsed_seconds=time.perf_counter()-started, records=records, objective_cache_hits=0,
        timing_scope='Includes symbolic groups and independent actual simulations; no objective-table lookup')


def run_study(output, template_contract, resource_path, base_packet, seeds=CPU_SEEDS):
    if tuple(seeds) != CPU_SEEDS:
        raise ValueError('Fresh declared CPU seeds313/314 required')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract, base, resources = read(template_contract), read(base_packet), read(resource_path)
    contract['horizon_seconds'] = base['config']['horizon_seconds']
    if contract['text_rate'] != 1.85 or contract['horizon_seconds'] != 600.:
        raise ValueError('Frozen PRIMARY600s text1.85 workload required')
    if contract['deployment'] != {'text':2, 'generalist_2':1, 'ocr':1}:
        raise ValueError('Exact frozen PRIMARY deployment required')
    provider = load_provider(resources, contract['deployment'])
    config = base['config']
    samples = Path(base['inputs'])/'runtime-inputs.jsonl'
    if hashlib.sha256(samples.read_bytes()).hexdigest() != config['catalog']['quality']['identity']['input_pool_sha256']:
        raise ValueError('Frozen input pool changed')
    api = frozen_api()
    workload = {'text': contract['text_rate'], **{t: sum(v.values()) for t, v in contract['background_routing'].items()}}
    cohorts = {}
    for seed in seeds:
        pool = api['workload'].SamplePool(samples, list(workload), seed)
        trace = api['trace'].make_trace(workload, 600., seed, config['arrival_process'])
        cohorts[seed] = [dict(request_id=f'coupled-cpu{seed}:{i}', arrival_ns=round(arrival*1e9), task=task,
                              pool_id=pool.next(task)['pool_id']) for i, (arrival, task) in enumerate(trace)]
    memory = resources.get('memory_evidence') or {}
    budget = config.get('gpu_memory_mib')
    memory_norm = memory.get('safe_memory_mib')/budget if (memory.get('measured') is True and memory.get('resource_valid') is True
        and finite(memory.get('safe_memory_mib')) and memory['safe_memory_mib'] >= 0 and finite(budget) and budget > 0) else None
    domain = list(candidates(contract['deployment']))
    save(output/'domain.json', dict(candidates=domain, seeds=list(seeds), full_workload=workload, horizon_seconds=600.,
        formal_U=None, resource_sha256=identity(resources), declaration='PRIMARY finite24 policy domain; not full seven-model deployment domain'))
    studies = {}
    for name, text_only in (('full_workload', False), ('text_only_mechanism_diagnostic', True)):
        directory = output/name
        directory.mkdir()
        def evaluate(candidate, seed):
            return evaluate_cohort(contract, resources, cohorts[seed], candidate, text_only=text_only,
                                   provider=provider, memory_norm=memory_norm)
        dfs = execute_search(domain, seeds, evaluate)
        optimized = execute_search(domain, seeds, evaluate, optimized=True)
        dfs_by = {r['candidate_id']:r for r in dfs['records']}
        equivalence_checks = []
        for row in optimized['records']:
            other = dfs_by[row['candidate_id']]
            same = (row['status']==other['status'] and row['objective']==other['objective']
                and [r.get('route_trace_sha256') for r in row['runs']]==[r.get('route_trace_sha256') for r in other['runs']]
                and [r.get('completion_trace_sha256') for r in row['runs']]==[r.get('completion_trace_sha256') for r in other['runs']])
            if not same:
                raise ValueError('Independent DFS/equivalence mismatch:'+row['candidate_id'])
            equivalence_checks.append(dict(candidate_id=row['candidate_id'], identical=True))
        same_optimum = dfs['winner_id']==optimized['winner_id'] and dfs['tie_ids']==optimized['tie_ids']
        if not same_optimum:
            raise ValueError('Independent DFS optimum/ties mismatch')
        save(directory/'dfs.json', dfs)
        save(directory/'optimized.json', optimized)
        unknown = Counter(r.get('reason') for row in dfs['records'] for r in row['runs'] if r['status']=='UNKNOWN')
        summary = dict(domain=name, workload={'text':workload['text']} if text_only else workload,
            same_optimum_and_all_ties=same_optimum, optimum_identified=dfs['winner_id'] is not None,
            whole_declared_cpu_domain_certified=dfs['whole_declared_cpu_domain_certified'],
            dfs_oracle_calls=dfs['oracle_calls'], optimized_oracle_calls=optimized['oracle_calls'],
            equivalent_candidates_skipped=optimized['proved_equivalent_candidates_skipped'],
            strict_bound_candidates_pruned=optimized['strict_bound_candidates_pruned'],
            dfs_seconds=dfs['elapsed_seconds'], optimized_seconds=optimized['elapsed_seconds'],
            process_local_speedup=dfs['elapsed_seconds']/optimized['elapsed_seconds'] if optimized['elapsed_seconds'] else None,
            unknown_run_reason_counts=dict(unknown), winner_id=dfs['winner_id'], tie_ids=dfs['tie_ids'],
            exact_max_objective=dfs['exact_max_objective'], equivalence_checks=equivalence_checks,
            physical_U=None, physical_optimality=False, original_stability_thresholds_not_changed=True,
            scope='Conditional text-only mechanism diagnostic; cannot certify original full workload' if text_only else
                  'Original mixed workload numerical coverage; unsupported background is UNKNOWN, never zero or infeasible')
        save(directory/'comparison.json', summary)
        studies[name] = summary
    result = dict(schema='gsdeploy.coupled_cpu_search.v1', seeds=list(seeds), studies=studies,
        physical_optimality=False, physical_U=None, GPU_started=False,
        unknown_never_imputed=True, old_reserved_results_used=False,
        limitations=['One PRIMARY resident deployment,24 policy candidates only; no seven-model global deployment certificate.',
            'CPU environment is conditional controlled progress proxy, not calibrated physical counterfactual truth.',
            'Quality is frozen family/task aggregate loss, not measured per-input native accuracy.',
            'Empirical resident M is constant if available; actual route-dependent peak and startup E remain unknown.',
            'DFS equality certifies the finite numerical domain only; native validation and original stability gates remain necessary.'])
    save(output/'summary.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--template-contract', type=Path, required=True)
    parser.add_argument('--resources', type=Path, required=True)
    parser.add_argument('--base-packet', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_study(args.output, args.template_contract, args.resources, args.base_packet), ensure_ascii=False))


if __name__ == '__main__':
    main()
