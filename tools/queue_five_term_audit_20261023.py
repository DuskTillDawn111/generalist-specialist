"""External, read-only native five-term evidence audit; never imports a GPU API.

Only terminal raw receipts are consumed. Incomplete files remain pending. This
audits evidence and finite comparisons, never full-pool physical optimality.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WEIGHTS = dict(Q=.2, T=.2, L=.2, P=0., M=.2, E=.2)
OBJECTIVE = 'gsdeploy.five_term_no_tail.v1'
TERMS = tuple('QTLPME')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    path = Path(path).resolve()
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            value.update(chunk)
    return dict(path=str(path), sha256=value.hexdigest())


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(left, right):
    return finite(left) and finite(right) and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-8)


def percentile(values, q=.95):
    values = sorted(values)
    if not values:
        return None
    position = (len(values)-1)*q
    a, b = math.floor(position), math.ceil(position)
    return values[a] + (values[b]-values[a])*(position-a)


def integrate(samples, start, end):
    points = sorted((row['time_ns'], row['watts']) for row in samples)
    if (type(start) is not int or type(end) is not int or end <= start or len(points) < 2
            or points[0][0] > start or points[-1][0] < end
            or any(type(t) is not int or not finite(w) or w < 0 for t, w in points)):
        return None
    result = 0.
    for (a, wa), (b, wb) in zip(points, points[1:]):
        lo, hi = max(a, start), min(b, end)
        if hi <= lo:
            continue
        if b <= a or b-a > 2_000_000_000:
            return None
        wlo = wa + (wb-wa)*(lo-a)/(b-a)
        whi = wa + (wb-wa)*(hi-a)/(b-a)
        result += (wlo+whi)*.5*(hi-lo)/1e9
    return result


def stable_json(path):
    """Do not consume a partially written or changing terminal receipt."""
    path = Path(path)
    if not path.is_file():
        return None, None, 'NOT_YET_WRITTEN'
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    stamp = lambda s: (s.st_ino, s.st_size, s.st_mtime_ns)
    if stamp(before) != stamp(after) or len(payload) != after.st_size:
        return None, None, 'FILE_CHANGED_WHILE_READING'
    try:
        data = json.loads(payload)
    except ValueError:
        return None, None, 'INCOMPLETE_JSON_PENDING'
    if data.get('status') in {'starting', 'running', 'loading'}:
        return None, None, 'NONTERMINAL_RAW_PENDING'
    return data, dict(path=str(path.resolve()), sha256=hashlib.sha256(payload).hexdigest()), None


def bindings(packet):
    all_refs = [*packet.get('inputs', []), *packet.get('sources', {}).values()]
    failures = []
    unique = {}
    for ref in all_refs:
        if ref['path'] in unique and unique[ref['path']] != ref['sha256']:
            failures.append('CONFLICTING_RECEIPTS:' + ref['path'])
        unique[ref['path']] = ref['sha256']
    for path, expected in unique.items():
        try:
            if sha(path)['sha256'] != expected:
                failures.append('BINDING_CHANGED:' + path)
        except OSError:
            failures.append('BINDING_MISSING:' + path)
    return dict(verified=not failures, unique_receipts=len(unique), errors=failures)


def mechanisms(raw):
    families = raw.get('replica_to_family', {})
    specialists = sorted(a for a, family in families.items() if family == 'text')
    generalists = sorted(a for a, family in families.items() if family.startswith('generalist'))
    counts, gates, triggers = Counter(), Counter(), Counter()
    examples, queued = [], []
    actual_spills = 0
    busy_witness_count = queued_witness_count = 0
    for row in raw.get('request_records', []):
        family = families.get(row.get('model'), 'UNKNOWN')
        counts[row['task'] + '->' + family] += 1
        routing = row.get('routing_decision') or {}
        decision = routing.get('decision') or {}
        gates.update(decision.get('gates', []))
        triggers[decision.get('trigger', 'UNKNOWN')] += 1
        if row.get('task') != 'text' or family not in {'generalist_1', 'generalist_2', 'generalist_3'}:
            continue
        actual_spills += 1
        snapshot = routing.get('snapshot') or {}
        outstanding = {a: len(snapshot.get(a, {}).get('outstanding', [])) for a in families}
        idle_g = [a for a in generalists if a in snapshot and outstanding[a] == 0]
        if len(specialists) >= 2 and all(outstanding[a] >= 1 for a in specialists) and idle_g:
            busy_witness_count += 1
            example = dict(request_id=row['request_id'], pool_id=row.get('pool_id'),
                           scheduled_offset_seconds=row.get('scheduled_offset_seconds'),
                           selected_alias=row['model'], outstanding_counts=outstanding,
                           host_visible_empty_generalist_aliases=idle_g,
                           both_specialists_have_waiting_requests=all(outstanding[a] >= 2 for a in specialists),
                           decision_trigger=decision.get('trigger'),
                           baseline_specialist=decision.get('baseline_specialist'),
                           scenario_delta_cost_vs_specialist=next((a.get('scenario_delta_cost_vs_selected_specialist')
                                                                  for a in decision.get('audit', []) if a.get('alias') == row['model']), None),
                           scope='Host-visible outstanding queues only; no physical GPU-idle or counterfactual latency certificate')
            if len(examples) < 5:
                examples.append(example)
            if example['both_specialists_have_waiting_requests'] and len(queued) < 5:
                queued.append(example)
            queued_witness_count += example['both_specialists_have_waiting_requests']
    return dict(routing_counts=dict(counts), actual_text_spills=actual_spills,
                gates=dict(gates), triggers=dict(triggers),
                dual_specialist_outstanding_generalist_empty_examples=examples,
                dual_specialist_waiting_generalist_empty_examples=queued,
                dual_specialist_outstanding_generalist_empty_count=busy_witness_count,
                dual_specialist_waiting_generalist_empty_count=queued_witness_count,
                witness_scope='Host-visible mechanism instances; benefits require separate matched native U comparisons')


def audit_run(packet, job, raw_path, *, replay=True, terminal_marker=None):
    if terminal_marker is not None and not Path(terminal_marker).exists():
        return dict(label=job['label'], role=job['role'], seed=job['seed'], status='PENDING', reason='AWAITING_POST_WRITE_COMPLETION_MARKER')
    raw, raw_receipt, pending = stable_json(raw_path)
    if pending:
        return dict(label=job['label'], role=job['role'], seed=job['seed'], status='PENDING', reason=pending)
    checks, errors, diagnostic = [], [], {}
    observed_utility = (raw.get('evaluation') or {}).get('utility') or {}
    diagnostic.update(recorded_U=observed_utility.get('U'), eligible=observed_utility.get('eligible'),
                      recorded_terms={t:observed_utility.get(t+'_norm') for t in TERMS},
                      raw_status=raw.get('status'), raw_errors=raw.get('errors'),
                      stability=(raw.get('evaluation') or {}).get('stability'), mechanism=mechanisms(raw))

    def check(condition, name, detail=None):
        checks.append(dict(name=name, passed=bool(condition), **({'detail':detail} if detail is not None else {})))
        if not condition:
            errors.append(name)

    try:
        from tools.queue_five_term_20261023 import evaluation, protocol
        from tools.queue_refined_20261023 import admission
        from tools.queue_refined_20261023.native_runner import frozen_runtime
        config_module = importlib.import_module(frozen_runtime().__package__.split('.runtime')[0] + '.config')
        config = config_module.SystemConfig(**raw['config'])
        contract, transport, window = raw['policy_contract'], raw['transport_config'], raw['measurement_window']
        utility = raw['evaluation']['utility']
        check(raw.get('status') == 'completed' and raw.get('backend') == 'native' and raw.get('errors') == [], 'NATIVE_TERMINAL_SUCCESS')
        check(raw.get('experiment_job_id') == job['id'] and raw.get('executed_measurement_identity') == job['identity']
              and raw.get('executed_measurement_id') == canonical(job['identity']), 'JOB_IDENTITY_EXACT')
        check(canonical(job['identity']) == job['id'] and canonical(contract) == job['identity']['policy_contract_sha256'], 'CONTRACT_JOB_HASHES')
        check(contract == job['policy_contract'], 'FROZEN_POLICY_CONTRACT_EXACT')
        if job['role'] in {'matched_no_spill','legacy_no_spill'}:
            check(diagnostic['mechanism']['actual_text_spills'] == 0, 'PREDECLARED_NO_SPILL_ARM_ROUTING')
        protocol.validate_contract(contract)
        check(True, 'FROZEN_POLICY_VALIDATOR')
        check(raw['config']['weights'] == contract['weights'] == job['identity']['weights'] == WEIGHTS, 'FIXED_NEW_WEIGHT_BINDING')
        check(raw.get('objective_schema') == contract.get('objective_schema') == utility.get('objective_schema') == OBJECTIVE, 'FIVE_TERM_OBJECTIVE_IDENTITY')
        check(utility.get('P_evaluation_role') == 'diagnostic_only' and utility.get('P_contribution') == 0., 'P_DIAGNOSTIC_ZERO_CONTRIBUTION')
        check(raw['plan'] == job['nominal_plan'] and raw['plan']['deployment'] == {'text':2,'generalist_2':1,'ocr':1}, 'PRESPECIFIED_PRIMARY_PLAN')
        check(raw.get('source_packet') == job['identity']['base_packet'], 'FROZEN_BASE_PACKET_IDENTITY')
        base = json.loads(Path(job['identity']['base_packet']['path']).read_bytes())
        pool = lambda catalog:{m:{k:spec.get(k) for k in ('name','role','supported_tasks')} for m,spec in catalog['models'].items()}
        check(pool(config.catalog) == pool(base['config']['catalog']), 'FROZEN_SEVEN_MODEL_SOURCE_POOL_IDENTITY')
        diagnostic['model_pool'] = pool(config.catalog)
        check(raw.get('dynamic_capacity_evidence') == job['capacity_evidence'], 'PRIOR_CAPACITY_EVIDENCE_IDENTITY')
        check(config.workload == transport['workload_rps'] == job['identity']['workload'], 'WORKLOAD_CONFIG_TRANSPORT_IDENTITY')
        check(transport['random_seed'] == config.random_seed == job['seed'], 'SEED_CONFIG_TRANSPORT_IDENTITY')
        check(transport['measurement_seconds'] == config.horizon_seconds == window['duration_seconds'] == 600
              and transport['warmup_seconds'] == job['identity']['warmup_seconds'] == 30, 'PRESPECIFIED_WINDOW_AND_WARMUP')
        check(transport['arrival_process'] == job['identity']['arrival_process'] == 'poisson', 'PRESPECIFIED_ARRIVAL_PROCESS')
        identity = admission.measurement_identity(config, raw)
        check(raw.get('policy_identity') == admission.policy_identity(contract)
              and raw.get('measurement_identity') == identity, 'TRANSPORT_POLICY_SYSTEM_OBJECTIVE_IDENTITY')
        diagnostic['transport_weight_scope'] = 'Transport schema has no independent weight field; measurement identity binds full config, policy and dispatch log.'
        rows = raw['request_records']
        check(len(rows) == len({r['request_id'] for r in rows}) == window['offered_requests'] > 0, 'EXACT_UNIQUE_MEASUREMENT_COHORT')
        check(all(r['phase'] == 'measurement' and r['status'] == 'completed' for r in rows), 'ALL_REQUESTS_COMPLETED')
        cohort = [[r.get(k) for k in ('request_id','task','pool_id','sample_id','scheduled_offset_seconds')] for r in rows]
        check(all(v is not None for row in cohort for v in row), 'COMPLETE_REQUEST_SAMPLE_SCHEDULE_IDENTITY')
        cohort.sort(key=lambda r:json.dumps(r, ensure_ascii=False))
        diagnostic['cohort_signature'] = canonical(cohort)
        stats, total = {}, math.fsum(config.workload.values())
        for task, rate in config.workload.items():
            task_rows = [r for r in rows if r['task'] == task and r['status'] == 'completed']
            latencies = [(r['completion_time_ns']-r['scheduled_time_ns'])/1e6 for r in task_rows]
            check(bool(latencies) and all(close(v,r.get('e2e_latency_ms')) for v,r in zip(latencies,task_rows)), 'RAW_E2E_TIMESTAMPS_'+task)
            stats[task] = dict(mean_ms=statistics.fmean(latencies), p95_ms=percentile(latencies), requests=len(latencies))
        L = math.fsum(config.workload[t]/total*stats[t]['mean_ms']/config.latency_reference_ms[t] for t in sorted(stats))
        P = math.fsum(config.workload[t]/total*stats[t]['p95_ms']/config.latency_reference_ms[t] for t in sorted(stats))
        inside = sum(r['completion_time_ns'] <= window['end_time_ns'] for r in rows)
        T = 1-inside/len(rows)
        memory, energy = raw['memory_evidence'], raw['energy_evidence']
        memory_samples = [r for r in [memory.get('before'),memory.get('after'),*memory.get('samples',[])] if r is not None]
        check(memory.get('measured') is True and memory.get('resource_valid') is True
              and memory.get('measurement_identity') == identity and bool(memory_samples)
              and all(not r['foreign_pids'] for r in memory_samples), 'MEASURED_OWNED_MEMORY_IDENTITY')
        safe = config.catalog['safe_factor']*max(r['owned_used_mib'] for r in memory_samples)
        check(close(safe,memory.get('safe_memory_mib')) and safe <= config.gpu_memory_mib, 'MEMORY_SAMPLING_ARITHMETIC')
        M = safe/config.gpu_memory_mib
        check(energy.get('kind') == 'measured' and energy.get('scope') == 'whole_gpu_board'
              and energy.get('coverage_valid') is True and energy.get('sampling_errors') == []
              and energy.get('measurement_identity') == identity and energy.get('profile_key') == identity, 'WHOLE_BOARD_ENERGY_IDENTITY_COVERAGE')
        run_j = integrate(energy['samples'],window['start_time_ns'],window['end_time_ns'])
        startup_j = integrate(energy['samples'],energy['startup_begin_time_ns'],window['start_time_ns'])
        check(close(run_j,energy.get('run_joules')) and close(startup_j,energy.get('startup_joules')), 'INDEPENDENT_ENERGY_INTEGRALS')
        E = (run_j/window['duration_seconds'] + startup_j/config.startup_amortization_seconds)/config.energy_reference_watts if run_j is not None and startup_j is not None else None
        Q, quality = evaluation.original.measure_quality(config.catalog,raw,config.workload)
        check(quality == utility.get('quality_evidence') and admission.quality_identity_covers(config,raw), 'QUALITY_OUTPUT_RESCORING_AND_IDENTITIES')
        check(utility.get('absolute_quality_gate') is False and contract.get('absolute_quality_gate') is False, 'NO_ABSOLUTE_QUALITY_GATE')
        terms = dict(Q=Q,T=T,L=L,P=P,M=M,E=E)
        for term,value in terms.items():
            check(close(value,utility.get(term+'_norm')), 'INDEPENDENT_NORMALIZED_TERM_'+term)
            check(close(-WEIGHTS[term]*value,utility.get(term+'_contribution')), 'WEIGHTED_CONTRIBUTION_'+term)
        recomputed_U = -math.fsum(WEIGHTS[t]*terms[t] for t in 'QTLME') if all(finite(terms[t]) for t in 'QTLME') else None
        check(close(recomputed_U,utility.get('diagnostic_U')), 'INDEPENDENT_FIVE_TERM_U_ARITHMETIC')
        check(close(recomputed_U,utility.get('U')) if utility.get('eligible') else utility.get('U') is None, 'FORMAL_U_ELIGIBILITY_GATING')
        check(utility.get('utility_complete') is True and utility.get('original_stability_thresholds_unchanged') is True, 'COMPLETE_TERMS_ORIGINAL_STABILITY_RULES')
        if replay:
            recomputed = evaluation.evaluate_run(config,raw)
            check(recomputed.get('dynamic_audit',{}).get('verified') is True, 'NEW_DISPATCHER_EXACT_CAUSAL_REPLAY')
            check(recomputed.get('dynamic_audit',{}).get('warmup',{}).get('verified') is True, 'FRESH_WARMUP_CAUSAL_REPLAY')
            check(recomputed.get('stability') == raw['evaluation']['stability'], 'ORIGINAL_STABILITY_RECONSTRUCTION')
            check(recomputed['utility'].get('eligible') == utility.get('eligible')
                  and (close(recomputed['utility'].get('U'),utility.get('U')) if utility.get('eligible') else recomputed['utility'].get('U') is None), 'REPLAYED_ELIGIBILITY_AND_U')
        else:
            errors.append('CAUSAL_REPLAY_NOT_RUN')
        check(all(not value['forced'] and value['exit_code'] == 0 and value['bye_received'] for value in raw['shutdown'].values())
              and set(raw['shutdown']) == set(raw['replica_to_family']), 'GRACEFUL_ALL_WORKER_SHUTDOWN')
        drain_j = integrate(energy['samples'],window['start_time_ns'],window['drain_end_time_ns'])
        diagnostic.update(terms=terms, recomputed_U=recomputed_U, recorded_U=utility.get('U'), eligible=utility.get('eligible'),
                          per_task_stats=stats, quality_identity=quality.get('identity'),
                          quality_task_losses={t:v.get('loss') for t,v in quality.get('per_task',{}).items()},
                          completed_within_window=inside, completed_during_drain=len(rows)-inside,
                          drain_seconds=(window['drain_end_time_ns']-window['end_time_ns'])/1e9,
                          energy_full_cohort_diagnostic_joules=drain_j,
                          energy_after_window_diagnostic_joules=drain_j-run_j if drain_j is not None and run_j is not None else None,
                          recorded_cohort_energy_diagnostic=raw.get('cohort_energy_diagnostic'),
                          stability=raw['evaluation']['stability'],
                          end_backlog=raw['evaluation'].get('stability_evidence',{}).get('end_backlog_requests'),
                          final_backlog_after_drain=raw['evaluation'].get('stability_evidence',{}).get('final_backlog_after_drain'),
                          mechanism=mechanisms(raw))
        # Formal energy deliberately excludes this drain diagnostic.
        check(max(r['completion_time_ns'] for r in rows) <= window['drain_end_time_ns']
              and window['drain_end_time_ns'] >= window['end_time_ns'], 'BOUNDED_DRAIN_COMPLETION')
    except (KeyError,ValueError,TypeError,OSError,OverflowError,ZeroDivisionError,statistics.StatisticsError) as exc:
        errors.append('AUDIT_EXCEPTION:'+type(exc).__name__+':'+str(exc))
    # Prevent a terminal receipt silently changing underneath an expensive replay.
    check(sha(raw_path) == raw_receipt, 'RAW_UNCHANGED_AFTER_READ_ONLY_AUDIT')
    return dict(label=job['label'],role=job['role'],seed=job['seed'],status='AUDITED',
                verified=not errors, raw=raw_receipt, checks=checks, errors=errors,
                diagnostics=diagnostic, GPU_started=False, physical_full_pool_optimality=False)


def aggregate(packet, observations):
    pairs = []
    by_label = {row['label']:row for row in observations}
    for on, off in [('coupled_spill','matched_no_spill'),('legacy_latency','legacy_no_spill')]:
        if not any(j['role'] == on for j in packet['jobs']):
            continue
        for seed in packet['seeds']:
            a,b = by_label.get(f'{on}-seed{seed}'),by_label.get(f'{off}-seed{seed}')
            da,db = (a or {}).get('diagnostics',{}),(b or {}).get('diagnostics',{})
            same_cohort = (da['cohort_signature'] == db['cohort_signature']
                           if da.get('cohort_signature') and db.get('cohort_signature') else None)
            valid = bool(a and b and a.get('verified') and b.get('verified') and same_cohort
                         and da.get('eligible') and db.get('eligible') and finite(da.get('recorded_U')) and finite(db.get('recorded_U')))
            deltas = {t:da['terms'][t]-db['terms'][t] for t in TERMS} if valid else None
            pairs.append(dict(strategy=on,seed=seed,same_cohort=same_cohort,both_evidence_valid_eligible=valid,
                              delta_U=da['recorded_U']-db['recorded_U'] if valid else None,
                              term_deltas_spill_minus_control=deltas,
                              contribution_to_delta_U={t:-WEIGHTS[t]*v for t,v in deltas.items()} if deltas else None,
                              actual_text_spills=da.get('mechanism',{}).get('actual_text_spills'),
                              control_text_spills=db.get('mechanism',{}).get('actual_text_spills')))
    primary = [p for p in pairs if p['strategy'] == 'coupled_spill']
    all_complete = len(observations) == len(packet['jobs']) and all(r.get('status') == 'AUDITED' for r in observations)
    all_valid = all_complete and all(r.get('verified') and r.get('diagnostics',{}).get('eligible') for r in observations)
    development_gate = (packet['phase']=='development' and len(packet['jobs'])==8 and all_valid and len(primary)==2
                        and all(p['both_evidence_valid_eligible'] and p['delta_U']>.0001
                                and p['actual_text_spills']>0 and p['control_text_spills']==0 for p in primary))
    confirmation = (packet['phase']=='independent_confirmation' and len(packet['jobs'])==12 and all_valid and len(primary)==6
                    and all(p['both_evidence_valid_eligible'] and p['delta_U']>0 and p['actual_text_spills']>0
                            and p['control_text_spills']==0 for p in primary))
    return dict(expected_jobs=len(packet['jobs']), terminal_jobs=sum(r.get('status')=='AUDITED' for r in observations),
                all_expected_jobs_terminal=all_complete, all_evidence_valid_eligible=all_valid, paired=pairs,
                development_gate_passed=development_gate, all_six_confirmation_pairs_positive=confirmation,
                physical_full_pool_optimality=False,
                scope='Fixed primary resident and declared strategy comparisons only; full mixed-workload CPU winner may remain UNKNOWN')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--execution',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--job',help='Optional one label; aggregate stays pending until all declared jobs are audited')
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite an audit: '+str(args.output))
    packet=json.loads(args.packet.read_bytes())
    binding_check=bindings(packet)
    jobs=[j for j in packet['jobs'] if not args.job or j['label']==args.job]
    if not jobs:
        raise SystemExit('Requested job not present in packet')
    observations=[]
    for job in jobs:
        # study.execute creates this marker only after execute_job has closed run.json.
        result=audit_run(packet,job,args.execution/job['label']/'run.json',
                         terminal_marker=args.execution/(job['label']+'-interim-analysis.json'))
        observations.append(result)
        print(json.dumps(dict(label=job['label'],status=result['status'],verified=result.get('verified'),
                              errors=result.get('errors')),ensure_ascii=False),flush=True)
    report=dict(schema='gsdeploy.five_term_external_runtime_audit.v1',packet=sha(args.packet),audit_source=sha(__file__),
                source_input_bindings=binding_check,observations=observations,summary=aggregate(packet,observations),
                verified_so_far=binding_check['verified'] and all(r.get('verified',r['status']=='PENDING') for r in observations),
                GPU_started=False, frozen_sources_modified=False)
    report['complete'] = report['summary']['all_expected_jobs_terminal']
    report['verified'] = report['complete'] and report['verified_so_far']
    args.output.mkdir(parents=True)
    (args.output/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines=['# 五项在线目标原始运行审计','',
           '本审计读取已完成的原始收据，独立重算各项归一化、功率积分与总效用，并调用冻结新dispatcher做因果重放。'
           'P只作诊断；不将主驻留下的有限比较称为七模型池全局最优。','',
           f"终结记录：{report['summary']['terminal_jobs']}/{report['summary']['expected_jobs']}；"
           f"全部证据有效且eligible：{report['summary']['all_evidence_valid_eligible']}；"
           f"开发门禁：{report['summary']['development_gate_passed']}。",'',
           '| 角色/seed | 状态 | 证据审计 | U | 实际文本分流 |','|---|---|---|---:|---:|']
    for row in observations:
        diag=row.get('diagnostics',{})
        lines.append(f"| {row['label']} | {row['status']} | {row.get('verified')} | {diag.get('recorded_U')} | {diag.get('mechanism',{}).get('actual_text_spills')} |")
    lines+=['','队列机制实例只代表决策时的host可见队列。双专有等待且通用队列为空的实例单独列出；'
            '不能从队列实例推断物理GPU空闲或反事实延迟收益。尚未写完或未出现的文件保持PENDING，失败或缺失不替代为候选不可行。']
    (args.output/'README.zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report['summary'],ensure_ascii=False))


if __name__=='__main__':
    main()
