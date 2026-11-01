"""Independent CPU design/packet audit; never starts workers or opens live raw.

This module intentionally does not import the mutable refined implementation.
It can freeze an old-evidence ledger, draft a seed register, and verify later
packets. A passing packet audit is not GPU admission or measured optimality.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
PHASE_SEEDS = {"calibration": [201, 202], "development": [203, 204],
               "independent_validation": [205, 206],
               "final_confirmation": [207, 208, 209]}
PHASE_ALIASES = {"independent-validation": "independent_validation",
                 "final-confirmation": "final_confirmation"}
CLAIMS = {"calibration": "NEW_REFINED_RANDOM_PROBE_CALIBRATION_NOT_VALIDATION",
          "development": "NEW_REFINED_DEVELOPMENT_NOT_INDEPENDENT_VALIDATION",
          "independent_validation": "INDEPENDENT_FIXED_SELECTION_NATIVE_VALIDATION",
          "final_confirmation": "FINAL_FIXED_SELECTION_NATIVE_CONFIRMATION"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def check_receipts(receipts):
    issues = []
    for bound in receipts:
        try:
            if receipt(bound["path"]) != bound:
                issues.append("FINGERPRINT_CHANGED:" + str(bound.get("path")))
        except (OSError, KeyError, TypeError) as exc:
            issues.append("FINGERPRINT_UNAVAILABLE:" + str(exc))
    return issues


def old_evidence_ledger():
    """Bind two completed workflows and their declared source/input receipts."""
    paths = [ROOT / "artifacts/queue-aware-study-20261023/native-joint-workflow-v1/execution-v2/progress.json",
             ROOT / "artifacts/static-spillover-study-20261022/search-top-two-prequential-development-v1/execution-v1/progress.json"]
    bindings = {}; terminals = []; issues = []
    def add(bound):
        previous = bindings.setdefault(bound["path"], bound)
        if previous != bound:
            issues.append("CONFLICTING_OLD_RECEIPT:" + bound["path"])
    def add_packet(bound):
        add(bound); packet = read(bound["path"])
        for field in ["inputs", "bindings"]:
            values = packet.get(field, [])
            if isinstance(values, list):
                for item in values: add(item)
        sources = packet.get("sources", {})
        if isinstance(sources, dict):
            for item in sources.values():
                if isinstance(item, dict) and "path" in item: add(item)
    for path in paths:
        state = read(path)
        if not state.get("status", "").startswith("COMPLETE"):
            raise ValueError("Old workflow is not terminal: " + str(path))
        add(receipt(path))
        for item in state.get("bindings", state.get("sources", [])): add(item)
        if "stages" in state:
            for stage in state["stages"]:
                if stage["status"] != "COMPLETE_AUDITED": raise ValueError("Old stage incomplete")
                add_packet(stage["packet"]); add(stage["report"])
                for job in stage["jobs"]:
                    add(job["raw"]); terminals.append(job["raw"])
            add(state["frozen_selection"])
            for item in state["reserved_phase_packets"]: add(item)
            add(state["result"])
        else:
            add_packet(state["packet"]); add(state["collection"]["queue"])
            for key in ("audit", "uncertainty"): add(state[key]["report"])
            for job in read(state["collection"]["queue"]["path"])["jobs"]:
                add(job["raw"]); terminals.append(job["raw"])
    issues += check_receipts(bindings.values())
    return dict(schema="gsdeploy.refined.old_evidence_ledger.v1", protected=list(bindings.values()),
        protected_terminal_count=len(terminals), terminal_receipts=terminals,
        issues=issues, valid=not issues, GPU_started=False)


def draft_design():
    return dict(schema="gsdeploy.refined.random_probe_design_review.v1",
        status="DRAFT_CPU_ONLY_NOT_EXECUTED", phase_seeds=PHASE_SEEDS, phase_claims=CLAIMS,
        calibration=dict(jobs=4, deployment=dict(text=2, generalist_2=1, ocr=1),
            duration_seconds=600, warmup_seconds=30,
            workload_rps=dict(text=1.85, image=.01, asr=.01, ocr=.01),
            conditions=["zero_probe", "random_probe"],
            order=[dict(seed=201, conditions=["zero_probe", "random_probe"]),
                   dict(seed=202, conditions=["random_probe", "zero_probe"])],
            probe_probability=1/16,
            guard=dict(schema="queue_aware_token_bucket_v1", task="text", family="generalist_2", rate_rps=.23125, burst=1),
            randomization="One dedicated deterministic PRNG coin per text arrival, including zero arm; seed/phase namespace frozen before execution. No pool_id, sample offset, reference, answer, input quality, or future queue/response controls the coin.",
            control="Same resident deployment, background routes, priming, phase reset, and queue-balanced specialist routing. Coin+available guard token sends probe to G2; rejected coin/token returns to the same specialist policy. No redraw, delay, drop, or reassignment of earlier requests.",
            expected_uncapped_probes_per_600s=600*1.85/16,
            calibration_objective="Identify within-context G2 text service and queue/interference behavior across diverse inputs; all original measured six utility terms retained. Probe treatment is a whole routing policy, not an isolated request-level treatment.",
            coverage_rule="Report probe successes, guard rejection, unique text pool IDs, per-ID counts and selected service distributions for each arm/seed. Missing or sparse evidence stays unknown; no adaptive run extension after inspecting outcomes.",
            no_absolute_quality_gate=True),
        admission=dict(rho_max=.8, budget_mib=49140, safe_factor=1.110854,
            load_rule="Keep1.85; any increased load requires a distinct predeclared packet and prospective original rho/memory admission. Observed route shares cannot substitute for the prior guard envelope.",
            original_stability_thresholds_unchanged=True,
            measured_energy_required=True, measured_memory_required=True),
        seed_isolation="Calibration201/202 may fit estimates; development203/204 may select; freeze candidates before validation205/206 and confirmation207/208/209. Never fit or reselect from reserved outcomes.",
        near_tie_rule="New isolated search should compute exact global max first, then collect every candidate within1e-12 of that same global max; deterministic smallest candidate ID breaks ties. Report exact max separately from chosen candidate mean. Pruning retains equality and every potential near-tie. Old frozen search remains unchanged.",
        scientific_scope="Four calibration runs are the minimal paired intervention. They do not certify all deployment contexts, input service tails, or sustainable capacity. Reusing a common historical input pool is not unseen-input generalization.",
        GPU_started=False)


def _phase(job):
    value = job.get("phase", job.get("identity", {}).get("phase"))
    return PHASE_ALIASES.get(value, value)


def _bound_receipts(packet):
    values = list(packet.get("inputs", []))
    sources = packet.get("sources", {})
    if isinstance(sources, dict):
        values += [value if isinstance(value, dict) else dict(path=key, sha256=value)
                   for key, value in sources.items()]
    else: values += sources
    return values


def _planned_input_coverage(packet, job):
    """Use frozen arrival/pool code; never import mutable refined dispatch."""
    if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
    from tools.spillover_capacity_20261022.exposure import frozen_api
    api = frozen_api(); base = read(packet["base_packet"]["path"])
    config = base["config"]; contract = job["policy_contract"]; seed = job["seed"]
    rates = dict(config["workload"]); rates["text"] = contract["text_rate"]
    trace = api["trace"].make_trace(rates, job["identity"]["horizon_seconds"], seed, job["identity"]["arrival_process"])
    pool_path = Path(base["inputs"]) / "runtime-inputs.jsonl"
    expected_pool = config["catalog"]["quality"]["identity"]["input_pool_sha256"]
    if receipt(pool_path)["sha256"] != expected_pool: raise ValueError("INPUT_POOL_CHANGED")
    pool = api["workload"].SamplePool(pool_path, list(rates), seed)
    counts = Counter(); ids = {}; cohort = []
    for offset, task in trace:
        item = pool.next(task); sid = item["pool_id"]
        counts[task] += 1; ids.setdefault(task, set()).add(sid)
        cohort.append([offset, task, sid])
    if set(counts) != set(rates): raise ValueError("MISSING_PLANNED_TASK_OBSERVATIONS")
    paths = {(m,t) for t,row in contract.get("background_routing", {}).items()
             for m,q in row.items() if q>0}
    strategy = contract.get("strategy", "")
    text_families = {s["family"] for s in contract["aliases"].values()
                     if "text" in s["supported_tasks"]}
    if strategy in {"no_split_swrr", "queue_specialist_only", "zero_probe"}:
        text_families &= {"text"}
    paths |= {(m,"text") for m in text_families}
    for family,task in paths:
        service = contract.get("service_ms", {}).get(family, {}).get(task, {})
        if not ids[task].issubset(service):
            raise ValueError("MISSING_POTENTIAL_INPUT_SERVICE:"+family+"/"+task)
        if not all(type(service[sid]) in (int,float) and math.isfinite(service[sid]) and service[sid]>0 for sid in ids[task]):
            raise ValueError("INVALID_POTENTIAL_INPUT_SERVICE:"+family+"/"+task)
        loss = contract.get("quality_loss", {}).get(family, {}).get(task)
        if type(loss) not in (int,float) or not math.isfinite(loss) or loss<0:
            raise ValueError("MISSING_POTENTIAL_AGGREGATE_QUALITY:"+family+"/"+task)
    return dict(task_counts=dict(counts), unique_inputs={t:len(v) for t,v in ids.items()},
        cohort_sha256=digest(cohort), pool=receipt(pool_path), potential_paths=sorted(paths),
        scope="Planned arrivals and task/input cohort only. Host queue evolution and realized probe/input exposure remain unknown until terminal audit.")


def audit_packet(packet_path, *, design=None, old_ledger=None, inspect_inputs=True):
    """Return issues instead of approving implementation by its own validator.

    Caller supplies its frozen declaration if claim names differ from defaults;
    phase_seeds must retain mutually disjoint pools. No terminal raw is opened.
    """
    packet = read(packet_path); design = design or draft_design(); issues = []
    phases = design["phase_seeds"]; claims = design.get("phase_claims", CLAIMS)
    source_package = packet.get("source_package", "tools.queue_refined_20261023")
    package_path = str(source_package).replace(".", "/").strip("/")
    allowed_packages = {"tools/queue_refined_20261023", "tools/queue_refined_calibration_20261023"}
    if package_path not in allowed_packages: issues.append("UNDECLARED_REFINED_SOURCE_PACKAGE")
    required_root = ROOT/package_path
    all_seeds = [s for seeds in phases.values() for s in seeds]
    if len(all_seeds) != len(set(all_seeds)): issues.append("PHASE_SEED_OVERLAP")
    issues += check_receipts(_bound_receipts(packet))
    jobs = packet.get("jobs", []); ids = set(); labels = set(); cohorts = {}; rows = []
    for job in jobs:
        local = []; identity = job["identity"]; contract = job["policy_contract"]
        phase = _phase(job); seed = job.get("seed")
        if phase not in phases or seed not in phases.get(phase, []): local.append("UNDECLARED_PHASE_OR_SEED")
        if identity.get("phase") is not None and _phase(dict(phase=identity["phase"])) != phase: local.append("PHASE_IDENTITY_MISMATCH")
        if identity.get("seed") != seed or identity.get("measurement_claim") != claims.get(phase): local.append("SEED_OR_PHASE_CLAIM_MISMATCH")
        if digest(identity) != job.get("id"): local.append("JOB_IDENTITY_HASH_MISMATCH")
        if digest(contract) != identity.get("policy_contract_sha256"): local.append("POLICY_CONTRACT_HASH_MISMATCH")
        if job.get("id") in ids or job.get("label") in labels: local.append("DUPLICATE_JOB_ID_OR_LABEL")
        ids.add(job.get("id")); labels.add(job.get("label"))
        if contract.get("absolute_quality_gate") is not False: local.append("ABSOLUTE_QUALITY_GATE_NOT_REMOVED")
        if identity.get("horizon_seconds") != 600 or identity.get("warmup_seconds") != 30: local.append("UNDECLARED_WINDOW")
        sources = contract.get("source_fingerprints", {})
        if not sources: local.append("MISSING_POLICY_SOURCES")
        local += check_receipts([dict(path=p, sha256=s) for p,s in sources.items()])
        if not any(str(required_root.resolve())+"/" in p for p in sources): local.append("NEW_REFINED_SOURCES_NOT_BOUND")
        for name in ["protocol.py", "native_runner.py", "runtime_window.py", "policy.py", "guarded_policy.py", "admission.py", "evaluation.py"]:
            if str((required_root/name).resolve()) not in sources: local.append("MISSING_REQUIRED_REFINED_SOURCE:"+name)
        if contract.get("probe"):
            if str((required_root/"probe_policy.py").resolve()) not in sources: local.append("MISSING_REFINED_PROBE_SOURCE")
            probe=contract["probe"]
            if probe.get("schema")!="randomized_text_probe_v1" or probe.get("probability") not in [0.,1/16] or probe.get("namespace")!="queue-refined-calibration-20261023":
                local.append("UNDECLARED_CALIBRATION_PROBE_CONTRACT")
            if phase!="calibration": local.append("CALIBRATION_PROBE_IN_OTHER_PHASE")
        if contract.get("deployment") != job.get("nominal_plan", {}).get("deployment"): local.append("RESIDENT_PLAN_MISMATCH")
        if identity.get("deployment", contract.get("deployment")) != contract.get("deployment"): local.append("IDENTITY_RESIDENT_MISMATCH")
        if contract.get("service_replica_states") != contract.get("deployment"): local.append("SERVICE_REPLICA_STATE_MISMATCH")
        coverage = None
        if inspect_inputs:
            try:
                coverage = _planned_input_coverage(packet, job)
                key = (phase, seed); previous = cohorts.setdefault(key, coverage["cohort_sha256"])
                if previous != coverage["cohort_sha256"]: local.append("PAIRED_COHORT_MISMATCH")
            except (OSError, KeyError, TypeError, ValueError) as exc: local.append("INPUT_COVERAGE_UNKNOWN:"+str(exc))
        rows.append(dict(label=job.get("label"), phase=phase, seed=seed,
            issues=local, planned_input_coverage=coverage))
        issues += [str(job.get("label"))+":"+v for v in local]
    if not jobs: issues.append("NO_DECLARED_JOBS")
    if set(packet.get("seeds", [])) != {j.get("seed") for j in jobs}: issues.append("PACKET_SEED_LIST_MISMATCH")
    if old_ledger is not None: issues += check_receipts(old_ledger["protected"])
    return dict(schema="gsdeploy.refined.independent_packet_audit.v1", packet=receipt(packet_path),
        valid=not issues, issues=issues, jobs=rows, phases=phases,
        old_evidence_checked=old_ledger is not None, source_package=source_package, raw_files_opened=0,
        GPU_started=False, scope="CPU source/identity/phase/cohort audit only; runtime admission, probe causal replay, measured resource and full utility checks remain required.")


def audit_calibration_terminal(packet_path, progress_path, job_id):
    """Open only a SHA-bound terminal already published in supervisor progress.

    Independently reconstruct the arrival/input cohort, dedicated probe coins
    and token arithmetic. Do not import the policy or trust its replay result.
    This audits random exposure, not the counterfactual service model or U.
    """
    packet=read(packet_path);state=read(progress_path)
    if state.get('packet')!=receipt(packet_path):raise ValueError('Packet changed')
    terminal=next(t for t in state['jobs'] if t['job_id']==job_id)
    if receipt(terminal['raw']['path'])!=terminal['raw']:raise ValueError('Terminal SHA changed')
    job=next(j for j in packet['jobs'] if j['id']==job_id)
    raw=read(terminal['raw']['path']);issues=[]
    if raw.get('experiment_job_id')!=job_id or raw.get('executed_measurement_identity')!=job['identity'] or raw.get('executed_measurement_id')!=job_id:
        issues.append('TERMINAL_IDENTITY_MISMATCH')
    if raw.get('policy_contract')!=job['policy_contract'] or raw.get('measurement_claim')!=job['identity']['measurement_claim']:
        issues.append('TERMINAL_CONTRACT_OR_CLAIM_MISMATCH')
    contract=job['policy_contract'];spec=contract['probe'];guard=contract['guard']
    base=read(packet['base_packet']['path']);rates=dict(base['config']['workload']);rates['text']=contract['text_rate']
    from tools.spillover_capacity_20261022.exposure import frozen_api
    api=frozen_api();rows=[]
    def equal(actual,expected):
        return type(actual) in (int,float) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=0.,abs_tol=1e-12)
    for phase,duration,field in [('warmup',30,'warmup_request_records'),('measurement',600,'request_records')]:
        records=raw.get(field,[]);window=raw.get(phase+'_window') or {}
        trace=api['trace'].make_trace(rates,duration,job['seed'],job['identity']['arrival_process'])
        pool=api['workload'].SamplePool(Path(base['inputs'])/'runtime-inputs.jsonl',list(rates),job['seed'])
        if len(records)!=len(trace):issues.append(phase+':COHORT_COUNT_MISMATCH')
        stream=hashlib.sha256(f"{spec['namespace']}:{job['seed']}:{phase}".encode()).hexdigest()
        rng=random.Random(int(stream,16));index=0;tokens=float(guard['burst']);last=None;consumed=0
        requested=Counter();selected=Counter();rejected=Counter();services={};cohort=[]
        for ordinal,((offset,task),record) in enumerate(zip(trace,records)):
            item=pool.next(task);pool_id=item['pool_id'];cohort.append([offset,task,pool_id])
            if record.get('task')!=task or record.get('pool_id')!=pool_id or record.get('request_id')!=f'{phase}:{ordinal}' or record.get('scheduled_time_ns')!=window.get('start_time_ns',0)+round(offset*1e9):
                issues.append(phase+':COHORT_ROW_MISMATCH:'+str(ordinal))
            if task!='text':continue
            index+=1;coin=rng.random();wants=coin<spec['probability'];decision=record.get('routing_decision',{}).get('decision',{})
            observed=decision.get('probe',{});rate=decision.get('rate_guard',{})
            expected=dict(spec=spec,stream_sha256=stream,draw_index=index,coin=coin,requested_probe=wants)
            if any(observed.get(k)!=v for k,v in expected.items()):issues.append(phase+':PROBE_STREAM_MISMATCH:'+str(index))
            now=record.get('routing_decision',{}).get('decision_time_ns')
            if type(now) is not int or (last is not None and now<last):
                issues.append(phase+':INVALID_GUARD_CLOCK');continue
            elapsed=0. if last is None else (now-last)/1e9;refill=elapsed*guard['rate_rps'];available=min(guard['burst'],tokens+refill);allowed=available>=1.
            alias=record.get('model');family=raw.get('replica_to_family',{}).get(alias);consume=family==guard['family']
            if decision.get('alias')!=alias or (consume and (not wants or not allowed)):
                issues.append(phase+':UNAUTHORIZED_PROBE_ROUTE:'+str(index))
            consumed+=int(consume)
            expected_numbers=dict(bucket_before_tokens=tokens,elapsed_seconds=elapsed,refill_tokens=refill,available_tokens=available,bucket_after_tokens=available-int(consume))
            if rate.get('spec')!=guard or rate.get('last_refill_ns')!=last or rate.get('allowed')!=allowed or rate.get('consume')!=consume or rate.get('consumed_requests')!=consumed or any(not equal(rate.get(k),v) for k,v in expected_numbers.items()):
                issues.append(phase+':GUARD_ARITHMETIC_MISMATCH:'+str(index))
            if rate.get('requests_dropped')!=0 or rate.get('requests_delayed_by_guard')!=0:
                issues.append(phase+':GUARD_DROPPED_OR_DELAYED')
            if wants:requested[pool_id]+=1
            if wants and not allowed:rejected[pool_id]+=1
            if consume:
                selected[pool_id]+=1
                service=record.get('worker_response',{}).get('service_ms')
                if type(service) not in (int,float) or not math.isfinite(service) or service<=0:
                    issues.append(phase+':MISSING_ACTUAL_PROBE_SERVICE')
                else:services.setdefault(pool_id,[]).append(service)
            tokens=available-int(consume);last=now
        observations=[v for values in services.values() for v in values]
        rows.append(dict(phase=phase,text_arrivals=index,cohort_sha256=digest(cohort),requested_probes=sum(requested.values()),
            guard_rejected_probes=sum(rejected.values()),realized_G2_text=sum(selected.values()),
            unique_requested_input_ids=len(requested),unique_realized_input_ids=len(selected),requested_per_input=dict(sorted(requested.items())),
            realized_per_input=dict(sorted(selected.items())),guard_rejected_per_input=dict(sorted(rejected.items())),
            realized_service_ms_per_input=dict(sorted(services.items())),
            realized_service_ms_summary=dict(n=len(observations),minimum=min(observations),median=statistics.median(observations),maximum=max(observations)) if observations else None))
    return dict(schema='gsdeploy.refined.independent_calibration_terminal_exposure.v1',valid=not issues,issues=issues,
        packet=receipt(packet_path),terminal=terminal,phases=rows,
        reported_native_utility=raw.get('evaluation',{}).get('utility'),
        scope='Independent cohort/probe/guard replay and factual selected-path service coverage. Reported native utility is copied, not independently certified here. No request-level independence or efficacy claim.',GPU_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--old-ledger", type=Path)
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    if args.draft:
        design = draft_design(); ledger = old_evidence_ledger()
        write(args.output / "draft-preregistration.json", design)
        write(args.output / "old-evidence-ledger.json", ledger)
        result = dict(status="DRAFT_CPU_ONLY", protected_terminals=ledger["protected_terminal_count"], old_evidence_valid=ledger["valid"], GPU_started=False)
    else:
        if args.packet is None: parser.error("--packet or --draft required")
        result = audit_packet(args.packet, design=read(args.design) if args.design else None,
            old_ledger=read(args.old_ledger) if args.old_ledger else None)
        write(args.output / "report.json", result)
    print(json.dumps(result if args.draft else {k:result[k] for k in ["valid", "issues", "raw_files_opened", "GPU_started"]}, ensure_ascii=False))
    return 0 if result.get("valid", result.get("old_evidence_valid", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
