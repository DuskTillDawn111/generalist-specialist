"""Make a bounded, scoped handoff from the existing sealed validation.

Read-only with respect to validation, model and captures. Never collects,
refits, or expands predictor coverage. General qualification stays unchanged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.continue_activation_aggregate_validation import assess_cells


def decide(validation, plans, policy):
    reports = validation['reports']
    expected = {p['run_id']: p for p in plans}
    if len(expected) != 2 or len(reports) != 2 or {r['run_id'] for r in reports} != set(expected):
        raise ValueError('Both declared validation runs are required exactly once')
    cells, reasons = [], []
    for report in reports:
        plan = expected[report['run_id']]
        local = assess_cells(report, plan, policy)
        if local != report['tolerance_reasons']:
            raise ValueError('Stored tolerance reasons differ from frozen assessment')
        reasons.extend(report['run_id']+':'+reason for reason in local)
        def belongs(reason, job_id):
            return reason.startswith((job_id+':', job_id+'/task:'))
        for job in plan['jobs']:
            job_reasons = [r for r in local if belongs(r, job['id'])
                           or not any(belongs(r, j['id']) for j in plan['jobs'])]
            cells.append(dict(run_id=report['run_id'], route_share=job['design']['generalist_share'],
                arrival_regime=job['design']['arrival_regime'],
                within_tolerances=not job_reasons, reasons=job_reasons,
                task_metrics={t: report['aggregate_metrics'].get(job['id']+'/task:'+t)
                              for t in ('asr', 'image', 'ocr', 'text')}))
    # Capture-integrity issues cannot be attributed to just one route.
    integrity = validation['execution_integrity_passed']
    if integrity and reasons != validation['reasons']:
        raise ValueError('Validation reasons do not match declared cells')
    if validation['known_input_history_accuracy_within_tolerances'] != (integrity and not reasons):
        raise ValueError('Validation pass flag is inconsistent')
    shares = sorted({c['route_share'] for c in cells})
    eligible = [s for s in shares if integrity and
                sum(c['route_share'] == s for c in cells) == 2 and
                all(c['within_tolerances'] for c in cells if c['route_share'] == s)]
    if not integrity:
        status, action = 'CAPTURE_INTEGRITY_REVIEW', 'Identify affected evidence; no automatic recollection'
    elif not reasons:
        status, action = 'SCOPED_PASS', 'Freeze current scope and proceed to bounded optimization experiments'
    elif eligible:
        status, action = 'SCOPED_RESTRICTED', 'Freeze reliable tested routes; compare only eligible routes and review decision impact of excluded cells'
    else:
        status, action = 'DECISION_IMPACT_REVIEW', 'List unreliable scenarios; inspect feasibility/ranking/selection impact before any targeted repair'
    return dict(schema='gsdeploy.scoped_predictor_release.v1', status=status, next_action=action,
        scoped_prediction_accepted=status in ('SCOPED_PASS', 'SCOPED_RESTRICTED'),
        tested_route_shares_eligible_in_both_histories=eligible, cells=cells,
        validation_reasons=validation['reasons'],
        supported_scope=dict(hardware='same-model RTX 4090', residents='G2 plus four specialists',
            replicas_per_model=1, input_domain='existing profiled pool',
            histories='declared stratified jitter / paired bursts at the sealed offered rate'),
        excluded=['unseen inputs', 'other replica counts or resident sets', 'other load levels',
                  'six-request OCR bursts', 'unmeasured routes as validated winners'],
        formal_utility_qualified=False, candidate_ranking_qualified=False,
        general_physical_predictor_qualified=False,
        restriction_is_post_validation=True,
        optimization_note='Same-residency routing comparison; additional route proposals require bounded prospective measured confirmation, not predictor retraining',
        automatic_extra_gpu_runs=0, automatic_refits=0)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-rule', type=Path, required=True)
    parser.add_argument('--wait-seconds', type=float, default=0.)
    args = parser.parse_args()
    base = args.base.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, obj):
        (args.output/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tracked = [base/'protocol.json', base/'fit-frozen.json', base/'policy-frozen.json',
               args.stop_rule.resolve(), Path(__file__).resolve(),
               ROOT/'experiments/continue_activation_aggregate_validation.py']
    identities = {str(p): sha(p) for p in tracked}
    save('handoff-plan.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        files=identities, wait_seconds=args.wait_seconds, extra_gpu_runs=0, automatic_refits=0))
    deadline = time.monotonic()+args.wait_seconds
    save('handoff-state.json', dict(status='waiting_for_existing_validation'))
    try:
        while True:
            state = load(base/'state.json')
            if state['status'] == 'known_input_history_validation_completed':
                break
            if state['status'] == 'known_input_history_validation_failed':
                save('handoff-state.json', dict(status='validation_execution_failed',
                    upstream=state, next_action='Review affected evidence; no automatic retry'))
                return
            if time.monotonic() >= deadline:
                save('handoff-state.json', dict(status='validation_still_pending',
                    next_action='Read existing validation when complete; no duplicate launch'))
                return
            time.sleep(min(30., deadline-time.monotonic()))
        if any(sha(Path(p)) != value for p, value in identities.items()):
            raise ValueError('Handoff source or frozen input changed')
        protocol = load(base/'protocol.json')
        for filename, field in [('fit-frozen.json', 'fit_sha256'), ('policy-frozen.json', 'policy_sha256'),
                                ('queue.json', 'queue_sha256')]:
            if sha(base/filename) != protocol[field]:
                raise ValueError('Frozen artifact changed: '+filename)
        for receipt in protocol['source_receipts']:
            if any(sha(Path(receipt[k])) != receipt['sha256'] for k in ('path', 'snapshot')):
                raise ValueError('Frozen executable changed')
        plans = []
        evidence = [base/'validation.json', base/'capture-audit.json']
        validation = load(base/'validation.json')
        if validation['protocol_sha256'] != sha(base/'protocol.json'):
            raise ValueError('Validation protocol mismatch')
        for directory, receipt in zip(protocol['captures'], protocol['predictions']):
            directory = Path(directory)
            for filename, field in [('plan.json', 'plan_sha256'), ('prediction-sealed.json', 'prediction_sha256')]:
                if sha(directory/filename) != receipt[field]:
                    raise ValueError('Sealed plan or prediction changed')
                evidence.append(directory/filename)
            plans.append(load(directory/'plan.json'))
        decision = decide(validation, plans, load(base/'policy-frozen.json'))
        decision['evidence_sha256'] = {str(p): sha(p) for p in evidence}
        decision['frozen_inputs_sha256'] = identities
        save('release-decision.json', decision)
        save('handoff-state.json', dict(status=decision['status'], next_action=decision['next_action']))
        print(json.dumps(dict(status=decision['status'], next_action=decision['next_action']), ensure_ascii=False))
    except BaseException as exc:
        save('handoff-state.json', dict(status='handoff_failed', error=type(exc).__name__+': '+str(exc)))
        raise


if __name__ == '__main__':
    main()
