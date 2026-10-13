"""Read-only scientific checks for the completed, unchanged frozen campaign."""
from .common import *
import numpy as np
from scipy.stats import t


def main():
    f = read(OUT / 'frozen_gpu_manifest_v2.json')
    observations = read(OUT / 'gpu_observations.json')
    status = read(OUT / 'campaign_status.json')
    slots = {s['run_id']: s for s in f['slots']}
    checks = {
        'original_campaign_completed': status['status'] == 'COMPLETED',
        'all_and_only_frozen_windows': len(observations) == len(slots) == 24
        and {o['run_id'] for o in observations} == set(slots),
        'software_environment_record_present': (OUT / 'software_environment_versions.json').is_file(),
    }
    manifest_ref = ref(OUT / 'frozen_gpu_manifest_v2.json')
    run_checks = []
    raw_by_id = {}
    for o in observations:
        s = slots[o['run_id']]
        raw = read(o['raw']['path'])
        raw_by_id[o['run_id']] = raw
        rc = dict(
            raw_SHA256_preserved=ref(o['raw']['path']) == o['raw'],
            actual_manifest_preserved=raw['frozen_manifest'] == manifest_ref,
            slot_equals_original_frozen_slot=raw['slot'] == s,
            actual_GPU_matches_frozen_GPU=all(raw['baseline_GPU'][k] == f['GPU'][k]
                for k in ['uuid', 'name', 'driver', 'total_mib']),
            valid_shutdown_and_release=raw['valid_session'] and raw['gpu_release']['verified'],
            eligibility_accounting=o['eligible'] == (o['technical_valid'] and o['stable']),
            legal_U_accounting=(o['U'] is not None) == o['eligible'],
            actual_checks_accounting=o['technical_valid'] == all(o['checks'].values()),
            observed_U_arithmetic=o['U'] is None or abs(o['U'] - o['components']['raw_arithmetic_U']) < 1e-12,
        )
        run_checks.append(dict(run_id=o['run_id'], checks=rc))
    checks['all_run_provenance_and_accounting'] = all(all(r['checks'].values()) for r in run_checks)
    contrasts = read(OUT / 'gpu_contrasts.json')
    by = {o['run_id']: o for o in observations}
    pair_checks = []
    for c in contrasts:
        deltas = []
        for rep in [1, 2, 3]:
            arms = {s['arm']: s for s in slots.values()
                if s['stage'] == 'INDEPENDENT' and s['group'] == c['group'] and s['repetition'] == rep}
            a, b = arms['ORACLE'], arms['COMPETITOR']
            ra, rb = raw_by_id[a['run_id']], raw_by_id[b['run_id']]
            # Compare what was actually offered, not just manifest file references.
            def offered(raw):
                return [(r['task'], r['sample_id'], r['frozen_offset_seconds'],
                         r['routing_uniform'], r['replica_uniform'])
                        for r in raw['phases']['measurement']['request_records']]
            pair_checks.append(dict(group=c['group'], repetition=rep,
                same_seed=a['seed'] == b['seed'], same_trace_refs=a['trajectories'] == b['trajectories'],
                same_actual_offered_cohort=offered(ra) == offered(rb)))
            if by[a['run_id']]['eligible'] and by[b['run_id']]['eligible']:
                deltas.append(by[a['run_id']]['U'] - by[b['run_id']]['U'])
        ss = c['delta_statistics']
        checks['paired_statistic_' + c['group']] = ss['n'] == len(deltas) == 3
        if len(deltas) == 3:
            mean = float(np.mean(deltas))
            half = float(t.ppf(.975, 2) * np.std(deltas, ddof=1) / np.sqrt(3))
            checks['paired_statistic_' + c['group']] &= (
                abs(mean - ss['mean']) < 1e-12
                and max(abs(a-b) for a,b in zip([mean-half,mean+half], ss['ci95'])) < 1e-12)
    checks['all_actual_independent_pairs_matched'] = all(
        all(r[k] for k in ['same_seed', 'same_trace_refs', 'same_actual_offered_cohort'])
        for r in pair_checks)
    gate = read(OUT / 'module34_gate.json')
    checks['gate_counts_current'] = gate['GPU']['observed_runs'] == len(observations)
    checks['no_full_pool_GPU_optimality_claim'] = (
        not gate['GPU']['full_pool_oracle'] and gate['flags']['GPU_ORACLE_INCOMPLETE']
        and not gate['flags']['MIXED_EMPIRICAL_OPTIMALITY_SUPPORTED'])
    save('continuation_scientific_audit.json', dict(passed=all(checks.values()), checks=checks,
        runs=run_checks, independent_pairs=pair_checks,
        scope='Final accounting and frozen paired-statistic verification; no algorithm changes or new GPU measurements.'))
    print('CONTINUATION_AUDIT', all(checks.values()), len(observations), flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
