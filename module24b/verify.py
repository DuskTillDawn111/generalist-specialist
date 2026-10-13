"""Independent arithmetic and frozen-route/cohort checks on saved real data."""
from .common import *
from module21.trajectory import choose

def main():
    f=read(OUT/'frozen_high_manifest.json');verify(f.get('implementation',[])+[f['predictor']])
    checks=[];observations=read(OUT/'formal_observations.json') if (OUT/'formal_observations.json').exists() else []
    pre=read(OUT/'replica_colocation_preflight.json')
    observations=[pre['full_HIGH_observation'],*observations]
    assert sum(not o['run_id'].startswith('high_preflight') for o in observations)<=6
    for o in observations:
        r=read(o['raw']['path']);s=r['slot'];checks.append(dict(run_id=s['run_id'],SHA256_verified=ref(o['raw']['path'])['sha256']==o['raw']['sha256']))
        assert checks[-1]['SHA256_verified']
        if len(r['worker_identities'])==8:assert len({w['pid'] for w in r['worker_identities'].values()})==8
        ph=r.get('phases',{}).get('measurement')
        if not ph:
            assert o['U'] is None;continue
        trajectory=read(s['trajectories']['measurement']['path']);verify([s['trajectories']['measurement']]);rows=ph['request_records']
        assert len(rows)==len(trajectory['events'])
        for row,event in zip(rows,trajectory['events']):
            assert row['task']==event['task'] and row['pool_id']==event['item']['pool_id'] and row['sample_id']==event['item']['sample'].get('id')
            assert row['model']==choose(event,s['routing'],s['deployment'])
            assert abs((row['scheduled_time_ns']-ph['window']['start_time_ns'])/1e9-event['offset_seconds'])<=1e-9
        if not o['eligible']:
            assert o['U'] is None;continue
        assert all(row['status']=='completed' for row in rows)
        assert not o['hard_unstable'] and not r['errors'] and r['gpu_release']['verified']
        w=ph['window'];idle=max(np.mean([v['owned_used_mib'] for v in r[k]]) for k in ['postwarm_idle','postmeasurement_idle'])
        memory=f['safe_factor']*idle;assert memory<=BUDGET and o['sampled_peak_mib']<=BUDGET
        L=P=0.
        for task,q in f['rates'].items():
            latencies=[row['e2e_latency_ms'] for row in rows if row['task']==task];assert latencies
            for row in [row for row in rows if row['task']==task]:
                assert abs(row['e2e_latency_ms']-(row['completion_time_ns']-row['scheduled_time_ns'])/1e6)<1e-9
            L+=q/sum(f['rates'].values())*np.mean(latencies)/f['refs_ms'][task]
            P+=q/sum(f['rates'].values())*np.percentile(latencies,95)/f['refs_ms'][task]
        within=sum(row['completion_time_ns']<=w['end_time_ns'] for row in rows)
        T=1-min(1.,within/(sum(f['rates'].values())*120.))
        manual=-.25*T-.3*L-.3*P-.15*memory/BUDGET
        assert abs(manual-o['U'])<1e-10,(s['run_id'],manual,o['U'])
        checks[-1].update(frozen_routes_inputs_and_all_planned_requests_verified=True,manual_U=manual,recorded_U=o['U'],eligible=True)
    result=dict(passed=True,checks=checks,no_completion_subset_utility_used=True,formula_and_denominators_independently_recomputed=True,
        sources=[ref(ROOT/'module24b/verify.py')],historical_inputs_unchanged=True)
    verify(read(OUT/'read_only_input_inventory.json'));save('final_verification.json',result)
    print('INDEPENDENT_RECALCULATION_AND_FROZEN_COHORT_CHECKS_PASS',len(checks),flush=True)

if __name__=='__main__':main()
