"""Prior dynamic exposure bounds use original frozen proxy semantics only."""
import copy
import importlib
import json
from pathlib import Path
import unittest

from tools.spillover_capacity_20261022.exposure import frozen_api,SOURCE
from tools.queue_aware_20261023.admission import capacity_envelope,source_checks,digest,cover_plan
from tools.queue_aware_20261023.protocol import make_policy_contract,nominal_plan

ROOT=Path(__file__).resolve().parents[1]
GUARD=dict(schema='queue_aware_token_bucket_v1',task='text',family='generalist_2',rate_rps=1.85/8,burst=1.)

def fixture(text_only=False,strategy='queue_quality'):
    data=json.loads((ROOT/'artifacts/static-spillover-study-20261022/text185-fine-development-v1/system-config.json').read_text())
    if text_only:data.update(workload={'text':1.85},horizon_seconds=4.,random_seed=113,arrival_process='deterministic_periodic')
    Config=importlib.import_module(frozen_api()['trace'].__package__+'.config').SystemConfig;c=Config(**data)
    sim=json.loads((ROOT/'artifacts/queue-aware-study-20261023/cpu-closed-loop-v1/frozen-simulation-contract.json').read_text())
    sources={str(p.resolve()):digest(p) for p in (ROOT/'tools/queue_aware_20261023').glob('*.py')}
    sources.update({str(p.resolve()):digest(p) for p in SOURCE.rglob('*.py')})
    contract=make_policy_contract(sim['deployment'],sim['service_ms'],sim['aggregate_quality_loss'],c.weights,c.latency_reference_ms,
        strategy=strategy,models=c.catalog['models'],sources=sources,guard=GUARD,
        background_routing={} if text_only else None)
    return c,contract,nominal_plan(contract)

class PriorEnvelopeTests(unittest.TestCase):
    def test_guard_caps_include_all_text_to_specialist_and_fixed_background(self):
        c,p,w=fixture();r=capacity_envelope(c,w,p)
        self.assertTrue(r['constraints_verified'],r)
        self.assertEqual(r['rate_caps_by_family_task']['text']['text'],1.85)
        self.assertEqual(r['rate_caps_by_family_task']['generalist_2'],{'text':1.85/8,'image':.01,'asr':.01})
        for m,row in r['rate_caps_by_family_task'].items():
            expected=sum(q*c.catalog['task_replica_profiles'][m][t][str(w['deployment'][m])]['mean_service_ms']/1000 for t,q in row.items())/c.catalog['capacity_factors'][m][str(w['deployment'][m])]
            self.assertAlmostEqual(r['rho_by_family'][m],expected,places=12)
        self.assertFalse(r['observed_routing_share_used']);self.assertFalse(r['physical_stability_certified'])

    def test_no_guard_or_posthoc_share_does_not_admit_dynamic_path(self):
        c,p,w=fixture();p['guard']=None;p['observed_routing_share']={'generalist_2':0.0001}
        r=capacity_envelope(c,w,p);self.assertFalse(r['constraints_verified']);self.assertIn('GUARD',r['reason'])

    def test_static_and_specialist_control_coverage_remains_explicit(self):
        for strategy in ['no_split_swrr','fixed_1of32_swrr','queue_specialist_only']:
            c,p,w=fixture(strategy=strategy);r=capacity_envelope(c,w,p)
            self.assertTrue(r['constraints_verified'],r)
            cover=cover_plan(c,w,p)
            self.assertEqual('generalist_2' in cover['routing']['text'],strategy=='fixed_1of32_swrr')

    def test_source_drift_and_capacity_excess_are_not_unknown_success(self):
        c,p,w=fixture();ok,issues=source_checks(p);self.assertTrue(ok,issues)
        p['source_fingerprints'][str((ROOT/'tools/queue_aware_20261023/policy.py').resolve())]='0'*64
        self.assertFalse(source_checks(p)[0])
        c,p,w=fixture();p['guard']['rate_rps']=1.85
        self.assertFalse(capacity_envelope(c,w,p)['constraints_verified'])
        c.catalog['performance_mode']='route_load_replay_v1'
        self.assertEqual(capacity_envelope(c,w,p)['status'],'UNKNOWN')

if __name__=='__main__':unittest.main()
