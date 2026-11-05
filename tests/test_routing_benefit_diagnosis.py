"""Independent checks of diagnostic observation, not invented research data."""

import copy
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

from gsdeploy.search_session import SearchSession
from tests.test_calibrated_replay_v2 import fitted_replay_fixture
from experiments.routing_benefit_diagnosis import ROOT, nominal_events, reconcile_events, split_route


class RoutingBenefitDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=replace(fitted_replay_fixture(),horizon_seconds=30.)

    def test_observer_matches_original_fifo_with_multiple_replicas(self):
        with SearchSession(ROOT,self.config.as_dict()) as session:
            x={'generalist':2,'specialist':1}
            route={t:{'generalist':q*.371,'specialist':q*.629} for t,q in self.config.workload.items()}
            p=session.predict(x,route)
            ledger=nominal_events(session,x,route)
            reconcile_events(session.config,p,ledger)
            self.assertEqual(sum(s['requests'] for s in ledger['per_worker'].values()),p['offered_requests'])
            self.assertTrue(all('sample_id' in e for e in ledger['events']))
            altered=copy.deepcopy(ledger);altered['per_task']['text']['p95_ms']+=10
            with self.assertRaises(AssertionError): reconcile_events(session.config,p,altered)

    def test_idle_resident_is_observed_without_unloading_or_fake_latency(self):
        with SearchSession(ROOT,self.config.as_dict()) as session:
            route={t:{'generalist':q} for t,q in self.config.workload.items()}
            x={'generalist':1,'specialist':1}
            a=nominal_events(session,x,route)
            self.assertEqual(a['per_model']['specialist']['requests'],0)
            self.assertIsNone(a['per_model']['specialist']['mean_waiting_ms'])
            self.assertIn('specialist',a['pool_service_ms'])
            self.assertGreater(session.predict(x,route)['safe_memory_mib'],
                               session.predict({'generalist':1},route)['safe_memory_mib'])

    def test_ratio_endpoints_conserve_flow_and_do_not_mutate_reference(self):
        c=self.config
        ref={t:{'generalist':q} for t,q in c.workload.items()}
        before=copy.deepcopy(ref)
        for r in (0.,1e-9,.371,1.):
            route=split_route(c,ref,'text','generalist','specialist',r)
            self.assertAlmostEqual(sum(route['text'].values()),c.workload['text'])
            self.assertTrue(all(q>0 for q in route['text'].values()))
        self.assertEqual(ref,before)
        with self.assertRaises(ValueError):split_route(c,ref,'text','generalist','specialist',1.1)


if __name__=='__main__': unittest.main()
