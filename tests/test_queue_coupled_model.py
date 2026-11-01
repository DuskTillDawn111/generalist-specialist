import unittest
from tools.queue_coupled_utility_20261023.model import CoupledEventModel,UnknownEvidence


class Provider:
    def service(self,row,active,aliases,scenario):
        base=100. if aliases[row['alias']]['family']=='text' else 50.
        return dict(known=True,context_supported=True,service_ms=base*len(active))
    def residual(self,row,age,active,aliases,scenario):
        return dict(known=age<500,tail_supported=age<500,remaining_ms=50.)
    def power(self,active,aliases,scenario):return dict(known=True,watts=100*len(active)+100 if active else 50.)

class CoupledModelTests(unittest.TestCase):
    def aliases(self):return {'t':dict(family='text',supported_tasks=['text']),'g':dict(family='generalist_2',supported_tasks=['text'])}
    def request(self,rid):return dict(request_id=rid,task='text',pool_id='x',arrival_ns=0,head_age_ms=0)
    def test_every_active_head_repriced_after_another_finishes(self):
        result=CoupledEventModel(self.aliases(),Provider()).forecast({'t':[self.request('a')],'g':[self.request('b')]},0,1_000_000_000)
        self.assertAlmostEqual(result['finishes']['b']['finish_ns'],100_000_000)
        self.assertAlmostEqual(result['finishes']['a']['finish_ns'],150_000_000)
        self.assertEqual(result['transitions'][0]['service_ms'],{'t':200.,'g':100.})
        self.assertEqual(result['transitions'][1]['service_ms'],{'t':100.})
        # Whole-board, not per-alias: .1s*300W+.05s*200W+.85s*50W.
        self.assertAlmostEqual(result['board_energy_joules'],82.5)
    def test_future_service_after_deadline_does_not_add_measurement_energy(self):
        result=CoupledEventModel(self.aliases(),Provider()).forecast({'t':[self.request('a')],'g':[self.request('b')]},0,50_000_000)
        self.assertAlmostEqual(result['board_energy_joules'],15.)
        self.assertGreater(result['drain_finish_ns'],50_000_000)
    def test_unsupported_tail_context_and_power_stay_unknown(self):
        row=self.request('a');row['head_age_ms']=600.
        with self.assertRaisesRegex(UnknownEvidence,'RESIDUAL_TAIL'):
            CoupledEventModel(self.aliases(),Provider()).forecast({'t':[row]},0,1000000000)
        class Missing(Provider):
            def power(self,*args):return dict(known=False,watts=None)
        with self.assertRaisesRegex(UnknownEvidence,'POWER_CONTEXT'):
            CoupledEventModel(self.aliases(),Missing()).forecast({'t':[self.request('a')]},0,1000000000)
        class Uncovered(Provider):
            def service(self,*args):return dict(known=True,context_supported=False,service_ms=1.)
        with self.assertRaisesRegex(UnknownEvidence,'SERVICE_CONTEXT'):
            CoupledEventModel(self.aliases(),Uncovered()).forecast({'t':[self.request('a')]},0,1000000000)
    def test_fifo_and_no_future_arrivals(self):
        a=self.request('a');b=self.request('b')
        result=CoupledEventModel(self.aliases(),Provider()).forecast({'t':[a,b]},0,1000000000)
        self.assertEqual(result['finishes']['b']['finish_ns'],200000000.)
        a['arrival_ns']=1
        with self.assertRaisesRegex(ValueError,'Future'):
            CoupledEventModel(self.aliases(),Provider()).forecast({'t':[a]},0,1000000000)

if __name__=='__main__':unittest.main()
