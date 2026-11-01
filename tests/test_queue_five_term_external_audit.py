"""Meaningful audit checks: missing arms cannot pass; witness and power scope."""
import json
from pathlib import Path
import tempfile
import unittest

from tools import queue_five_term_audit_20261023 as audit


class AuditTests(unittest.TestCase):
    def test_partial_or_unmarked_raw_is_pending(self):
        with tempfile.TemporaryDirectory() as root:
            raw=Path(root)/'run.json';raw.write_text('{"status":"completed",')
            self.assertEqual(audit.stable_json(raw)[2],'INCOMPLETE_JSON_PENDING')
            job=dict(label='arm',role='coupled_spill',seed=401)
            self.assertEqual(audit.audit_run({},job,raw,terminal_marker=Path(root)/'done')['status'],'PENDING')

    def test_independent_energy_requires_coverage_and_original_maximum_gap(self):
        samples=[dict(time_ns=0,watts=100.),dict(time_ns=1_000_000_000,watts=200.),dict(time_ns=2_000_000_000,watts=100.)]
        self.assertAlmostEqual(audit.integrate(samples,500_000_000,1_500_000_000),175.)
        self.assertIsNone(audit.integrate(samples,-1,1_000_000_000))
        self.assertIsNone(audit.integrate([samples[0],dict(time_ns=3_000_000_000,watts=100.)],0,3_000_000_000))

    def test_dual_busy_witness_does_not_imply_dual_waiting(self):
        raw=dict(replica_to_family={'s0':'text','s1':'text','g':'generalist_2'},request_records=[
            dict(request_id='r',task='text',model='g',pool_id='p',routing_decision=dict(
                snapshot={'s0':{'outstanding':[{}]},'s1':{'outstanding':[{}]},'g':{'outstanding':[]}},
                decision={'trigger':'spill'}))])
        result=audit.mechanisms(raw)
        self.assertEqual(result['actual_text_spills'],1)
        self.assertEqual(len(result['dual_specialist_outstanding_generalist_empty_examples']),1)
        self.assertEqual(result['dual_specialist_waiting_generalist_empty_examples'],[])
        for a in ['s0','s1']:
            raw['request_records'][0]['routing_decision']['snapshot'][a]['outstanding'].append({})
        self.assertEqual(len(audit.mechanisms(raw)['dual_specialist_waiting_generalist_empty_examples']),1)

    def test_positive_primary_pairs_alone_cannot_pass_all_eight_evidence_gate(self):
        roles=['coupled_spill','matched_no_spill','legacy_latency','legacy_no_spill']
        packet=dict(phase='development',seeds=[401,402],jobs=[dict(role=r) for r in roles for _ in range(2)])
        observations=[]
        for seed in packet['seeds']:
            for role in roles[:2]:
                spill=role=='coupled_spill'
                observations.append(dict(label=f'{role}-seed{seed}',status='AUDITED',verified=True,
                    diagnostics=dict(cohort_signature=str(seed),eligible=True,recorded_U=-.4 if spill else -.5,
                                     terms={t:.1 if spill else .2 for t in audit.TERMS},
                                     mechanism={'actual_text_spills':3 if spill else 0})))
        report=audit.aggregate(packet,observations)
        self.assertTrue(all(p['delta_U']>0 for p in report['paired'][:2]))
        self.assertFalse(report['development_gate_passed'])
        self.assertFalse(report['all_expected_jobs_terminal'])
        observations[0]['diagnostics']['cohort_signature']='different-arrivals'
        self.assertIsNone(audit.aggregate(packet,observations)['paired'][0]['delta_U'])


if __name__=='__main__':
    unittest.main()
