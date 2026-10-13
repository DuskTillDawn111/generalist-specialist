"""Scientific invariants and known analytic CPU queue cases."""
import unittest
from .optimizer import predict,components,Generator
D={'safe_factor':1.1,'models':{'g':{'role':'generalist','supported_tasks':['text'],'service_time_ms':{'text':1000}},'s':{'role':'specialist','supported_tasks':['text'],'service_time_ms':{'text':100}}},'capacity_factors':{'g':{'0':0.,'1':1.},'s':{'0':0.,'1':1.,'2':1.5}},'memory_table_mib':{'g':{'0':0.,'1':100.},'s':{'0':0.,'1':10.,'2':20.}}}
W={'Q':0,'T':.25,'L':.3,'P':.3,'M':.15}
class Invariants(unittest.TestCase):
 def test_underloaded_periodic_latency(self):
  p=predict(D,{'s':1},{'text':{'s':2.}},{'text':2.},200,{'text':100},W)
  self.assertTrue(p['eligible']);self.assertAlmostEqual(p['per_task']['text']['mean_ms'],100,places=7)
 def test_unstable_is_never_sortable(self):
  p=predict(D,{'g':1},{'text':{'g':2.}},{'text':2.},200,{'text':100},W)
  self.assertFalse(p['eligible']);self.assertIsNone(p['U']);self.assertGreater(p['T_norm'],0)
 def test_quality_is_missing_not_imputed(self):
  p=components({'text':{'mean_ms':50,'p95_ms':100}},{'text':1},{'text':100},.2,.5,W)
  self.assertIsNone(p['Q_norm']);self.assertEqual(p['Q_contribution'],0);self.assertAlmostEqual(p['U'],sum(p[k+'_contribution'] for k in ['Q','T','L','P','M']))
 def test_no_task_completion_reweighting(self):
  stats={'a':{'mean_ms':100,'p95_ms':200},'b':{'mean_ms':1000,'p95_ms':2000}}
  p=components(stats,{'a':9,'b':1},{'a':100,'b':100},0,0,W)
  self.assertAlmostEqual(p['L_norm'],1.9);self.assertAlmostEqual(p['P_norm'],3.8)
 def test_illegal_capacity_or_memory_not_accepted(self):
  g=Generator(D,{'text':100});r=g.solve({'text':100},5,'text',0,.8,W)
  self.assertNotIn('deployment',r);self.assertEqual(r['status'],2)
 def test_replica_gain_changes_aggregate_capacity(self):
  p=predict(D,{'s':2},{'text':{'s':12.}},{'text':12.},200,{'text':100},W)
  self.assertAlmostEqual(p['rho']['s'],.8);self.assertTrue(p['eligible'])
 def test_route_conservation_is_hard(self):
  p=predict(D,{'s':1},{'text':{'s':1.}},{'text':2.},200,{'text':100},W)
  self.assertFalse(p['eligible']);self.assertIsNone(p['U'])
if __name__=='__main__':unittest.main()
