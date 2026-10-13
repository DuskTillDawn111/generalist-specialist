import unittest,math
from .common import *
from .optimizer import PoissonUtility
from .trajectory import arrivals,choose
class Tests(unittest.TestCase):
 def setUp(self):
  f=read(OUT/'input_protocol_freeze.json');self.d=read(f['catalog']['path']);self.p=PoissonUtility(self.d,read(f['utility_references']['path'])['latency_reference_ms'],read(f['utility_definition']['path'])['weights'])
 def test_queue_mean_PK(self):
  x={m:1 for m in MODELS};z=[0,0,0];r=self.p.evaluate(x,RATES['HIGH'],z,True);s=self.d['models'][MMS]['service_time_ms']['lid']/1000;rho=12*s
  self.assertAlmostEqual(r['per_task']['lid']['mean_ms'],(s+12*s*s/(2*(1-rho)))*1000,places=7)
 def test_no_target_forcing_and_background_free(self):
  x={m:1 for m in MODELS};t,b=self.p.domain(x,RATES['LOW']);self.assertEqual(b,[(0,1)]*3)
 def test_unstable_cannot_have_U(self):
  x={G:1,MODELS[3]:1};r=self.p.evaluate(x,RATES['HIGH'],[1,1,1],True);self.assertFalse(r['eligible']);self.assertIsNone(r['U'])
 def test_poisson_independent_task_streams_and_pair_reproducibility(self):
  a=arrivals(RATES['HIGH'],120,21101,'measurement');b=arrivals(RATES['HIGH'],120,21101,'measurement');self.assertEqual(a,b)
  lid=[r['offset_seconds'] for r in a if r['task']=='lid'];gaps=[lid[i+1]-lid[i] for i in range(len(lid)-1)];self.assertGreater(max(gaps)-min(gaps),.2)
  solo=arrivals({'lid':12},120,21101,'measurement');self.assertEqual(lid,[r['offset_seconds'] for r in solo])
 def test_uniform_routing_and_replica(self):
  event={'task':'lid','routing_uniform':.2,'replica_uniform':.9};r=choose(event,{'lid':{MMS:12}},{MMS:2});self.assertEqual(r,MMS+'#1')
 def test_inherited_utility_unchanged(self):
  f=read(OUT/'input_protocol_freeze.json');u=read(f['utility_definition']['path']);self.assertEqual(u['weights'],{'Q':0.,'T':.25,'L':.3,'P':.3,'M':.15});self.assertEqual(ref(f['utility_definition']['path'])['sha256'],f['utility_definition']['sha256'])
if __name__=='__main__':unittest.main()
