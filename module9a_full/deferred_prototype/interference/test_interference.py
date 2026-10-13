"""Meaningful invariants: two-phase residual work, split determinism, routing conservation."""
import copy,math,unittest
import numpy as np
from .task_path_profile import reconstruct
from .occupancy import DeviceLimits
from .kernel_overlap import simulate_arrays
from .pairwise_predictor import feasible_alignments
from .routing_conditioned import InterferenceEstimator
from .validation import paired_model_bootstrap
class Tests(unittest.TestCase):
 def setUp(self):self.limits=DeviceLimits(48,24,65536,102400,128)
 def arr(self,d):return np.array([[v,24,32768,51200,2,2] for v in d],dtype=np.float64)
 def test_two_phase_survivor(self):
  # Half occupancy while shorter B runs: B finishes at 2ms, A has 9ms work left.
  delay=simulate_arrays(self.arr([10]),self.arr([1]),(0,0),self.limits,'EQUAL_PARTITION')
  self.assertAlmostEqual(delay,1)
 def test_serial_wait_and_end(self):
  self.assertEqual(simulate_arrays(self.arr([10]),self.arr([1]),(0,0),self.limits,first=0),0)
  self.assertEqual(simulate_arrays(self.arr([10]),self.arr([1]),(0,0),self.limits,first=1),1)
  self.assertEqual(simulate_arrays(self.arr([10,10]),self.arr([1]),(0,0),self.limits,first=0),1)
 def test_anchor(self):
  for size in [1,3,19001]:self.assertTrue(math.isclose(math.fsum(reconstruct(list(range(1,size+1)),654.0784572333333)),654.0784572333333,abs_tol=1e-10))
 def test_alignment_sampling(self):
  a=feasible_alignments(100,200,'pair',64);self.assertEqual(a,feasible_alignments(100,200,'pair',128)[:64]);self.assertEqual(len(set(a)),64)
 def estimator(self):return InterferenceEstimator({('a','t'):100,('b','u'):200},{'a':{0:0,1:1,2:1.5},'b':{0:0,1:1}},{(('a','t'),('b','u')):50,(('b','u'),('a','t')):20})
 def test_routing_replica_and_busy(self):
  e=self.estimator();r={'t':{'a':3},'u':{'b':2}};before=copy.deepcopy(r);v=e.evaluate(deployment={'a':2,'b':1},routing=r,workload={'t':3,'u':2})
  self.assertEqual(r,before);self.assertAlmostEqual(v['effective_service_time']['t']['a'],120);self.assertAlmostEqual(v['base_utilization']['a'],.2)
  self.assertEqual(v['task_mix']['a']['t'],1);self.assertTrue(v['SELF_INTERFERENCE_HANDLED_BY_G'])
 def test_zero_and_overload(self):
  v=self.estimator().evaluate(deployment={'a':1,'b':1},routing={'t':{'a':0},'u':{'b':20}},workload={'t':0,'u':20});self.assertEqual(v['task_mix']['a']['t'],0);self.assertEqual(v['effective_service_time']['t']['a'],150);self.assertTrue(v['overload_flags']['b'])
 def test_invalid_input(self):
  for x,r,l in [({'a':0},{'t':{'a':1}},{'t':1}),({'a':1},{'t':{'a':2}},{'t':1}),({'a':1},{'t':{'a':float('nan')}},{'t':1})]:
   with self.assertRaises(ValueError):self.estimator().evaluate(deployment=x,routing=r,workload=l)
 def test_model_bootstrap_direction(self):
  v=paired_model_bootstrap([1,2,3],[0,0,0],[1,2,3]);self.assertLess(v['CI95'][1],0)
if __name__=='__main__':unittest.main()
