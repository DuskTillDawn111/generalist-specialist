import unittest
from gsdeploy.runtime.schema import MeasurementConfig
from gsdeploy.trace import make_trace

class IdleResidencyTests(unittest.TestCase):
 def test_zero_demand_requires_explicit_idle_mode(self):
  config=MeasurementConfig({'g':1},{'image':0},{'image':{}})
  with self.assertRaisesRegex(ValueError,'nonzero workload'):config.validate({'g':['image']})
  config.idle_residency=True
  config.validate({'g':['image']})
  config.workload_rps={'image':1};config.routing_rps={'image':{'g':1}}
  with self.assertRaisesRegex(ValueError,'positive demand'):config.validate({'g':['image']})
 def test_zero_rate_tasks_create_no_arrivals_in_both_trace_modes(self):
  for mode in ['deterministic_periodic','poisson']:
   self.assertEqual(make_trace({'image':0},1,42,mode),())
   self.assertEqual(make_trace({'image':5,'asr':0},1,42,mode),make_trace({'image':5},1,42,mode))
  for value in [-1,True,float('nan')]:
   with self.assertRaises(ValueError):make_trace({'image':value},1)
