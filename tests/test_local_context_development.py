"""Weights and positive model semantics without any experiment-result reads."""
import ast
from collections import Counter,defaultdict
from pathlib import Path
import statistics
import unittest

import numpy as np
from scipy.optimize import lsq_linear

PATH=Path(__file__).resolve().parents[1]/'tools/spillover_context_20261022/fit_local_context.py'
tree=ast.parse(PATH.read_text());ns=dict(Counter=Counter,defaultdict=defaultdict,np=np,statistics=statistics,lsq_linear=lsq_linear)
for name in ('aggregate','balanced_weights','fit','predict'):
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(PATH),'exec'),ns)


class LocalContextDevelopmentTests(unittest.TestCase):
    def test_request_repetition_does_not_increase_run_or_input_weight(self):
        def row(run,family,sid):return dict(run_id=run,family=family,task='text',n=1,sample_id=sid,features=[1.],baseline_ms=2.,log_service_ratio=.3)
        raw=[row('a','x','1')]*100+[row('a','x','2')]+[row('a','y','3')]+[row('b','x','4')]
        grouped=ns['aggregate'](raw);w=ns['balanced_weights'](grouped)
        byrun=defaultdict(float);bypath=defaultdict(float)
        for r,weight in zip(grouped,w):byrun[r['run_id']]+=weight;bypath[r['run_id'],r['family']]+=weight
        self.assertEqual(byrun,{'a':.5,'b':.5})
        self.assertEqual(bypath,{('a','x'):.25,('a','y'):.25,('b','x'):.5})
        self.assertEqual(w[0],w[1]);self.assertEqual(grouped[0]['response_repetitions'],100)

    def test_congestion_slopes_nonnegative_and_service_multiplier_positive(self):
        rows=[]
        for i in range(5):
            f=[0.]*12;f[3]=1.;f[7]=i/4
            rows.append(dict(run_id=str(i),family='text',task='text',n=1,sample_id='a',features=f,y=-i))
        model=ns['fit'](rows,[3,7]);self.assertTrue(model['success'])
        self.assertGreaterEqual(model['coefficients'][1],0)
        pred=ns['predict'](rows,[3,7],model)
        self.assertTrue(np.all(np.exp(pred)>0))


if __name__=='__main__':unittest.main()
