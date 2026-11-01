"""Alias-cell repetition weights and group identity without result-file reads."""
import ast
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import unittest

import numpy as np
from scipy.optimize import lsq_linear

PATH=Path(__file__).resolve().parents[1]/'tools/spillover_context_20261022/fit_alias_context.py'
tree=ast.parse(PATH.read_text());ns=dict(np=np,lsq_linear=lsq_linear,Counter=Counter,defaultdict=defaultdict,statistics=statistics,PRESSURES={7,8,12,13})
for name in ('aggregate','weights','fit'):
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(PATH),'exec'),ns)


class AliasContextDevelopmentTests(unittest.TestCase):
    def test_two_alias_cells_share_input_weight_without_repetition_inflation(self):
        def record(run,alias,sid,family='text'):
            return dict(run_id=run,family=family,task='text',n=2,alias=alias,sample_id=sid,deployment_key='fixed',
                alias_features=[1.]*14,baseline_ms=2.,log_service_ratio=.2,service_ms=3.)
        raw=[record('a','text#0','x')]*100+[record('a','text#1','x')]+[record('a','text#0','y')]+[record('b','text#0','z')]
        rows=ns['aggregate'](raw);w=ns['weights'](rows);byrun=defaultdict(float);byinput=defaultdict(float)
        for r,v in zip(rows,w):byrun[r['run_id']]+=v;byinput[r['run_id'],r['sample_id']]+=v
        self.assertEqual(dict(byrun),{'a':.5,'b':.5})
        self.assertEqual(dict(byinput),{('a','x'):.25,('a','y'):.25,('b','z'):.5})
        self.assertEqual(len(rows),4);self.assertEqual(rows[0]['response_repetitions'],100)

    def test_alias_partner_collinearity_stays_unknown_without_ridge(self):
        rows=[]
        for i in range(4):
            fs=[0.]*14;fs[3]=1.;fs[12]=i;fs[13]=2*i
            rows.append(dict(run_id=str(i),family='text',task='text',n=2,alias='text#0',sample_id='x',features=fs,y=.1*i))
        model=ns['fit'](rows,[3,12,13])
        self.assertFalse(model['success']);self.assertEqual(model['rank'],2);self.assertIsNone(model['coefficients'])


if __name__=='__main__':unittest.main()
