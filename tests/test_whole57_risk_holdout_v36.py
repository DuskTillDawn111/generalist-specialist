import ast
import copy
import inspect
import unittest

from tools.predictor_validation_20261027 import whole57_risk_holdout_v36 as holdout


class Whole57RiskHoldoutTests(unittest.TestCase):
    def test_derived_functions_change_only_exhaustive_evidence_constants(self):
        namespace,proof=holdout.build_execution_namespace()
        self.assertEqual({row['function'] for row in proof},set(holdout.FUNCTIONS))
        for row in proof:
            for change in row['declared_changes']:
                if change['kind']=='string_constant':
                    self.assertEqual(holdout.STRING_MAP[change['original']],change['replacement'])
                else:
                    self.assertEqual(change['original'],'training_only=True')
                    self.assertEqual(change['replacement'],'training_only=False')
            transform=holdout.EvidenceOnlyTransform()
            expected=transform.visit(ast.parse(inspect.getsource(getattr(holdout.original,row['function']))))
            actual=ast.parse(row['derived_body'])
            self.assertEqual(ast.dump(expected,include_attributes=False),ast.dump(actual,include_attributes=False))
        unchanged=next(row for row in proof if row['function']=='clock_anchor')
        self.assertEqual(unchanged['declared_changes'],[])
        self.assertEqual(unchanged['original_AST_SHA256'],unchanged['derived_AST_SHA256'])
        self.assertEqual(namespace['RAW_SCHEMA'],holdout.RAW_SCHEMA)
        self.assertIs(namespace['audit'],holdout.audit)

    def test_offer_retains_every_original_arrival_wait_route_and_evidence_role(self):
        namespace,_=holdout.build_execution_namespace()
        def record(kind,index,task,alias,pool,when):return dict(kind=kind,index=index,task=task,alias=alias,scheduled_time_ns=when)
        namespace['record']=record
        calls=[];waits=[]
        class Runtime:
            def enqueue(self,row,sample):calls.append((copy.deepcopy(row),sample))
        plan=dict(trace=[dict(request_id='held:req000',task='text',pool_id='x',offset_ns=0),
            dict(request_id='held:req001',task='ocr',pool_id='y',offset_ns=17)],routes={'held:req000':'text#1','held:req001':'ocr#0'})
        pool={'x':dict(sample='original-text'),'y':dict(sample='original-image')};raw={}
        rows=namespace['register_and_offer'](raw,Runtime(),plan,pool,None,start_ns=100,wait=waits.append)
        self.assertEqual(waits,[100,117,100+30_000_000_000])
        self.assertEqual([r['request_id'] for r in rows],['held:req000','held:req001'])
        self.assertEqual([r['alias'] for r in rows],['text#1','ocr#0'])
        self.assertEqual([sample for row,sample in calls],['original-text','original-image'])
        self.assertTrue(all(row['split']=='risk_validation' and row['validation_view']=='INDEPENDENT_RISK_HOLDOUT_NOT_FOR_TRAINING' for row in rows))
        self.assertEqual(raw['request_records'],rows)

    def test_explicit_context_mapping_preserves_real_execution_differences(self):
        base=dict(collection_schema=holdout.original.RAW_SCHEMA,execution_nonce='train',whole57_training_route_id='balanced',
            training_only=True,execution=dict(batch_size=1,admission='uncapped'),execution_sources=[],sources=[])
        target=copy.deepcopy(base);target.update(collection_schema=holdout.RAW_SCHEMA,execution_nonce='held',training_only=False)
        self.assertEqual(holdout.execution_projection(base),holdout.execution_projection(target))
        target['execution']['batch_size']=2
        self.assertNotEqual(holdout.execution_projection(base),holdout.execution_projection(target))

    def test_execution_launcher_selects_explicit_validation_module_with_same_natural_command(self):
        namespace,_=holdout.build_execution_namespace()
        command=namespace['outer_command']('/declared/holdout',dict(controller_python='/python'),holdout.original.NATURAL)
        self.assertEqual(command,['/python','-m','tools.predictor_validation_20261027.whole57_risk_holdout_v36','cohort','--run','/declared/holdout'])


if __name__=='__main__':unittest.main()
