import copy
import unittest
from unittest.mock import patch
import json
from pathlib import Path
import tempfile
from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.resources import identity
from tools.predictor_validation_20261027.dag_host_release_v7 import (
    SCHEMA,DagHostReleaseProvider,aggregate,binding_identity,FIELDS)
from tools.predictor_validation_20261027.dag_host_release_v5 import fit_observation
from tools.predictor_validation_20261027.cuda_dag_simulator_v6 import DagFullHorizonModel
from tests.test_dag_host_release_v4 import profile,natural,ALIASES,ROW


def fixture():
    p=profile();p['source_sqlite']=dict(path='training.sqlite',sha256='exact')
    binding=dict(alias='a',family='family',task='task',pool_id='input',input_content_sha256='public',
        input_tokens=414,protocol_sha256='protocol',effective_generation={'max_new_tokens':128},
        native_execution_projection_sha256='projection',original_light_context_sha256='light',
        original_natural_context_sha256='natural',template=dict(dag=dict(path='dag.json',sha256='dag'),
        alias='a',task='task',pool_id='input'),source_sqlite=p['source_sqlite'],work={'forward_count':5})
    obs=fit_observation(p,natural())
    rows=[dict(obs,source='raw1',request_id='one',binding_sha256=identity(binding)),
          dict(obs,source='raw2',request_id='two',binding_sha256=identity(binding))]
    doc=dict(schema=SCHEMA,execution_context_sha256='natural',dag=binding['template']['dag'],
             public_bindings=[binding],groups=aggregate(rows))
    return p,binding,rows,doc


def provider(p,b,doc,scenario='host_issue_depth_supply'):
    with patch('tools.predictor_validation_20261027.dag_host_release_v7.verify_model'):
        return DagHostReleaseProvider(doc,'natural',{('a','task','input'):p},ALIASES,scenario=scenario,
        allow_identification_hypothesis=True,allow_exact_input_conditioning_hypothesis=True,
        public_bindings={('a','task','input'):b})


class ExactInputHostTests(unittest.TestCase):
    def test_same_capture_rows_never_count_as_independent_support(self):
        p,b,rows,doc=fixture()
        rows[1]['source']='raw1'
        self.assertFalse(aggregate(rows)[0]['supported'])
        # A different input is a different group, not support for this one.
        rows[1].update(pool_id='other',source='raw2')
        self.assertTrue(all(not g['supported'] for g in aggregate(rows)))

    def test_exact_owner_and_input_are_separate_and_equal_capture_weighted(self):
        p,b,rows,doc=fixture()
        rows[0]['terminal_tail_ns']=1;rows[1]['terminal_tail_ns']=9
        rows.extend([dict(rows[0],request_id='repeat') for _ in range(3)])
        self.assertEqual(aggregate(rows)[0]['joint_mean']['terminal_tail_ns'],5)
        rows[1]['alias']='replica'
        self.assertTrue(all(not g['supported'] for g in aggregate(rows)))

    def test_actual_dag_path_and_both_partial_identification_shapes(self):
        p,b,rows,doc=fixture();values=[]
        for scenario in ('front_loaded_supply','host_issue_depth_supply'):
            release=provider(p,b,doc,scenario)
            result=DagFullHorizonModel(ALIASES,{('a','task','input'):p},release_provider=release,
                duration_provider=lambda n,r:n['duration_ns']).forecast([ROW],0,1000)
            self.assertEqual(result['finishes']['a']['handler_end_ns'],107)
            values.append(release(p['nodes'][0],ROW))
        self.assertNotEqual(*values)
        changed=DagFullHorizonModel(ALIASES,{('a','task','input'):p},release_provider=provider(p,b,doc),
            duration_provider=lambda n,r:n['duration_ns']*2).forecast([ROW],0,1000)
        self.assertGreater(changed['finishes']['a']['handler_end_ns'],107)

    def test_no_family_fallback_and_every_public_condition_is_bound(self):
        p,b,rows,doc=fixture();release=provider(p,b,doc)
        with self.assertRaisesRegex(UnknownEvidence,'NO_SUPPORTED_EXACT_INPUT'):
            release(p['nodes'][0],dict(ROW,pool_id='unseen'))
        for field in ('input_content_sha256','input_tokens','protocol_sha256','effective_generation',
                      'native_execution_projection_sha256','work','source_sqlite','template'):
            with self.subTest(field=field):
                changed=copy.deepcopy(b);changed[field]='poison'
                with self.assertRaises((ValueError,TypeError,AttributeError)):provider(p,changed,doc)

    def test_query_durations_and_absolute_issue_times_cannot_retarget_host_fit(self):
        p,b,rows,doc=fixture();release=provider(p,b,doc);expected=[release(n,ROW) for n in p['nodes']]
        poisoned=copy.deepcopy(p)
        for n in poisoned['nodes']:
            n.update(duration_ns=None,release_ns=-10**12,handler_end_ns=123)
            if 'issue_start_ns' in n:n['issue_start_ns']+=10**15;n['issue_end_ns']+=10**15
        new=provider(poisoned,b,doc)
        self.assertEqual(expected,[new(n,ROW) for n in poisoned['nodes']])
        poisoned['edges']=[]
        with self.assertRaises(ValueError):provider(poisoned,b,doc)(poisoned['nodes'][0],ROW)

    def test_nonfinite_parameter_and_forged_capture_support_stay_unknown(self):
        p,b,rows,doc=fixture()
        for field in FIELDS:
            changed=copy.deepcopy(doc);changed['groups'][0]['joint_mean'][field]=float('nan')
            with self.assertRaises(UnknownEvidence):provider(p,b,changed)(p['nodes'][0],ROW)
        changed=copy.deepcopy(doc);changed['groups'][0]['independent_capture_ids']=['raw1','raw1']
        with self.assertRaises(UnknownEvidence):provider(p,b,changed)(p['nodes'][0],ROW)

    def test_ocr_binding_keeps_no_token_or_generation_assumption(self):
        p,b,rows,doc=fixture();b.update(task='ocr',input_tokens=None,effective_generation=None,
                                       work={'kind':'NON_AUTOREGRESSIVE_OCR_SEPARATE'})
        self.assertEqual(binding_identity(b)['work'],{'kind':'NON_AUTOREGRESSIVE_OCR_SEPARATE'})
        b['input_tokens']=1
        with self.assertRaises(ValueError):binding_identity(b)


class SourceReconstructionTests(unittest.TestCase):
    def setUp(self):
        from tools.predictor_validation_20261027 import dag_host_release_v7 as tool
        self.tool=tool;self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        def write(name,value):
            path=self.root/name;path.write_text(json.dumps(value));return tool.receipt(path)
        self.write=write
        p,b,rows,doc=fixture()
        sqlite=self.root/'training.sqlite';sqlite.write_bytes(b'actual training identity')
        p['source_sqlite']=tool.receipt(sqlite)
        dag=write('dag.json',dict(schema='gsdeploy.request_cuda_dag_profiles.v2',training_only=True,
                                 execution_context_sha256='light',profiles=[p]))
        b['source_sqlite']=p['source_sqlite'];b['template']['dag']=dag
        self.binding=b
        catalog=write('catalog.json',{})
        context=write('context.json',{})
        captures=[];index=[]
        for i in range(2):
            row=dict(request_id='train'+str(i),model='a',task='task',pool_id='input',split='training',
                     status='completed',**natural())
            raw=write('raw'+str(i)+'.json',dict(status='completed',kind='measured',
                    execution_context_sha256='natural',request_records=[row]))
            review=write('review'+str(i)+'.json',dict(raw=raw,context=context,training_only=True))
            captures.append(dict(review=review,raw=raw,context=context))
            index.append(dict(run_id='raw-sha256:'+raw['sha256'],request_id=row['request_id'],task='task',pool_id='input'))
        comp=dict(target=dict(execution_context_sha256='natural'),
                  components=dict(lifecycle=dict(original_context_sha256='natural')))
        comp['composite_execution_sha256']=identity(comp)
        composite=write('composite.json',comp)
        oldindex=write('old-index.json',dict(schema='gsdeploy.independent_prediction_gate_training_index.v2',
                                          split='training',records=index))
        source=write('source.json',dict(schema='gsdeploy.dag_host_supply_training_hypothesis.v5',
            execution_context_sha256='natural',light_execution_context_sha256='light',dag=dag,composite=composite,
            captures=captures,sources=[tool.receipt(tool.__file__)],training_index=oldindex))
        self.source=source;self.captures=captures
        def loader(path,domain):
            report=json.loads(Path(path).read_text())
            raw=tool.checked(report['raw']);tool.checked(report['context'])
            return report,{'jobs':[{'inputs':{'a':'input'}}]},raw
        self.loadpatch=patch('tools.predictor_validation_20261027.kernel_domain_bridge_review_v3.load_report',side_effect=loader)
        self.bindpatch=patch.object(tool,'public_binding',return_value=(b,catalog))
        self.loadpatch.start();self.bindpatch.start()
        self.model=tool.fit(source['path'],self.root/'fit')

    def tearDown(self):
        self.bindpatch.stop();self.loadpatch.stop();self.tmp.cleanup()

    def test_reconstruction_accepts_unchanged_evidence_and_rejects_fitted_cell_mutations(self):
        self.assertIs(self.tool.verify_model(self.model),self.model)
        for field in ('mean','capture','observation','binding'):
            model=copy.deepcopy(self.model)
            if field=='mean':model['groups'][0]['joint_mean']['terminal_tail_ns']+=1
            if field=='capture':model['groups'][0]['independent_capture_ids']=['invented1','invented2']
            if field=='observation':model['observations'][0]['handler_ns']+=1
            if field=='binding':model['public_bindings'][0]['input_tokens']+=1
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'rederived'):
                self.tool.verify_model(model)

    def test_nested_raw_review_and_producer_edits_are_reverified(self):
        targets=[self.captures[0]['raw'],self.captures[0]['review']]
        for ref in targets:
            path=Path(ref['path']);old=path.read_bytes();path.write_bytes(old+b' ')
            try:
                with self.assertRaises(ValueError):self.tool.verify_model(self.model)
            finally:path.write_bytes(old)
        model=copy.deepcopy(self.model);model['sources'][0]['sha256']='forged'
        with self.assertRaisesRegex(ValueError,'source changed'):self.tool.verify_model(model)

    def test_refreshed_outer_receipt_cannot_legitimize_bad_training_index(self):
        model=copy.deepcopy(self.model)
        index=json.loads(Path(model['training_index']['path']).read_text());index['records'].pop()
        model['training_index']=self.write('forged-index.json',index)
        with self.assertRaisesRegex(ValueError,'training index'):self.tool.verify_model(model)
        # Refresh the complete model's outer receipt too: constructor still rederives.
        outer=self.write('forged-model.json',model)
        with self.assertRaisesRegex(ValueError,'training index'):
            DagHostReleaseProvider(self.tool.checked(outer),'natural',{},ALIASES,
                scenario='host_issue_depth_supply',allow_identification_hypothesis=True,
                allow_exact_input_conditioning_hypothesis=True,public_bindings={})

if __name__=='__main__':unittest.main()
