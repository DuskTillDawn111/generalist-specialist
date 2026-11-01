import copy
import json
import tempfile
from pathlib import Path
import unittest
from tools.queue_refined_20261023.calibration import (ServiceModel,fit,visible_context,cross_seed,
    extract_completed_calibration,extract_development,digest,sha,CALIBRATION_CLAIM)

def records(values=(10,20,100),seed=113,family='generalist_2',task='text',pool='inputA',context=None):
    return [dict(family=family,task=task,pool_id=pool,service_ms=v,seed=seed,
        context=context or dict(other_active_bin='1',g2_visible_outstanding=False),
        deployment={family:1},raw_sha256=f'run{seed}',request_id=str(i)) for i,v in enumerate(values)]

class CalibrationTests(unittest.TestCase):
    def test_residual_conditions_on_survival_not_subtracted_mean(self):
        model=ServiceModel(fit(records()))
        pred=model.predict('generalist_2','text','inputA')
        self.assertAlmostEqual(pred['mean_ms'],130/3)
        tail=model.residual('generalist_2','text','inputA',25)
        self.assertAlmostEqual(tail['conditional_mean_ms'],75)
        self.assertAlmostEqual(tail['quantile_ms'],75)
        self.assertAlmostEqual(tail['survival_probability'],1/3)
        self.assertGreater(tail['conservative_ms'],5)
        self.assertNotAlmostEqual(tail['conditional_mean_ms'],pred['mean_ms']-25)

    def test_expired_head_retains_tail_scale_with_unverified_status(self):
        model=ServiceModel(fit(records()))
        tail=model.residual('generalist_2','text','inputA',200)
        self.assertEqual(tail['evidence_status'],'TAIL_EXTRAPOLATION_UNVERIFIED')
        self.assertIsNone(tail['conditional_mean_ms'])
        self.assertIsNone(tail['quantile_ms'])
        self.assertGreaterEqual(tail['conservative_ms'],100)
        self.assertEqual(tail['survival_probability'],0)

    def test_missing_input_shrinks_to_task_but_missing_task_unknown(self):
        model=ServiceModel(fit(records()))
        p=model.predict('generalist_2','text','unseen')
        self.assertAlmostEqual(p['mean_ms'],130/3)
        self.assertEqual(p['support']['input'],0)
        missing=model.predict('generalist_2','asr','unseen')
        self.assertIsNone(missing['conservative_ms'])
        self.assertEqual(missing['evidence_status'],'UNKNOWN_FAMILY_TASK')

    def test_context_changes_distribution_without_future_fields(self):
        aliases={'g':dict(family='generalist_2'),'s':dict(family='text'),'o':dict(family='ocr')}
        snapshot={'g':dict(outstanding=[]),'s':dict(outstanding=[dict(request_id='x')]),'o':dict(outstanding=[])}
        c=visible_context(snapshot,'g',aliases)
        altered=copy.deepcopy(snapshot);altered['s']['outstanding'][0].update(actual_future_service_ms=999,reference='answer')
        self.assertEqual(c,visible_context(altered,'g',aliases))
        data=records((10,)*40,context=c)+records((100,)*40,context=dict(other_active_bin='2plus',g2_visible_outstanding=True))
        model=ServiceModel(fit(data))
        self.assertLess(model.predict('generalist_2','text','inputA',c)['mean_ms'],
                        model.predict('generalist_2','text','inputA',dict(other_active_bin='2plus',g2_visible_outstanding=True))['mean_ms'])

    def test_reserved_seeds_rejected_and_no_quality_retained(self):
        with self.assertRaisesRegex(ValueError,'Reserved'):
            fit(records(seed=81))
        data=records();data[0].update(quality_truth=1,reference='secret',prediction='answer')
        doc=fit(data)
        self.assertFalse(doc['quality_used'])
        self.assertNotIn('secret',str(doc))
        self.assertNotIn('quality_truth',str(doc))

    def test_cross_seed_fit_does_not_consume_target_values(self):
        folds=cross_seed(records((10,10),113)+records((100,100),114))
        a=folds[0]['groups'][0];b=folds[1]['groups'][0]
        self.assertAlmostEqual(a['MAE_mean_ms'],90)
        self.assertAlmostEqual(b['MAE_mean_ms'],90)
        self.assertEqual(a['q90_empirical_coverage'],0)
        self.assertEqual(b['q90_empirical_coverage'],1)

    def test_deployment_isolation_and_alias_validation(self):
        a=records();b=records();b[0]['deployment']={'generalist_2':2}
        with self.assertRaisesRegex(ValueError,'Mixed deployments'):fit(a+b)
        model=ServiceModel(fit(a))
        self.assertEqual(model.document['deployment'],{'generalist_2':1})
        self.assertTrue(model.validate_aliases({'g#0':{'family':'generalist_2'}}))
        with self.assertRaisesRegex(ValueError,'deployment'):
            model.validate_aliases({'g#0':{'family':'generalist_2'},'g#1':{'family':'generalist_2'}})
        with self.assertRaisesRegex(ValueError,'deployment'):
            model.validate_aliases({'s#0':{'family':'text'}})
        with self.assertRaisesRegex(ValueError,'deployment'):
            fit([dict(a[0],deployment={'generalist_2':True})])

    def test_new_calibration_seed_needs_training_phase(self):
        with self.assertRaisesRegex(ValueError,'provenance'):fit(records(seed=201))
        doc=fit([dict(r,training_phase='calibration') for r in records(seed=201)])
        self.assertEqual(doc['training_seeds'],[201])

    def test_reserved_calibration_packet_fails_before_raw_read(self):
        with tempfile.TemporaryDirectory() as d:
            packet=Path(d)/'packet.json';progress=Path(d)/'progress.json'
            packet.write_text(json.dumps(dict(schema='gsdeploy.refined_calibration_packet.v1',phase='confirmation',seeds=[71,72])))
            progress.write_text(json.dumps(dict(status='COMPLETE',jobs=[dict(raw={'path':'/does/not/exist'})])))
            with self.assertRaisesRegex(ValueError,'calibration201/202'):
                extract_completed_calibration(packet,progress)

    def test_completed_probe_packet_hash_stream_and_raw_checks(self):
        import hashlib,random
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);prereg=root/'prereg.json';prereg.write_text('{}')
            jobs=[];terminals=[]
            for seed in (201,202):
                for prob in (0.,1/16):
                    probe=dict(schema='randomized_text_probe_v1',probability=prob,namespace='queue-refined-calibration-20261023')
                    contract=dict(deployment={'generalist_2':1},aliases={'g':dict(family='generalist_2')},strategy='queue_latency',probe=probe,
                        service_ms={'generalist_2':{'text':{'inputA':20}}})
                    identity=dict(schema='gsdeploy.refined_calibration_job.v1',seed=seed,phase='calibration',policy_contract_sha256=digest(contract),
                        deployment=contract['deployment'],measurement_claim=CALIBRATION_CLAIM,predeclaration=sha(prereg))
                    job=dict(id=digest(identity),phase='calibration',identity=identity,seed=seed,policy_contract=contract);jobs.append(job)
                    stream=hashlib.sha256(f"{probe['namespace']}:{seed}:measurement".encode()).hexdigest();coin=random.Random(int(stream,16)).random()
                    row=dict(status='completed',worker_response=dict(service_ms=30),model='g',task='text',pool_id='inputA',request_id='r',
                        routing_decision=dict(snapshot={'g':dict(outstanding=[])},decision=dict(probe=dict(spec=probe,stream_sha256=stream,draw_index=1,coin=coin,requested_probe=coin<prob))))
                    raw=dict(status='completed',backend='native',transport_config=dict(random_seed=seed),policy_contract=contract,replica_to_family={'g':'generalist_2'},
                        request_records=[row],experiment_job_id=job['id'],executed_measurement_identity=identity,executed_measurement_id=job['id'],experiment_phase='calibration',measurement_claim=CALIBRATION_CLAIM)
                    path=root/(job['id']+'.json');path.write_text(json.dumps(raw))
                    terminals.append(dict(job_id=job['id'],raw=sha(path),status='completed',exit_code=0))
            packet=root/'packet.json';packet.write_text(json.dumps(dict(schema='gsdeploy.refined_calibration_packet.v1',phase='calibration',seeds=[201,202],jobs=jobs,no_retries=True)))
            progress=root/'progress.json';state=dict(status='COMPLETE',packet=sha(packet),no_retries=True,jobs=terminals);progress.write_text(json.dumps(state))
            rows,sources=extract_completed_calibration(packet,progress,{'generalist_2':1})
            self.assertEqual(len(rows),4);self.assertEqual(fit(rows)['training_seeds'],[201,202]);self.assertEqual(len(sources),6)
            self.assertNotIn('probe',rows[0]);self.assertNotIn('coin',rows[0])
            state['status']='RUNNING';progress.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError,'COMPLETE'):extract_completed_calibration(packet,progress)
            state['status']='COMPLETE';progress.write_text(json.dumps(state))
            Path(terminals[0]['raw']['path']).write_text('{}')
            with self.assertRaisesRegex(ValueError,'raw SHA'):extract_completed_calibration(packet,progress)

    def test_development_filter_preserves_complete_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);analysis=root/'analysis';analysis.mkdir();observations=[]
            for i in range(14):
                seed=113+i%2;deployment={'generalist_2':1 if i<10 else 2};aliases={f'g{j}':dict(family='generalist_2') for j in range(deployment['generalist_2'])}
                raw=dict(status='completed',backend='native',transport_config=dict(random_seed=seed),
                    policy_contract=dict(deployment=deployment,aliases=aliases,strategy='queue_latency',service_ms={'generalist_2':{'text':{'inputA':20}}}),
                    replica_to_family={a:'generalist_2' for a in aliases},request_records=[dict(status='completed',worker_response=dict(service_ms=30),
                        model='g0',task='text',pool_id='inputA',request_id='r',routing_decision=dict(snapshot={a:dict(outstanding=[]) for a in aliases}))])
                path=root/(str(i)+'.json');path.write_text(json.dumps(raw));observations.append(dict(seed=seed,raw=sha(path)))
            report=analysis/'report.json';report.write_text(json.dumps(dict(seeds=[113,114],observations=observations)))
            rows,sources=extract_development(report,{'generalist_2':1})
            self.assertEqual(len(rows),10);self.assertEqual(len(sources),15);self.assertEqual(fit(rows)['deployment'],{'generalist_2':1})
            observations[-1]['raw']['sha256']='bad';report.write_text(json.dumps(dict(seeds=[113,114],observations=observations)))
            with self.assertRaisesRegex(ValueError,'Raw SHA'):extract_development(report,{'generalist_2':1})

if __name__=='__main__':unittest.main()
