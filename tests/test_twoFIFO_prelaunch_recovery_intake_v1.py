import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.predictor_validation_20261027 import twoFIFO_prelaunch_recovery_intake_v1 as m
from tests.test_twoFIFO_source_rebound_intake_v1 import fixture, write


def recovery_fixture():
    root,man,fin,plans,reviews,mrs,rds,mds=fixture()
    for i,(pr,mr) in enumerate(zip(plans,mrs)):
        plan=json.loads(Path(pr['path']).read_text());doc=mds[mr['path']];n=doc['measured_requests']
        arrivals=[dict(request_id='request-'+str(i)+'-'+str(j)) for j in range(n)]
        plan['cases']=[dict(arrivals=arrivals)]
        folder=root/('canonical'+str(i));folder.mkdir();pr.update(write(folder,'plan.json',plan))
        rr=rds[reviews[i]['path']];raw=json.loads(Path(rr['raw']['path']).read_text());raw['plan']=pr
        rr['raw']=write(root,Path(rr['raw']['path']).name,raw)
        doc['requests']=arrivals;mr.update(write(root,Path(mr['path']).name,doc))
    manifest=json.loads(Path(man['path']).read_text())
    for i,item in enumerate(manifest['attempts']):
        item.update(plan=plans[i]);item['execution_id']=json.loads(Path(plans[i]['path']).read_text())['execution_id']
    man.update(write(root,'manifest.json',manifest))
    final=json.loads(Path(fin['path']).read_text())
    for i,row in enumerate(final['completed_attempts']):
        row.update(training_review=reviews[i],macros=mrs[i]); rr=rds[reviews[i]['path']]
        rr.update(source_rebound_manifest=man,execution=row['execution']);reviews[i].update(write(root,Path(reviews[i]['path']).name,rr))
        doc=mds[mrs[i]['path']];doc['source_rebound_manifest']=man;mrs[i].update(write(root,Path(mrs[i]['path']).name,doc))
    prefix=copy.deepcopy(final['completed_attempts'][:4]);tail=[]
    rec=dict(proof=dict(prefix=prefix,stopped_report=write(root,'old-stopped.json',{}),tail=[],prelaunch_claim=write(root,'oldclaim.json',{})),run_output=str(root))
    rref=write(root,'recovery.json',{})
    for i,row in enumerate(final['completed_attempts'][4:]):
        plan=plans[4+i]
        item=dict(manifest['attempts'][4+i])
        # Canonical envelope checks depend only on item path; authenticated
        # original plan refs remain the manifest's actual fixture plan paths.
        rec['proof']['tail'].append(item);tail.append(row)
    rec['proof']['original_manifest']=man
    for i,(item,row) in enumerate(zip(rec['proof']['tail'],tail)):
        auth=dict(schema=m.recovery.SCHEMA+'.one_shot_authorization',recovery_manifest=rref,policy=m.recovery.POLICY,
            original_manifest=man,item=item,preserved_claim=rec['proof']['prelaunch_claim'] if i==0 else None,already_started_capture_retry=False)
        folder=Path(item['plan']['path']).parent;ar=write(folder,m.recovery.RECOVERY_CLAIM,auth)
        env=dict(schema=m.recovery.SCHEMA+'.execution_envelope',policy=m.recovery.POLICY,recovery_manifest=rref,
            authorization=ar,execution=row['execution'],item=item,original_manifest=man,
            source_guard='KERNEL_INOTIFY_AND_FULL_CLOSURE_BEFORE_AFTER',no_started_capture_retry=True)
        row['recovery_envelope']=write(folder,m.recovery.RECOVERY_ENVELOPE,env)
    final.update(schema=m.recovery.SCHEMA+'.run',status=m.FINAL_STATUS,no_started_capture_retries=True,
        recovery_manifest=rref,original_manifest=man,policy=m.recovery.POLICY,stopped_report=rec['proof']['stopped_report'],
        prefix_attempts=prefix,completed_tail=tail,automatic_fit=False,execute_invocations=8)
    fin.update(write(root,'report.json',final))
    return root,man,fin,rref,manifest,rec,rds,mds


class RecoveryIntakeTests(unittest.TestCase):
    def test_exact12_positive_new_producer_and_detached_schema(self):
        root,man,fin,rref,manifest,rec,rds,mds=recovery_fixture()
        def rebuild(raw,out,execution,**kwargs):
            return next(row['training_review'] for row in json.loads(Path(fin['path']).read_text())['completed_attempts'] if row['execution']['path']==execution)
        with patch.object(m.executor,'audit',return_value=manifest),patch.object(m.recovery,'audit',return_value=rec),patch.object(m.executor,'completed_execution'),patch.object(m.native_review,'review',side_effect=rebuild),patch.object(m.macros,'extract',side_effect=lambda path: next(mds[row['macros']['path']] for row in json.loads(Path(fin['path']).read_text())['completed_attempts'] if row['training_review']['path']==path)):
            doc=m.intake(man['path'],fin['path'],recovery_manifest_path=rref['path'])
        self.assertEqual(doc['schema'],m.SCHEMA);self.assertEqual(doc['measured_requests'],546);self.assertEqual(doc['priming_requests'],216)
        self.assertEqual(doc['preserved_prefix_attempts'],4);self.assertEqual(doc['new_policy_tail_attempts'],8)
        self.assertEqual(doc['manifest'],man);self.assertEqual(doc['recovery_manifest'],rref);self.assertFalse(doc['numerical_fit_run'])
    def test_old_run_and_bad_recovery_envelope_rejected_before_rederive(self):
        for mutation in ('old-schema','prefix','fit','wrong-envelope','wrong-authorization'):
            with self.subTest(mutation=mutation):
                root,man,fin,rref,manifest,rec,rds,mds=recovery_fixture();final=json.loads(Path(fin['path']).read_text())
                if mutation=='old-schema':final['schema']=m.executor.CONT_SCHEMA+'.run'
                elif mutation=='prefix':final['prefix_attempts']=[]
                elif mutation=='fit':final['automatic_fit']=True
                else:
                    er=final['completed_tail'][0]['recovery_envelope'];env=json.loads(Path(er['path']).read_text())
                    if mutation=='wrong-envelope':env['item']['execution_id']='other'
                    else:
                        auth=json.loads(Path(env['authorization']['path']).read_text());auth['policy']='old claimed-no-retry';env['authorization']=write(Path(env['authorization']['path']).parent,Path(env['authorization']['path']).name,auth)
                    er.update(write(Path(er['path']).parent,Path(er['path']).name,env))
                write(root,'report.json',final)
                with patch.object(m.executor,'audit',return_value=manifest),patch.object(m.recovery,'audit',return_value=rec),patch.object(m.native_review,'review') as rebuild:
                    with self.assertRaises(ValueError):m.intake(man['path'],fin['path'],recovery_manifest_path=rref['path'])
                    rebuild.assert_not_called()


if __name__=='__main__':unittest.main()
