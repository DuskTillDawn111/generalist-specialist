"""Audited-JSON-only fixtures; failure and pending rows cannot become evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'tools/spillover_discovery_20261022/build_development_data_report.py'
spec=importlib.util.spec_from_file_location('development_data_report',PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ReportTests(unittest.TestCase):
    def fixture(self,root):
        paths={k:root/k/'audit.json' for k in ('five','asr','text')}
        five=json.loads((module.STUDY/'five-development-point-audit-v1/audit.json').read_text())
        paths['five'].parent.mkdir();paths['five'].write_text(json.dumps(five))
        return paths,five

    def test_missing_stays_pending_only_audited_json_is_read_and_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);paths,five=self.fixture(root);read_text=Path.read_text;opened=[]
            def tracked(path,*a,**kw):
                self.assertEqual(path.name,'audit.json');opened.append(path)
                return read_text(path,*a,**kw)
            with patch.object(Path,'read_text',tracked):report=module.build(paths,root/'out')
            self.assertEqual(report['status'],'PENDING');self.assertEqual(len(opened),1)
            self.assertEqual(len(report['conditions']),5);self.assertEqual(sum(p['comparable'] for p in report['paired_comparisons']),4)
            self.assertTrue((root/'out/sources.csv').is_file())
            self.assertEqual([s['status'] for s in report['sources']],['AUDITED','PENDING','PENDING'])
            with self.assertRaises(FileExistsError):module.build(paths,root/'out')

    def test_invalid_or_unstable_and_paused_records_keep_U_null_no_diag_substitution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);paths,five=self.fixture(root)
            original=five['conditions'][0]
            c={'id':'failed','utility':copy.deepcopy(original['measured_six_term_utility']),
                'plan':original['plan'],'validity':False,'strict_import_passed':False,'seed':52,
                'stability':{'stability_state_v2':'unstable'},'source':{'sha256':'fixture'},'diagnostic_partial_U':123.}
            row=module.normalize('asr',c)
            self.assertIsNone(row['U']);self.assertFalse(row['valid_utility'])
            c.update(validity=True,strict_import_passed=True);c['stability']['stability_state_v2']='steady'
            self.assertIsNone(module.normalize('asr',c,selected=False)['U'])
            pair=module.pair('asr',{'a':row,'b':row},'a','b',123.,True,'fixture')
            self.assertIsNone(pair['delta_U']);self.assertFalse(pair['comparable'])

    def test_schema_and_heldout_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);paths,five=self.fixture(root)
            paths['asr'].parent.mkdir();paths['asr'].write_text(json.dumps(five))
            with self.assertRaisesRegex(ValueError,'schema'):module.load_audit('asr',paths['asr'])
            with self.assertRaisesRegex(ValueError,'Heldout'):module.load_audit('text',root/'transfer-validation-any/audit.json')

    def test_request_completion_is_independent_of_eligibility_and_retains_abort_reason(self):
        audit=json.loads((module.STUDY/'asr19-replication-development-audit-v3/audit.json').read_text())
        rows={c['id']:module.normalize('asr',c) for c in audit['conditions']}
        split=rows['seed52_n2_split'];nosplit=rows['seed52_n2_nosplit']
        self.assertEqual((split['offered_requests'],split['completed_requests'],split['not_completed_requests']),(1227,1227,0))
        self.assertEqual((split['completeness'],split['run_status'],split['utility_eligible'],split['stability']),('完整','completed',False,'unstable'))
        self.assertFalse(split['valid_utility']);self.assertIsNone(split['U'])
        self.assertEqual((nosplit['completeness'],nosplit['run_status'],nosplit['completed_requests'],nosplit['not_completed_requests']),('未完成229','failed',998,229))
        paused=module.normalize('asr',audit['excluded_user_paused_attempts'][0],False)
        self.assertEqual(paused['utility_reason'],'WINDOW_ABORTED')
        self.assertEqual(paused['request_status_counts'],{'completed':297,'error':42,'incomplete':821})
        self.assertEqual(paused['not_completed_requests'],863);self.assertIsNone(paused['U'])

    def test_unknown_completion_and_explicit_task_summaries_never_use_eligible_as_counts(self):
        u=dict(eligible=True,utility_complete=True)
        unknown=module.request_completion({'cohort':{'offered':9}},u)
        self.assertEqual(unknown['completeness'],'UNKNOWN');self.assertIsNone(unknown['completed_requests'])
        self.assertEqual(unknown['run_status'],'UNKNOWN')
        text=module.request_completion({'status':'completed','request_summaries':{'task':{
            'text':{'offered':8,'completed':8},'ocr':{'offered':1,'completed':1}}}},dict(eligible=False))
        self.assertEqual((text['offered_requests'],text['completed_requests'],text['completeness']),(9,9,'完整'))
        self.assertFalse(text['utility_eligible'])
        with self.assertRaisesRegex(ValueError,'differ'):
            module.request_completion({'cohort':{'offered':9},'status_counts':{'completed':8}},u)

    def test_complete_recoverable_memory_invalid_remains_distinct_and_no_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            paths,five=self.fixture(Path(temp));original=five['conditions'][0]
            c=dict(id='memory_invalid',seed=51,plan=original['plan'],workload=original['workload'],
                utility=copy.deepcopy(original['measured_six_term_utility']),strict_import_passed=False,
                utility_comparison_allowed=False,status='completed',
                stability={'stability_state_v2':'recoverable_congested'},
                request_summaries={'task':{'text':{'offered':9,'completed':9}}},
                memory_evidence={'resource_valid':False,'safe_memory_mib':50219.487632})
            c['utility'].update(eligible=False,U=None,M_norm=1.0219675952787954)
            r=module.normalize('text',c)
            self.assertEqual(r['completeness'],'完整');self.assertEqual(r['stability'],'recoverable_congested')
            self.assertEqual(r['utility_ineligibility_evidence'],'MEMORY_RESOURCE_INVALID')
            self.assertFalse(r['memory_resource_valid']);self.assertIsNone(r['U'])
            pair=module.pair('text',{'a':r,'b':r},'a','b',None,True,'same trace')
            self.assertTrue(pair['same_trace_verified_by_source']);self.assertFalse(pair['comparable'])

    def test_partial_source_rows_are_visible_but_never_compared(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);paths,five=self.fixture(root)
            five['status']='INCOMPLETE';paths['five'].write_text(json.dumps(five))
            report=module.build(paths,root/'out')
            self.assertTrue(report['conditions'])
            self.assertTrue(all(r['U'] is None for r in report['conditions']))
            self.assertTrue(all(not p['comparable'] and p['delta_U'] is None for p in report['paired_comparisons']))

    def test_asr_v3_and_text_schema_adapters_preserve_failed_pair_and_duplicate_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);paths,five=self.fixture(root);original=five['conditions'][3]
            conditions=[]
            for seed in (51,52,53):
                for split in (False,True):
                    plan=copy.deepcopy(original['plan'])
                    if split:plan['routing']['asr']={'asr':1.6625,'generalist_2':.2375}
                    utility=copy.deepcopy(original['measured_six_term_utility'])
                    valid=not(seed==52 and not split)
                    if not valid:utility.update(U=None,eligible=False)
                    conditions.append({'id':f'seed{seed}_n2_'+('split' if split else 'nosplit'),
                        'seed':seed,'plan':plan,'utility':utility,'validity':valid,'strict_import_passed':valid,
                        'stability':{'stability_state_v2':'steady' if valid else 'unstable'},
                        'source':original['source'] if seed==51 and not split else {'sha256':str(seed)+str(split)}})
            n3=copy.deepcopy(conditions[0]);n3['id']='seed51_n3_nosplit';n3['plan']['deployment']['asr']=3;conditions.append(n3)
            paused=copy.deepcopy(conditions[4]);paused.update(validity=False,strict_import_passed=False)
            paused['utility']=dict(U=None,eligible=False)
            audit={'schema':module.SCHEMAS['asr'],'status':module.COMPLETE['asr'],
                'conditions':conditions,'excluded_user_paused_attempts':[paused],'all_attempt_count':8,
                'paired':{'pairs':[dict(seed=s,status='INVALID_PAIRED_BLOCK' if s==52 else 'VALID_PAIRED_BLOCK',
                    delta_U_split_minus_nosplit=None if s==52 else 0.) for s in (51,52,53)]},'n3_structural_controls':[]}
            paths['asr'].parent.mkdir();paths['asr'].write_text(json.dumps(audit))
            text=[]
            labels=['text26_n2_specialist_share_4of7','text26_n2_specialist_share_1of2','text26_n1_specialist_share_1of2','text26_n3_specialist_share_4of7']
            for label in labels:
                text.append(dict(id=label,label=label,utility=original['measured_six_term_utility'],
                    plan=original['plan'],workload=original['workload'],seed=51,strict_import_passed=True,
                    utility_comparison_allowed=True,stability={'stability_state_v2':'steady'},status='completed',source={}))
            textaudit={'schema':module.SCHEMAS['text'],'status':module.COMPLETE['text'],'conditions':text,
                'same_trace_comparisons':[dict(a=labels[0],b=labels[1],delta_U=0.,utility_comparison_allowed=True,same_ordered_task_sample_schedule=True)]}
            paths['text'].parent.mkdir();paths['text'].write_text(json.dumps(textaudit))
            report=module.build(paths,root/'out')
            self.assertEqual(report['status'],'COMPLETE_AUDITED_SOURCES')
            invalid=next(p for p in report['paired_comparisons'] if p['dataset']=='asr' and p['a']=='seed52_n2_nosplit')
            self.assertFalse(invalid['comparable']);self.assertIsNone(invalid['delta_U'])
            preserved=next(r for r in report['conditions'] if not r['selected_for_condition'])
            self.assertIsNone(preserved['U'])
            duplicate=next(r for r in report['conditions'] if r['dataset']=='asr' and r['id']=='seed51_n2_nosplit')
            self.assertIsNotNone(duplicate['same_raw_already_shown_as'])


if __name__=='__main__':unittest.main()
