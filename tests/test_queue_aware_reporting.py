import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from tools.queue_aware_20261023.reporting import (
    candidate, digest, generate, native_observation, paired, rank_candidates,
)


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


def raw(contract, seed, cost, *, eligible=True):
    utility = {t + '_norm': cost for t in 'QTLPME'}
    utility.update(U=-cost if eligible else None, diagnostic_U=-.000001,
                   eligible=eligible, utility_complete=True)
    utility.update({key: True for key in ('quality_identity_verified', 'cohort_verified',
                  'constraints_verified', 'memory_verified', 'energy_verified')})
    return dict(backend='native', status='completed', policy_contract=contract,
        transport_config=dict(random_seed=seed), config=dict(weights={t: 1/6 for t in 'QTLPME'}),
        evaluation=dict(utility=utility), replica_to_family={'text#0': 'text', 'generalist_2#0': 'generalist_2'},
        request_records=[dict(task='text', model='text#0', worker_latency_ms=20, worker_queue_wait_ms=5),
                         dict(task='text', model='generalist_2#0', worker_latency_ms=40, worker_queue_wait_ms=0)],
        preflight=dict(gpu=dict(uuid='GPU-current')), errors=[])


class ReportingTests(unittest.TestCase):
    def fixture(self, root, *, complete=False):
        workflow, search, declaration = root/'workflow', root/'search', root/'protocol/protocol.json'
        workflow.mkdir(); search.mkdir()
        contracts = [dict(deployment={'text': 2, 'generalist_2': 1, 'ocr': 1}, strategy=s)
                     for s in ('no_split_swrr', 'fixed_1of32_swrr', 'queue_latency',
                               'queue_quality', 'queue_specialist_only')]
        contracts += [dict(deployment={'text': 2, 'image': 1, 'ocr': 1, 'asr': 1}, strategy='queue_specialist_only'),
                      dict(deployment={'text': 2, 'generalist_2': 1, 'image': 1, 'ocr': 1}, strategy='queue_latency')]
        packet = root/'packet/packet.json'
        write(packet, dict(jobs=[dict(policy_contract=c) for c in contracts], excluded_controls={'pureG': 'UNKNOWN'}))
        write(declaration, dict(contracts=dict(workload_rps=dict(text=1.85, image=.01, ocr=.01, asr=.01),
            horizon_seconds=600, warmup_seconds=30, rho_max=.8, gpu_budget_mib=49140,
            utility_weights={t: 1/6 for t in 'QTLPME'}),
            development=dict(packet=dict(path=str(packet))), execution=dict(expected_gpu_uuid='GPU-current')))
        stages = []
        progress = dict(status='COMPLETE' if complete else 'RUNNING', phase='COMPLETE' if complete else 'development', stages=stages)
        if complete:
            selected = [dict(candidate_id=digest(candidate(c)), candidate=candidate(c)) for c in contracts[:2]]
            write(workflow/'frozen-validation-packet.json', dict(selection=dict(candidates=selected)))
            write(workflow/'result.json', dict(status='COMPLETE'))
            for label, seeds in (('development', [113,114]), ('independent-validation', [81,82]), ('final-confirmation', [71,72,73])):
                stage = dict(label=label, status='COMPLETE_AUDITED', jobs=[]); stages.append(stage)
                for index, c in enumerate(contracts if label=='development' else contracts[:2]):
                    for seed in seeds:
                        # Selected winner reverses against control in seed72.
                        cost = (.4 if index == 0 and seed == 72 else .1 + index*.1)
                        path = workflow/label/f'candidate{index}-seed{seed}/run.json'
                        write(path, raw(c, seed, cost))
                        stage['jobs'].append(dict(raw=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
                write(workflow/label/'analysis/report.json', dict(seeds=seeds))
            write(workflow/'all-independent-pairs-analysis/report.json', {})
        write(workflow/'progress.json', progress)
        return workflow, declaration, search, contracts

    def test_in_progress_declares_all_seven_without_inventing_means(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); workflow, declaration, search, _ = self.fixture(root)
            before=(workflow/'progress.json').read_bytes()
            report=generate(workflow, declaration, search, root/'new-report')
            self.assertEqual(report['status'], 'IN_PROGRESS')
            ranking=report['stages']['development']['ranking']
            self.assertEqual(len(ranking), 7)
            self.assertTrue(all(r['mean_U'] is None for r in ranking))
            self.assertIsNone(report['combined_pairs'])
            self.assertEqual(before, (workflow/'progress.json').read_bytes())
            text=(root/'new-report/研究报告.md').read_text()
            self.assertIn('未完成/不可评价', text)
            self.assertIn('GPU-533a287b', text)
            self.assertIn('不是同一物理GPU', text)
            self.assertIn('G1/G3为UNKNOWN', text)

    def test_ineligible_diagnostic_is_never_native_utility(self):
        c=dict(deployment={'text': 2}, strategy='queue_quality')
        d=raw(c, 113, .1, eligible=False)
        observation=native_observation(d, '/unused')
        self.assertIsNone(observation['U'])
        ranking=rank_candidates({digest(candidate(c)): candidate(c)}, [observation], [113,114])
        self.assertIsNone(ranking[0]['mean_U'])
        self.assertFalse(ranking[0]['complete'])
        d['evaluation']['utility']['eligible']=True
        d['evaluation']['utility']['U']=-.1
        d['evaluation']['utility']['quality_identity_verified']=False
        self.assertIsNone(native_observation(d, '/unused')['U'])

    def test_final_report_separates_phases_and_preserves_reversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); workflow, declaration, search, _ = self.fixture(root, complete=True)
            report=generate(workflow, declaration, search, root/'final')
            self.assertEqual(report['status'], 'FINAL')
            self.assertEqual(report['stages']['independent-validation']['pair']['n_whole_pairs'], 2)
            confirmation=report['stages']['final-confirmation']['pair']
            self.assertEqual(confirmation['n_whole_pairs'], 3)
            self.assertEqual(confirmation['negative'], 1)
            self.assertIn('收益反转', confirmation['interpretation'])
            self.assertEqual(report['combined_pairs']['n_whole_pairs'], 5)
            self.assertFalse(report['physical_full_pool_optimality'])
            self.assertTrue((root/'final/manifest.json').exists())
            self.assertIn('不是第三次独立验证', (root/'final/研究报告.md').read_text())

    def test_registered_raw_mutation_and_existing_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); workflow, declaration, search, _ = self.fixture(root, complete=True)
            with self.assertRaisesRegex(ValueError, 'outside'):
                generate(workflow, declaration, search, workflow/'new-report')
            out=root/'taken'; out.mkdir()
            with self.assertRaises(FileExistsError):
                generate(workflow, declaration, search, out)
            path=next(workflow.glob('development/*/run.json'))
            path.write_text(path.read_text()+' ')
            with self.assertRaisesRegex(ValueError, 'changed'):
                generate(workflow, declaration, search, root/'bad')

    def test_partial_pairs_do_not_report_a_partial_mean(self):
        c1=dict(deployment={'text':2},strategy='a'); c2=dict(deployment={'text':2},strategy='b')
        rows=[native_observation(raw(c1,81,.1), '/a'), native_observation(raw(c2,81,.2), '/b')]
        result=paired(rows, digest(candidate(c1)), digest(candidate(c2)), [81,82])
        self.assertEqual(result['n_whole_pairs'],1)
        self.assertIsNone(result['mean'])
        self.assertEqual(result['per_seed'][0]['delta_selected_minus_control'], .1)
        self.assertIsNone(result['per_seed'][1]['delta_selected_minus_control'])

    def test_prelaunch_stop_is_not_native_experiment_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); workflow, declaration, search, _ = self.fixture(root)
            state=json.loads((workflow/'progress.json').read_text())
            state.update(status='STOPPED_FOR_DIAGNOSIS_NO_RETRY', error='Optional evidence serialization mismatch')
            write(workflow/'progress.json', state)
            result=generate(workflow, declaration, search, root/'prelaunch')
            self.assertEqual(result['status'], 'PRELAUNCH_STOPPED')
            self.assertEqual(result['registered_terminal_native_runs'], 0)
            self.assertIn('不是已运行GPU实验的失败', (root/'prelaunch/研究报告.md').read_text())

    def test_static_history_does_not_consume_half_run_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); workflow, declaration, search, _ = self.fixture(root)
            history=root/'old-execution/terminal-audit/report.json'
            history.parent.mkdir(parents=True)
            history.write_text('deliberately not a readable half-run ranking')
            write(history.parent.parent/'progress.json', dict(status='RUNNING'))
            partial=generate(workflow, declaration, search, root/'partial', static_audit=history)
            self.assertEqual(partial['static_history']['status'], 'WAITING_TERMINAL_LINK_ONLY')
            self.assertNotIn(str(history), [r['path'] for r in partial['inputs']])
            write(history, dict(observations=[dict(label='historical-fixed1of32', seed=109,
                audit_status='VERIFIED_TERMINAL_EVIDENCE', U=-.7, utility=dict(eligible=True))]))
            write(history.parent.parent/'progress.json', dict(status='COMPLETE_COLLECTION_AUDIT_AND_PAIR_SUMMARY'))
            finished=generate(workflow, declaration, search, root/'historical', static_audit=history)
            self.assertEqual(finished['static_history']['status'], 'TERMINAL_HISTORICAL_DEVELOPMENT')
            self.assertEqual(len(finished['stages']['development']['ranking']), 7)
            self.assertTrue(all(r['mean_U'] is None for r in finished['stages']['development']['ranking']))

    def test_posthoc_quality_audit_keeps_native_selection_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); workflow, declaration, search, _ = self.fixture(root)
            audit=root/'posthoc-quality/report.json'
            write(audit, dict(schema='gsdeploy.static_quality_equality_posthoc_audit.v1',
                historical_results_modified=False, GPU_used=False, per_request_scored_count=1111,
                per_seed=[dict(seed=109, observations=[dict(saved_Q_norm=.281),dict(saved_Q_norm=.281)],
                    spill_request_count=35, spill_unique_inputs=2, spill_score_changes=0)],
                historical_fullpool=dict(specialist_EM=.71875,generalist_EM=.65625,same_correct=20,
                    same_wrong=8,specialist_only_correct=3,generalist_only_correct=1)))
            result=generate(workflow, declaration, search, root/'quality-report', quality_audit=audit)
            self.assertEqual(result['status'],'IN_PROGRESS')
            self.assertTrue(all(r['mean_U'] is None for r in result['stages']['development']['ranking']))
            text=(root/'quality-report/研究报告.md').read_text()
            self.assertIn('相锁',text)
            self.assertIn('未使用aggregate Q代替实测',text)
            self.assertIn('Q贡献差为0仍原样保留',text)
            self.assertIn(str(audit), [r['path'] for r in result['inputs']])

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Optional matplotlib unavailable')
    def test_optional_png_pdf_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); workflow, declaration, search, _ = self.fixture(root, complete=True)
            generate(workflow, declaration, search, root/'figures', plots=True)
            self.assertEqual(len(list((root/'figures').glob('*.png'))), 2)
            self.assertEqual(len(list((root/'figures').glob('*.pdf'))), 2)
            self.assertTrue((root/'figures/native-development-utility.pdf').read_bytes().startswith(b'%PDF'))


if __name__=='__main__':
    unittest.main()
