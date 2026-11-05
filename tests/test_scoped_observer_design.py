"""A new runtime must receive fresh matched observer captures before training."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.collect_activation_training import digest, save
from experiments.prepare_activation_scoped_observer import prepare
from experiments.continue_activation_scoped_candidate import main as continue_candidate


class ScopedObserverDesignTests(unittest.TestCase):
    def fixture(self, base):
        diagnostic=base/'diagnostic';source=diagnostic/'scoped-round1';source.mkdir(parents=True)
        code=base/'worker.py';code.write_text('# immutable fixture\n')
        snapshot=base/'worker-snapshot.py';snapshot.write_bytes(code.read_bytes())
        sha=hashlib.sha256(code.read_bytes()).hexdigest()
        aliases={f+'#0':f for f in ('generalist_2','text','image','ocr','asr')}
        context=dict(source_content=[dict(path=str(code),sha256=sha)],resident_aliases=aliases,
            runtime_policy=dict(scope='g2_strict'),observer_diagnostic='five_model_runtime_scope_diagnostic',
            worker_bindings={a:dict(command=['python',str(code)],env=dict(GS_RUNTIME_SCOPE='g2_strict',GS_SCOPED_OBSERVER='semantic')) for a in aliases})
        save(source/'context.json',context);save(source/'plan.json',{})
        save(source/'raw.json',dict(kind='cpu_fixture'))
        save(source/'sources.json',[dict(path=str(code),snapshot=str(snapshot),sha256=sha)])
        receipt=str(source/'raw.json')+':sha256:'+hashlib.sha256((source/'raw.json').read_bytes()).hexdigest()
        save(diagnostic/'description.json',dict(output_repeat_diagnostic_passed=True,issues=[],sources={'scoped-round1':receipt}))
        cohort=base/'cohort';(cohort/'semantic-round1').mkdir(parents=True)
        jobs=[dict(id=str(i),requests=[dict(alias=a,task=f if f!='generalist_2' else 'text',pool_id=str(i),offset_ms=0.)])
              for i,(a,f) in enumerate(aliases.items())]
        save(cohort/'semantic-round1/plan.json',dict(split='development_observer_diagnostic',aliases=aliases,
            jobs=jobs,expected_semantic_components={}))
        save(cohort/'observer-criteria.json',dict(independent_paired_rounds=2,
            maximum_mean_absolute_e2e_deviation=.1,maximum_p95_absolute_e2e_deviation=.25))
        return diagnostic,cohort,code

    def test_paired_jobs_and_contexts_differ_only_in_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);diagnostic,cohort,_=self.fixture(base);output=base/'new'
            prepare(output,diagnostic,cohort,'g2_strict',6111171)
            queue=json.loads((output/'queue.json').read_text())
            self.assertEqual([Path(p).name for p in queue['captures']],
                ['control-round1','semantic-round1','semantic-round2','control-round2'])
            normalized=[]
            for name in ('control-round1','semantic-round1','semantic-round2','control-round2'):
                ctx=json.loads((output/name/'context.json').read_text())
                plan=json.loads((output/name/'plan.json').read_text())
                self.assertEqual(plan['execution_context_sha256'],digest(ctx))
                self.assertEqual(plan['split'],'development_observer_diagnostic')
                self.assertTrue(plan['not_for_training'])
                self.assertFalse(plan['formal_utility_qualified'])
                mode='complete' if name.startswith('control') else 'semantic'
                self.assertEqual(ctx.pop('observer_diagnostic'),'five_model_scoped_'+mode)
                for binding in ctx['worker_bindings'].values():
                    self.assertEqual(binding['env'].pop('GS_SCOPED_OBSERVER'),mode)
                normalized.append(ctx)
            self.assertTrue(all(c==normalized[0] for c in normalized))
            for round_no in (1,2):
                plans=[json.loads((output/f'{arm}-round{round_no}'/'plan.json').read_text()) for arm in ('control','semantic')]
                self.assertEqual(plans[0]['jobs'],plans[1]['jobs'])

    def test_rejects_unfinished_diagnostic_reserved_seed_and_source_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);diagnostic,cohort,code=self.fixture(base)
            with self.assertRaisesRegex(ValueError,'nonconfirmation'):
                prepare(base/'reserved',diagnostic,cohort,'g2_strict',6110491)
            report=json.loads((diagnostic/'description.json').read_text())
            save(diagnostic/'description.json',dict(report,output_repeat_diagnostic_passed=False))
            with self.assertRaisesRegex(ValueError,'Complete clean'):
                prepare(base/'failed',diagnostic,cohort,'g2_strict',6111171)
            save(diagnostic/'description.json',report);code.write_text('# drift\n')
            with self.assertRaisesRegex(ValueError,'source drift'):
                prepare(base/'drift',diagnostic,cohort,'g2_strict',6111171)
            self.assertFalse((base/'drift').exists())

    def test_failed_observer_gate_cannot_start_full_pool_training(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);observer=base/'observer';observer.mkdir()
            reference=base/'reference';reference.mkdir();save(reference/'system-frozen.json',{})
            save(observer/'design.json',dict(source_receipts=[]))
            save(observer/'observer-criteria.json',{});save(observer/'queue.json',{})
            calls=[]
            def run(command, **kwargs):
                calls.append(command)
                if Path(command[1]).name=='analyze_activation_scoped_observer.py':
                    save(observer/'description.json',dict(observer_repeat_check_passed=False,issues=['OUTPUT_CHANGED']))
            args=['continue','--observer-base',str(observer),'--solo-base',str(base/'solo'),
                  '--reference',str(reference),'--solo-seed','6111181']
            with patch('sys.argv',args),patch('experiments.continue_activation_scoped_candidate.subprocess.run',side_effect=run):
                self.assertEqual(continue_candidate(),2)
            self.assertEqual(len(calls),2)
            self.assertFalse((base/'solo').exists())
            state=json.loads((observer/'state.json').read_text())
            self.assertEqual(state['status'],'observer_gate_rejected')
            self.assertFalse(state['physical_predictor_qualified'])


if __name__=='__main__':unittest.main()
