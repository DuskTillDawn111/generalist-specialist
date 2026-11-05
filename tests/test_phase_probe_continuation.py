"""Rejected upstream evidence or fitting must stop dependent GPU/validation work."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.collect_activation_training import save
from experiments.continue_activation_phase_pair_probe import main


class ProbeContinuationTests(unittest.TestCase):
    def fixture(self, directory, status):
        base=Path(directory);observer=base/'observer';observer.mkdir()
        save(observer/'pipeline-plan.json',dict(source_receipts=[]))
        save(observer/'state.json',dict(status=status))
        args=['probe','--observer-base',str(observer),'--solo-base',str(base/'solo'),
              '--output',str(base/'probe'),'--seed','6111191','--upstream-pid','99999999']
        return base,args

    def test_failed_observer_prevents_every_dependent_command(self):
        with tempfile.TemporaryDirectory() as directory:
            base,args=self.fixture(directory,'observer_gate_rejected')
            with patch('sys.argv',args),patch('experiments.continue_activation_phase_pair_probe.subprocess.run') as run:
                self.assertEqual(main(),2);run.assert_not_called()
            state=json.loads((base/'probe/state.json').read_text())
            self.assertEqual(state['status'],'upstream_rejected')
            self.assertFalse(state['physical_predictor_qualified'])

    def test_rejected_fit_retains_support_audit_and_skips_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            base,args=self.fixture(directory,'full_pool_baseline_diagnostic_completed');calls=[]
            def run(command, **kwargs):
                name=Path(command[1]).name;calls.append(name)
                if name=='fit_activation_completion_bands.py':
                    save(base/'probe/fit.json',dict(training_fit_gates_passed=False,
                        profiles=dict(derived_host_call_band_model=dict(progress_law=dict(fit_diagnostics=[dict(accepted=False,reason='MISSING_REPEAT')]))),
                        energy_diagnostic=dict(accepted=False)))
                elif name=='audit_activation_band_support.py':save(base/'probe/support-audit.json',dict(fixture=True))
            with patch('sys.argv',args),patch('experiments.continue_activation_phase_pair_probe.subprocess.run',side_effect=run):
                self.assertEqual(main(),2)
            self.assertEqual(calls,['prepare_activation_phase_pair_probe.py','run_activation_queue.py',
                                   'fit_activation_completion_bands.py','audit_activation_band_support.py'])
            self.assertTrue((base/'probe/support-audit.json').exists())
            state=json.loads((base/'probe/state.json').read_text())
            self.assertEqual(state['status'],'phase_pair_training_fit_rejected')
            self.assertFalse(state['formal_utility_qualified'])


if __name__=='__main__':unittest.main()
