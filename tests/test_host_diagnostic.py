"""Host brackets must contain events and distinguish CPU from elapsed time."""
import copy
import unittest
from experiments.analyze_activation_host_diagnostic import validate_host_bracket
from experiments.prepare_activation_host_diagnostic import normalized_context


class HostDiagnosticTests(unittest.TestCase):
    def test_bracket_rejects_incomplete_or_wrong_thread_receipt(self):
        event=dict(thread_id=1,host_start_ns=10,host_end_ns=100,
            passive_host_bracket=dict(schema='gsdeploy.passive_host_call_bracket.v1',thread_id=1,
                wall_start_ns=0,wall_end_ns=110,cpu_start_ns=1000,cpu_end_ns=1020,
                usage_before=dict(involuntary_context_switches=2),usage_after=dict(involuntary_context_switches=3)))
        validate_host_bracket(event)
        for replacement in [dict(thread_id=2),dict(wall_start_ns=11),dict(wall_end_ns=99),
                            dict(cpu_end_ns=20000),dict(usage_after=dict(involuntary_context_switches=1))]:
            row=copy.deepcopy(event);row['passive_host_bracket'].update(replacement)
            with self.subTest(replacement=replacement),self.assertRaises(ValueError):validate_host_bracket(row)

    def test_context_match_retains_runtime_environment_and_model_binding(self):
        ctx=dict(source_content=[],observer_diagnostic='old',runtime_policy=dict(scope='g2_strict'),
            worker_bindings=dict(a=dict(command=['python','old','--spec','frozen'],env=dict(OMP_NUM_THREADS='1'))))
        new=copy.deepcopy(ctx);new['observer_diagnostic']='new';new['worker_bindings']['a']['command'][1]='new'
        new['worker_bindings']['a']['env']['GS_PASSIVE_GC_TRACE']='on'
        self.assertEqual(normalized_context(ctx),normalized_context(new))
        new['worker_bindings']['a']['env']['OMP_NUM_THREADS']='2'
        self.assertNotEqual(normalized_context(ctx),normalized_context(new))


if __name__=='__main__':unittest.main()
