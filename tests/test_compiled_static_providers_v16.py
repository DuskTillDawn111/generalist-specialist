"""Compare entire original/compiled trajectories, not just cached lookups."""
import copy
import tempfile
import unittest
from pathlib import Path

from tools.predictor_validation_20261027.compiled_static_providers_v16 import CompiledStaticProviders
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import (
    DagPhaseFullHorizonModel, SchedulingDecision)
from tools.predictor_validation_20261027.native_collect import receipt


class Scheduler:
    def decide(self, view):
        return SchedulingDecision({n.key: (.5 if n.key[0] == 'a' else 1.)
                                   for n in view.active_gpu_nodes}, view.clock_ns + 7)

    def provenance(self):
        return dict(kind='deterministic_test_reference_progress')


class Lifecycle:
    def delays(self, *args):
        return dict(known=True, parent_receipt_ms=.000003, worker_next_ready_ms=.000005)


class Resources:
    def power(self, handlers, *args):
        return dict(known=True, watts=10 + len(handlers))

    def memory(self, handlers, *args):
        return dict(known=True, peak_memory_mib=20 + len(handlers))


class CompilationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'source.json'
        self.path.write_text('{"training":true}')
        self.aliases = {a:dict(supported_tasks=['text']) for a in ('a', 'b')}
        self.plans = {}
        self.work = {}; self.releases = {}; self.phases = {}
        for alias in self.aliases:
            nodes = [dict(id=i, kind=k, retained_source_metadata=['source', i])
                     for i,k in [('c','copy'),('e','event_record'),('g','gpu'),
                                 ('m','memset'),('w','event_wait'),('h','handler_end')]]
            coordinate = alias, 'text', 'short'
            self.plans[coordinate] = dict(nodes=nodes,
                edges=[dict(before=x['id'],after=y['id']) for x,y in zip(nodes,nodes[1:])],
                default_stream_semantics='UNMEASURED_EXPLICIT_HYPOTHESIS')
            for n in nodes:
                key = (*coordinate,n['id'])
                self.releases[key] = dict(not_before_offset_ns=2,
                    after_predecessors_ns=3 if n['kind']=='handler_end' else 0)
                if n['kind'] in ('copy','gpu','memset'): self.work[key]=11 if alias=='a' else 5
                if n['kind']=='gpu': self.phases[key]='prefill'

    def provider(self, values):
        def call(node,row):
            original = next(n for n in self.plans[row['alias'],row['task'],row['pool_id']]['nodes']
                            if n['id']==node['id'])
            self.assertEqual(node, original)
            return copy.deepcopy(values[row['alias'],row['task'],row['pool_id'],node['id']])
        return call

    def compiled(self):
        return CompiledStaticProviders(self.aliases, self.plans,
            duration=self.provider(self.work),release=self.provider(self.releases),
            phase=self.provider(self.phases),sources=[receipt(self.path)])

    def engine(self, compiled=None):
        return DagPhaseFullHorizonModel(
            self.aliases if compiled is None else compiled.graph.aliases,
            self.plans if compiled is None else compiled.graph.plans,
            duration_provider=self.provider(self.work) if compiled is None else compiled.duration,
            release_provider=self.provider(self.releases) if compiled is None else compiled.release,
            phase_provider=self.provider(self.phases) if compiled is None else compiled.phase,
            scheduler=Scheduler(),lifecycle_provider=Lifecycle(),power_provider=Resources(),
            memory_provider=Resources(),require_energy=True,require_memory=True,
            allow_independent_copy_hypothesis=True,allow_no_implicit_default_stream_hypothesis=True)

    def test_complete_and_drain_limited_trajectories_are_identical(self):
        before=copy.deepcopy(self.plans);compiled=self.compiled()
        rows=[dict(alias=a,task='text',pool_id='short',arrival_ns=t,request_id=f'{a}-{t}')
              for t in (0,4,9,17) for a in self.aliases]
        for drain in (40, 1000):
            with self.subTest(drain=drain):
                old=self.engine().forecast(rows,0,30,drain_deadline_ns=drain)
                new=self.engine(compiled).forecast(rows,0,30,drain_deadline_ns=drain)
                self.assertEqual(old,new)
                self.assertEqual(len(new['cohort']),len(rows))
        self.assertEqual(self.plans,before)

    def test_immutable_tables_are_isolated_from_original_mutations(self):
        compiled=self.compiled()
        row=dict(alias='a',task='text',pool_id='short',request_id='another',arrival_ns=999)
        self.work['a','text','short','g']=99999
        self.plans['a','text','short']['nodes'].clear()
        self.assertEqual(compiled.duration(dict(id='g'),row),11)
        self.assertEqual(len(compiled.graph.plans['a','text','short']['nodes']),6)
        with self.assertRaises(TypeError):compiled.graph.plans['a','text','short']['cyclic']=True
        with self.assertRaises(TypeError):compiled.release._values['a','text','short','g']['not_before_offset_ns']=0

    def test_source_changes_fail_closed_and_unknown_coordinates_do_not_fallback(self):
        compiled=self.compiled()
        with self.assertRaisesRegex(ValueError,'support unavailable'):
            compiled.phase(dict(id='unseen'),dict(alias='a',task='text',pool_id='short'))
        self.path.write_text('{"training":false}')
        with self.assertRaisesRegex(ValueError,'source changed'):
            compiled.assert_sources_unchanged()


if __name__=='__main__': unittest.main()
