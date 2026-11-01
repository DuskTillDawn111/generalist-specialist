"""Materialize coordinate-only providers before full-horizon DAG simulation.

This is an execution representation, not a new fit. Callers must establish
that their providers depend only on (alias, task, pool_id, original node).
Every provider is called on the original node before metadata is removed;
every original node and edge survives in a read-only execution graph.
"""
from __future__ import annotations

import math

from . import dag_frozen_model_v10, native_queue_phase_busy_v10
from .dag_frozen_model_v10 import FrozenDagModel, FrozenNodeTable, SOURCE_CACHE, freeze, plain
from .native_collect import receipt
from .native_queue_phase_busy_v10 import compile_original_graph


class CompiledStaticProviders:
    __slots__ = ('graph', 'duration', 'release', 'phase', 'sources', 'manifest', '_sealed')

    def __init__(self, aliases, plans, *, duration, release, phase, sources):
        object.__setattr__(self, '_sealed', False)
        if not sources:
            raise ValueError('Source-bound original providers required')
        sources = [*sources, receipt(__file__), receipt(dag_frozen_model_v10.__file__),
                   receipt(native_queue_phase_busy_v10.__file__)]
        for reference in sources:
            SOURCE_CACHE.verify(reference)
        graph, manifest = compile_original_graph(plans)
        self.graph = FrozenDagModel(aliases, graph)
        work_values, release_values, phase_values = {}, {}, {}
        for coordinate, profile in plans.items():
            row = dict(alias=coordinate[0], task=coordinate[1], pool_id=coordinate[2],
                       request_id='coordinate-only-materialization', arrival_ns=0)
            for node in profile['nodes']:
                key = (*coordinate, node['id'])
                release_values[key] = release(node, row)
                if node['kind'] in ('gpu', 'copy', 'memset'):
                    value = duration(node, row)
                    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                        raise ValueError('Positive finite original reference work required')
                    work_values[key] = value
                if node['kind'] == 'gpu':
                    value = phase(node, row)
                    if not isinstance(value, str) or not value:
                        raise ValueError('Original semantic phase required')
                    phase_values[key] = value
        self.duration = FrozenNodeTable(work_values, sources=sources, domain='ORIGINAL_REFERENCE_WORK_NS')
        self.release = FrozenNodeTable(release_values, sources=sources, domain='ORIGINAL_HOST_RELEASE_NS')
        self.phase = FrozenNodeTable(phase_values, sources=sources, domain='ORIGINAL_SEMANTIC_PHASE')
        self.sources = freeze(sources)
        self.manifest = freeze(manifest)
        self.assert_sources_unchanged()
        object.__setattr__(self, '_sealed', True)

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            raise AttributeError('Compiled original provider representation cannot change')
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError('Compiled original provider representation cannot change')

    def assert_sources_unchanged(self):
        for reference in self.sources:
            SOURCE_CACHE.verify(plain(reference))

    def diagnostic(self):
        return dict(original_graph_compilation=plain(self.manifest),
                    original_aliases_sha256=self.graph.aliases_sha256,
                    execution_plans_sha256=self.graph.plans_sha256,
                    duration=self.duration.checkpoint_configuration(),
                    release=self.release.checkpoint_configuration(),
                    phase=self.phase.checkpoint_configuration(),
                    nodes_or_edges_dropped=False, fitted_values_changed=False,
                    independent_accuracy_established=False)
