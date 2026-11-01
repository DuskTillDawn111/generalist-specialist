"""v15 learned candidate using immutable, source-verified static DAG tables.

The input configuration and learned scheduler remain v15. This wrapper changes
CPU representation only. It does not repair the physical assumptions or certify
accuracy. The actual original provider classes are coordinate-only; arbitrary
request-dependent providers cannot silently be compiled by this adapter.
"""
from __future__ import annotations

from . import (compiled_static_providers_v16, dag_phase_scheduler_v8,
               dag_preparation_host_release_v8, solo_reference_work_v8,
               static_dag_phase_binding_v8)
from .compiled_static_providers_v16 import CompiledStaticProviders
from .dag_preparation_host_release_v8 import DagPreparationHostReleaseProvider
from .learned_availability_forecast_v15 import LearnedAvailabilityForecast as V15Forecast
from .native_collect import receipt
from .native_queue_phase_busy_v10 import _references
from .solo_reference_work_v8 import SoloOrdinalReferenceWork
from .static_dag_phase_binding_v8 import StaticDagPhaseBinding


class LearnedAvailabilityForecast(V15Forecast):
    def __init__(self, configuration, *, allow_unvalidated_training_hypothesis=False):
        super().__init__(configuration,
                         allow_unvalidated_training_hypothesis=allow_unvalidated_training_hypothesis)
        if (type(self.work) is not SoloOrdinalReferenceWork
                or type(self.release) is not DagPreparationHostReleaseProvider
                or type(self.phases) is not StaticDagPhaseBinding):
            raise ValueError('Only checked original coordinate-only providers can be compiled')
        sources = _references([configuration, self.work.provenance(), self.phases.provenance(),
                               self.host, receipt(__file__),
                               *[receipt(m.__file__) for m in (compiled_static_providers_v16,
                                   dag_phase_scheduler_v8, dag_preparation_host_release_v8,
                                   solo_reference_work_v8, static_dag_phase_binding_v8)]])
        self.compiled = CompiledStaticProviders(self.aliases, self.plans,
            duration=self.work, release=self.release, phase=self.phases, sources=sources)

    def engine(self, *, max_events):
        self.compiled.assert_sources_unchanged()
        engine = super().engine(max_events=max_events)
        engine.aliases = self.compiled.graph.aliases
        engine.plans = self.compiled.graph.plans
        engine.duration_provider = self.compiled.duration
        engine.release_provider = self.compiled.release
        engine.phase_provider = self.compiled.phase
        return engine

    def execute(self, *args, **kwargs):
        self.compiled.assert_sources_unchanged()
        result = super().execute(*args, **kwargs)
        self.compiled.assert_sources_unchanged()
        result['static_provider_compilation'] = self.compiled.diagnostic()
        result['CPU_representation_version'] = 16
        return result
