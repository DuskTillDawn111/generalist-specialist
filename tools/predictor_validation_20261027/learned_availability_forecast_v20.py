"""Retain actual engine returns when the original complete-cohort score refuses.

This adds diagnostic evidence to v19. UNKNOWN, its reason, score eligibility,
all numerical predictions and the original six-term scorer remain unchanged.
An engine exception never creates a partial or invented engine return.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .learned_availability_forecast_v19 import (
    LearnedAvailabilityForecast as V19Forecast,
    implementation_sources as v19_sources)
from .native_collect import receipt

SCHEMA = 'gsdeploy.unscored_actual_forecast_retention.v20'


def implementation_sources():
    return [*v19_sources(), receipt(__file__)]


def verify_sources(sources):
    """Hash actual bytes: this filesystem can preserve stat after an edit."""
    for ref in sources:
        sha = hashlib.sha256()
        with Path(ref['path']).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                sha.update(block)
        if sha.hexdigest() != ref['sha256']:
            raise ValueError('Forecast evidence source bytes changed: '+ref['path'])


class RetainReturnedForecast:
    def __init__(self, engine, sources=()):
        self._engine = engine
        self.last_returned_forecast = None
        self.sources = tuple(dict(ref) for ref in sources)

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def forecast(self, *args, **kwargs):
        # Reset before entry so a later failed call cannot expose a stale return.
        self.last_returned_forecast = None
        verify_sources(self.sources)
        try:
            result = self._engine.forecast(*args, **kwargs)
            self.last_returned_forecast = result
            return result
        finally:
            verify_sources(self.sources)


def attach_unscored_diagnostic(result, engine):
    if (result.get('status') == 'UNKNOWN' and 'forecast' not in result
            and engine.last_returned_forecast is not None):
        if 'metrics' in result or 'unscored_forecast_diagnostic' in result:
            raise ValueError('Original unscored UNKNOWN result without existing diagnostic required')
        result['unscored_forecast_diagnostic'] = dict(
            schema=SCHEMA, forecast=engine.last_returned_forecast,
            score_rejection_reason=result.get('reason'), diagnostic_only=True,
            complete_prediction_or_gate_evidence=False,
            score_recomputed=False, completion_or_drain_changed=False)
    return result


class LearnedAvailabilityForecast(V19Forecast):
    def engine(self, *, max_events):
        engine = super().engine(max_events=max_events)
        by_path = {}
        for ref in [*implementation_sources(), *engine.sources]:
            if ref['path'] in by_path and by_path[ref['path']] != ref:
                raise ValueError('Conflicting implementation or training source receipts')
            by_path[ref['path']] = dict(ref)
        wrapped = RetainReturnedForecast(engine, tuple(by_path.values()))
        self._latest_retention_engine = wrapped
        return wrapped

    def execute(self, *args, **kwargs):
        self._latest_retention_engine = None
        result = super().execute(*args, **kwargs)
        engine = self._latest_retention_engine
        if engine is None:
            raise ValueError('Original engine entry required for diagnostic retention')
        result = attach_unscored_diagnostic(result, engine)
        verify_sources(engine.sources)
        return result
