"""Retained semantic worker plus explicit affine-module observation.

This is a separate, opt-in profiling worker/execution identity. It reuses the
unchanged native loader/generate/cleanup and adds request-scoped metadata hooks.
It never enables these hooks in natural endpoint/performance measurements.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.predictor_validation_20261027 import semantic_observer, semantic_worker
from tools.predictor_validation_20261027.operator_observer_v2 import observe_affine_operators

_native_observer = semantic_observer.observe_language_forwards


class CombinedObservation:
    def __init__(self, semantic, operators):
        self.semantic, self.operators = semantic, operators

    def document(self):
        document = self.semantic.document()
        document['affine_operator_observations'] = self.operators.document()
        document['instrumentation_scope'] = 'language forward plus recognized affine leaf calls; not full multimodal operator graph'
        return document


@contextmanager
def combined_observer(executor, request_id, **kwargs):
    # Resolve the same causal-language module as the retained semantic observer.
    # ASR/vision frontend ops and the standalone OCR backend are outside this
    # module boundary and must not inherit these features.
    root = semantic_observer.language_module(executor)
    with _native_observer(executor, request_id, **kwargs) as semantic:
        with observe_affine_operators(root, request_id, nvtx=kwargs.get('nvtx'),
                                      sink=kwargs.get('sink')) as operators:
            yield CombinedObservation(semantic, operators)


def main():
    # Only the instrumentation factory is replaced, for this worker invocation.
    # Model forward/generate methods and parameters are never replaced.
    previous = semantic_observer.observe_language_forwards
    semantic_observer.observe_language_forwards = combined_observer
    try:
        semantic_worker.main()
    finally:
        semantic_observer.observe_language_forwards = previous


if __name__ == '__main__':
    main()
