"""Seal direct resource hypotheses with their entire training evidence closure.

This preserves the v5 resource estimates and original artifacts byte for byte.
The new schema adds provider-time checks of nested reviewers, training indices
and exact historical snapshots. It changes no physical model or fit threshold.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from . import natural_resources_v5 as original
from .historical_evidence_v4 import resolve_closure
from .native_collect import receipt, save

SCHEMA = 'gsdeploy.natural_direct_state_resource_hypothesis.v6'


def _read(ref):
    if receipt(ref['path']) != ref:
        raise ValueError('Changed resource evidence: ' + ref['path'])
    return json.loads(Path(ref['path']).read_text())


def verify_training_index(model):
    index = _read(model['training_index'])
    if index.get('schema') != 'gsdeploy.independent_prediction_gate_training_index.v2' or index.get('split') != 'training':
        raise ValueError('Explicit resource training index required')
    expected, captures = [], set()
    for capture in model['captures']:
        raw = _read(capture['raw'])
        run = 'raw-sha256:' + capture['raw']['sha256']
        if capture['run_id'] != run or run in captures:
            raise ValueError('Unique actual raw resource capture identities required')
        captures.add(run)
        ids = set()
        for row in raw['request_records']:
            if row.get('split') != 'training' or row.get('status') != 'completed' or row['request_id'] in ids:
                raise ValueError('Unique completed training requests required')
            ids.add(row['request_id'])
            expected.append(dict(run_id=run, request_id=row['request_id'], pool_id=row['pool_id'], task=row['task']))
    if index.get('records') != expected or not expected:
        raise ValueError('Resource training index must exactly cover every fitted capture')


def _closure(base, resolution):
    if resolution is not None:
        _read(resolution)
    return resolve_closure([base['path']], resolution['path'] if resolution else None)


def seal(base_model_path, output, *, historical_resolution_path=None):
    base = receipt(base_model_path)
    model = _read(base)
    if model.get('schema') != original.SCHEMA:
        raise ValueError('Unmodified direct-state v5 model required')
    verify_training_index(model)
    resolution = receipt(historical_resolution_path) if historical_resolution_path else None
    closure = _closure(base, resolution)
    result = copy.deepcopy(model)
    result.update(schema=SCHEMA, base_model=base, historical_resolution=resolution,
                  training_evidence_closure=closure,
                  closure_evaluator_sources=[receipt(__file__), receipt(Path(__file__).with_name('historical_evidence_v4.py'))],
                  resource_estimates_changed=False, training_index_exact_coverage_checked=True,
                  nested_training_evidence_checked_at_provider_construction=True)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'model.json', result)
    return result


class NaturalResourceProvider(original.NaturalResourceProvider):
    def __init__(self, model, target_context, **options):
        if model.get('schema') != SCHEMA:
            raise ValueError('Sealed v6 resource hypothesis required')
        if not model.get('closure_evaluator_sources') or any(receipt(s['path']) != s for s in model['closure_evaluator_sources']):
            raise ValueError('Changed resource closure evaluator')
        base = _read(model['base_model'])
        verify_training_index(base)
        closure = _closure(model['base_model'], model['historical_resolution'])
        if closure != model['training_evidence_closure']:
            raise ValueError('Resource nested training evidence closure changed')
        # Compare every original field, including estimates, raw evidence and
        # index receipt; a new wrapper cannot silently replace the old fit.
        if any(model.get(k) != v for k, v in base.items() if k != 'schema'):
            raise ValueError('Sealed resource model differs from original estimates/evidence')
        super().__init__(base, target_context, **options)
        self.document = copy.deepcopy(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--historical-resolution')
    args = parser.parse_args()
    result = seal(args.base_model, args.output, historical_resolution_path=args.historical_resolution)
    print(json.dumps(dict(states=len(result['states']), supported=sum(s['known'] for s in result['states'].values()),
                          closure_sources=len(result['training_evidence_closure']['materialized_sources']),
                          original_estimates_changed=False)))


if __name__ == '__main__':
    main()
