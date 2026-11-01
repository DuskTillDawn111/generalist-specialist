"""Resolve historical receipts to existing, byte-identical source snapshots.

This never changes historical reports or treats the currently edited file as
its earlier version. A prospective protocol can bind the surviving original
bytes and an explicit substitution manifest, alongside current executing code.
"""
from __future__ import annotations

import json
from pathlib import Path

from .native_collect import receipt

SCHEMA = 'gsdeploy.historical_receipt_snapshot_resolution.v4'


def _reference(value):
    return (isinstance(value, dict) and set(value) == {'path', 'sha256'}
            and isinstance(value['path'], str) and isinstance(value['sha256'], str))


class HistoricalReceiptResolver:
    def __init__(self, manifest_path=None):
        self.mapping = {}
        self.manifest_receipt = None
        if manifest_path is None:
            return
        self.manifest_receipt = receipt(manifest_path)
        doc = json.loads(Path(manifest_path).read_text())
        if (doc.get('schema') != SCHEMA or doc.get('historical_artifacts_rewritten') is not False
                or not doc.get('mappings')):
            raise ValueError('Explicit unchanged historical snapshot manifest required')
        for item in doc['mappings']:
            if set(item) != {'historical', 'snapshot'} or not all(_reference(r) for r in item.values()):
                raise ValueError('Exact historical/snapshot receipts required')
            old, snapshot = item['historical'], item['snapshot']
            key = old['path'], old['sha256']
            if key in self.mapping:
                raise ValueError('Duplicate historical receipt mapping')
            if old['sha256'] != snapshot['sha256'] or receipt(snapshot['path']) != snapshot:
                raise ValueError('Historical snapshot must contain the exact original bytes')
            self.mapping[key] = dict(snapshot)

    def resolve(self, ref):
        if not _reference(ref):
            raise ValueError('Exact source receipt required')
        snapshot = self.mapping.get((ref['path'], ref['sha256']))
        if snapshot is not None:
            if receipt(snapshot['path']) != snapshot:
                raise ValueError('Historical snapshot changed after manifest verification')
            return dict(snapshot)
        if receipt(ref['path']) != ref:
            raise ValueError('Changed receipt without an exact historical snapshot: ' + ref['path'])
        return dict(ref)


def resolve_closure(paths, manifest_path=None):
    resolver = HistoricalReceiptResolver(manifest_path)
    pending = [receipt(path) for path in paths]
    if resolver.manifest_receipt:
        pending.append(resolver.manifest_receipt)
    seen, materialized, substitutions = set(), {}, {}
    while pending:
        original = pending.pop()
        key = original['path'], original['sha256']
        if key in seen:
            continue
        seen.add(key)
        resolved = resolver.resolve(original)
        resolved_key = resolved['path'], resolved['sha256']
        if original != resolved:
            substitutions[key] = dict(historical=original, snapshot=resolved)
        if resolved_key in materialized:
            continue
        materialized[resolved_key] = resolved
        if Path(resolved['path']).suffix != '.json':
            continue
        stack = [json.loads(Path(resolved['path']).read_text())]
        while stack:
            node = stack.pop()
            if _reference(node):
                pending.append(node)
            elif isinstance(node, dict):
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
    refs = list(materialized.values())
    if any(receipt(ref['path']) != ref for ref in refs):
        raise ValueError('Historical closure changed during traversal')
    return dict(schema='gsdeploy.resolved_evidence_closure.v4', status='RESOLVED_BYTES_VERIFIED',
                materialized_sources=refs, historical_substitutions=list(substitutions.values()),
                resolution_manifest=resolver.manifest_receipt,
                historical_artifacts_rewritten=False, current_execution_equivalence_claimed=False,
                historical_direct_path_closure_pass_claimed=False)
