"""Content-addressed immutable DAG models and source receipt verification cache.

Initialization hashes model content once. Nested read-only maps and tuples keep
the cached identity truthful; mutable tables never receive a cached digest.
Source files are rehashed if their inode/size/mtime/ctime fingerprint changes.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

from .dag_checkpoint_v7 import receipt


def plain(value):
    if isinstance(value, Mapping):
        return {plain(k): plain(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return tuple(plain(v) for v in value)
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError('Explicit finite static model values required: '+type(value).__name__)


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({freeze(k): freeze(v) for k, v in value.items()})
    if type(value) in (list, tuple):
        return tuple(freeze(v) for v in value)
    return plain(value)


def static_digest(value):
    # Scalar type tagging in legacy checkpoints multiplies large CUDA plan
    # object counts. JSON already distinguishes its finite scalar values.
    # Tag containers only, preserving mapping key types and original order.
    def wire(item):
        if isinstance(item,Mapping):
            return ['mapping',[[wire(k),wire(v)] for k,v in item.items()]]
        if type(item) in (list,tuple):
            return ['sequence',[wire(v) for v in item]]
        return plain(item)
    return hashlib.sha256(json.dumps(wire(value),separators=(',',':'),allow_nan=False).encode()).hexdigest()


class FrozenDagModel:
    __slots__ = ('aliases', 'plans', 'aliases_sha256', 'plans_sha256', '_sealed')

    def __init__(self, aliases, plans):
        object.__setattr__(self, '_sealed', False)
        self.aliases_sha256 = static_digest(aliases)
        self.plans_sha256 = static_digest(plans)
        self.aliases = freeze(aliases)
        self.plans = freeze(plans)
        object.__setattr__(self, '_sealed', True)

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            raise AttributeError('Frozen original DAG model cannot change')
        object.__setattr__(self, name, value)

    def __delattr__(self,name):
        raise AttributeError('Frozen original DAG model fields cannot be deleted')


class SourceReceiptCache:
    """Memoize checked bytes only while filesystem change indicators agree."""
    def __init__(self):
        self._checked = {}

    def verify(self, reference):
        if type(reference) is not dict or set(reference) != {'path', 'sha256'}:
            raise ValueError('Exact source receipt required')
        path=Path(reference['path']).resolve()
        before=path.stat()
        signature=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
        key=(str(path),reference['sha256'])
        if self._checked.get(key) != signature:
            if receipt(path) != reference:
                raise ValueError('Frozen model source changed: '+str(path))
            after=path.stat()
            if signature != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                raise ValueError('Model source changed during receipt verification')
            self._checked[key]=signature
        return reference


SOURCE_CACHE=SourceReceiptCache()


class FrozenNodeTable:
    """Immutable numeric/release/phase table materialized from a checked model.

    Materialization provenance remains the caller's declared model evidence.
    A frozen table is a CPU execution representation, not an accuracy claim.
    """
    __slots__ = ('_values','table_sha256','sources','domain','_sealed')

    def __init__(self, values, *, sources, domain):
        object.__setattr__(self, '_sealed', False)
        if not sources or not isinstance(domain,str) or not domain:
            raise ValueError('Explicit source-bound table domain required')
        for ref in sources:
            SOURCE_CACHE.verify(ref)
        self.table_sha256=static_digest(values)
        self._values=freeze(values)
        self.sources=freeze(sources)
        self.domain=domain
        object.__setattr__(self, '_sealed', True)

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            raise AttributeError('Frozen node provider cannot change')
        object.__setattr__(self, name, value)

    def __delattr__(self,name):
        raise AttributeError('Frozen node provider fields cannot be deleted')

    def __call__(self, node, row):
        coordinate=(row['alias'],row['task'],row['pool_id'],node['id'])
        if coordinate not in self._values:
            raise ValueError('Original public-input node table support unavailable')
        return plain(self._values[coordinate])

    def checkpoint_configuration(self):
        return dict(table_sha256=self.table_sha256,domain=self.domain,sources=plain(self.sources),
            mutable_model_state=False)
