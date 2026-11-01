"""CPU interval index preserving the trace exporter's launch-owner semantics.

Each globalTid has a balanced start-ordered tree augmented with subtree maxend.
Containment uses both launch endpoints. Overlap alone is insufficient. Records
are retained by reference so appending kernels to an owner updates that record.
The indexed interval/label fields must remain unchanged after construction.
"""
from __future__ import annotations

from dataclasses import dataclass


def _integer(value, name):
    if type(value) is not int:
        raise ValueError('Integer ' + name + ' required')
    return value


@dataclass(frozen=True, slots=True)
class _Entry:
    ordinal: int
    start: int
    end: int
    stage: bool
    record: dict


@dataclass(frozen=True, slots=True)
class _Node:
    entry: _Entry
    maxend: int
    left: object
    right: object


def _build(entries, left, right):
    if left >= right:
        return None
    middle = (left+right)//2
    lo, hi = _build(entries, left, middle), _build(entries, middle+1, right)
    entry = entries[middle]
    return _Node(entry, max(entry.end, lo.maxend if lo else entry.end,
                            hi.maxend if hi else entry.end), lo, hi)


class NVTXRangeIndex:
    """Independent thread trees; no assumption that ranges form a proper stack.

    Build costs O(R log R) time and O(R) storage. Queries prune subtrees whose
    maximum end cannot contain the launch and branches whose starts are too
    late. A query can still visit O(R) nodes when many ranges contain it; output
    size, nesting and crossing intervals determine that cost. Statistics expose
    node visits instead of claiming constant-time attribution. owner_for_launch
    streams matches without sorting; containing restores source order.
    """

    def __init__(self, ranges):
        grouped = {}
        for ordinal, record in enumerate(ranges):
            tid = _integer(record['global_tid'], 'globalTid')
            start = _integer(record['nsys_start_ns'], 'range start')
            end = _integer(record['nsys_end_ns'], 'range end')
            if end < start:
                raise ValueError('Range end precedes start')
            grouped.setdefault(tid, []).append(_Entry(ordinal, start, end, bool(record.get('stage')), record))
        self.range_count = sum(map(len, grouped.values()))
        self.thread_sizes = {tid: len(entries) for tid, entries in grouped.items()}
        self.roots = {}
        for tid, entries in grouped.items():
            entries.sort(key=lambda item: (item.start, item.end, item.ordinal))
            self.roots[tid] = _build(entries, 0, len(entries))

    def _matches(self, global_tid, start_ns, end_ns, stats):
        _integer(global_tid, 'launch globalTid')
        _integer(start_ns, 'launch start')
        _integer(end_ns, 'launch end')
        if end_ns < start_ns:
            raise ValueError('Launch end precedes start')
        if stats is not None:
            stats.update(visited_nodes=0, pruned_subtrees=0, matched_ranges=0,
                         thread_range_count=self.thread_sizes.get(global_tid, 0))
        root = self.roots.get(global_tid)
        pending = [root] if root else []
        while pending:
            node = pending.pop()
            if stats is not None:
                stats['visited_nodes'] += 1
            if node.maxend < end_ns:
                if stats is not None:
                    stats['pruned_subtrees'] += 1
                continue
            entry = node.entry
            if entry.start > start_ns:
                if stats is not None:
                    stats['pruned_subtrees'] += int(node.right is not None)
                if node.left:
                    pending.append(node.left)
                continue
            if entry.end >= end_ns:
                if stats is not None:
                    stats['matched_ranges'] += 1
                yield entry
            if node.right:
                pending.append(node.right)
            if node.left:
                pending.append(node.left)

    def containing(self, global_tid, start_ns, end_ns, *, stats=None):
        """All containing ranges, in the original source order (ties retained)."""
        matches = sorted(self._matches(global_tid, start_ns, end_ns, stats), key=lambda entry: entry.ordinal)
        return [entry.record for entry in matches]

    def owner_for_launch(self, global_tid, start_ns, end_ns, *, stats=None):
        """Prefer a stage, then shortest duration, then first source occurrence.

        Returns the original range dict or None. Duplicate equal ranges are
        distinct candidates; stable source order exactly matches Python min in
        trace_export.parse_sqlite. A nonstage inner range cannot displace a
        containing stage, and a crossing launch must fit both endpoints.
        """
        best, best_key = None, None
        for entry in self._matches(global_tid, start_ns, end_ns, stats):
            key = (not entry.stage, entry.end-entry.start, entry.ordinal)
            if best_key is None or key < best_key:
                best, best_key = entry.record, key
        return best
