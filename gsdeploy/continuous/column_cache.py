"""Bounded exact Joint task-column reuse; no feedback or master-state reuse."""

from collections import OrderedDict
import copy
import json

from .contracts import digest


class JointColumnCache:
    def __init__(self, problem_identity, oracle_version, *, max_bytes=16*1024*1024):
        self.identity = digest([problem_identity,oracle_version,'joint-columns-v1'])
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = 0

    def call(self, original, optimizer, choices, K, *, no_split=False):
        # capacity_witness can depend on a particular optimizer's solve setup.
        # Include its exact content and effective column limit, never assume
        # columns are solely a function of the denominator.
        key = digest([self.identity,choices,K,no_split,optimizer.witness,
                      optimizer.column_limit,optimizer.policy,optimizer.pair_denominators])
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            columns,metrics,_ = self.entries[key]
            for k,v in metrics.items():
                optimizer.metrics[k] = v
            return copy.deepcopy(columns)
        self.misses += 1
        columns = original(choices,K,no_split=no_split)
        if columns is None:
            return None  # Partial/timed-out construction is never cached.
        metrics = {k:optimizer.metrics[k] for k in ('task_columns','column_memory_pruned') if k in optimizer.metrics}
        size = len(json.dumps([columns,metrics],sort_keys=True).encode())+len(key)
        if size<=self.max_bytes:
            self.entries[key] = copy.deepcopy(columns),metrics,size
            self.bytes += size
            while self.bytes>self.max_bytes:
                _,(_,_,removed) = self.entries.popitem(last=False)
                self.bytes -= removed
        return columns

    def report(self):
        return dict(hits=self.hits,misses=self.misses,entries=len(self.entries),serialized_bytes=self.bytes)
