"""Opt-in convex interpolation of observed stage/resource contexts.

This is a declared statistical hypothesis, not a physical interference bound.
It never extrapolates counts or SM/DRAM/L2 demand, transfers across execution
identities or invents missing counters. Exact context evidence takes priority.
"""

from collections import OrderedDict
import json

import numpy as np
from ..linear_solver import linprog

from ..forecast_errors import UnknownEvidence


SCHEMA = 'gsdeploy.convex_context_law.v1'
KINDS = ('host', 'gpu', 'sync', 'cleanup')


class ConvexContextLaw:
    def __init__(self, document, profiles):
        spec = document['context_interpolation']
        if (set(spec) != {'schema', 'method', 'scenarios'} or spec['schema'] != SCHEMA
                or spec['method'] != 'convex_observed_resource_features'
                or not isinstance(spec['scenarios'], list) or not spec['scenarios']
                or any(not isinstance(s, str) or not s for s in spec['scenarios'])
                or len(set(spec['scenarios'])) != len(spec['scenarios'])):
            raise ValueError('Explicit convex context interpolation specification required')
        self.scenarios = set(spec['scenarios'])
        self.profiles = profiles
        self.contexts = document['contexts']
        self.cache = OrderedDict()

    def features(self, members, *, omit=None):
        values = [0.] * 7
        for alias, member in members.items():
            if alias == omit:
                continue
            profile = self.profiles[member['profile_id']]
            segment = next(s for s in profile['segments'] if s['id'] == member['segment_id'])
            if any(v is None for v in segment['demand'].values()):
                return None
            values[KINDS.index(segment['kind'])] += 1.
            for j, name in enumerate(('sm', 'dram', 'l2'), 4):
                values[j] += segment['demand'][name]
        return values

    def mix(self, active, scenario, *, target=None):
        """Interpolate only inside the observed feature hull for this target."""
        features = self.features(active, omit=target)
        donors = []
        member = ({k:active[target][k] for k in ('profile_id','segment_id')}
                  if target is not None else None)
        for key, context in sorted(self.contexts.items()):
            value = context['scenarios'].get(scenario)
            if value is None or target is not None and context['members'].get(target) != member:
                continue
            if target is None and any(value[n] is None for n in ('board_watts', 'peak_memory_mib')):
                continue
            vertex = self.features(context['members'], omit=target)
            if vertex is not None:
                donors.append((key, vertex, value))
        if features is None or len(donors) < 2:
            raise UnknownEvidence('UNKNOWN_CONTEXT_LAW_FEATURES_OR_TRAINING_SUPPORT')
        matrix = np.array([[1.] + d[1] for d in donors]).T
        rhs = np.array([1.] + features)
        fit = linprog(np.zeros(len(donors)), A_eq=matrix, b_eq=rhs, bounds=(0., None),
                      method='highs', options={'time_limit': .1, 'primal_feasibility_tolerance': 1e-9})
        if (not fit.success or np.max(np.abs(matrix @ fit.x - rhs)) > 1e-8
                or min(fit.x) < 0):
            raise UnknownEvidence('UNKNOWN_CONTEXT_LAW_OUTSIDE_OBSERVED_HULL')
        weights = fit.x / sum(fit.x)
        support = [(d, float(w)) for d, w in zip(donors, weights) if w > 0]
        if target is not None:
            result = sum(w * d[2]['progress'][target] for d, w in support)
        else:
            result = dict(board_watts=sum(w * d[2]['board_watts'] for d, w in support),
                          peak_memory_mib=max(d[2]['peak_memory_mib'] for d, w in support))
        return result, [dict(context_key=d[0], weight=w) for d, w in support]

    def evaluate(self, active, scenario, key):
        cache_key = key, scenario
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        if scenario not in self.scenarios:
            raise UnknownEvidence('UNKNOWN_CONTEXT_LAW_SCENARIO:' + scenario)
        resources, evidence = self.mix(active, scenario)
        progress, sources = {}, {'resources': evidence}
        for alias in sorted(active):
            progress[alias], sources[alias] = self.mix(active, scenario, target=alias)
        result = dict(progress=progress, **resources), json.dumps(dict(
            method=SCHEMA, physical_bound=False, independent_accuracy_qualified=False,
            support=sources), sort_keys=True)
        self.cache[cache_key] = result
        if len(self.cache) > 128:
            self.cache.popitem(last=False)
        return result
