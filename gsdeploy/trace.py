"""A route-independent cohort shared by planning and execution."""

import hashlib
import heapq
import json
import random


def phases_for(rates, seed):
    return {
        t: ((random.Random(f"{seed}:phase:{t}").random() + 1e-12) % 1) / q
        for t, q in sorted(rates.items())
    }


def periodic_arrivals(rates, duration, seed):
    phases = phases_for(rates, seed)
    heap = [(p, t, 0) for t, p in phases.items() if p < duration]
    heapq.heapify(heap)
    while heap:
        offset, task, index = heapq.heappop(heap)
        yield offset, task
        next_offset = phases[task] + (index + 1) / rates[task]
        if next_offset < duration:
            heapq.heappush(heap, (next_offset, task, index + 1))


def poisson_arrivals(rates, duration, seed):
    rng = {t: random.Random(f"{seed}:arrival:{t}") for t in sorted(rates)}
    heap = [(rng[t].expovariate(rates[t]), t) for t in rng]
    heapq.heapify(heap)
    while heap:
        offset, task = heapq.heappop(heap)
        if offset >= duration:
            break
        yield offset, task
        heapq.heappush(heap, (offset + rng[task].expovariate(rates[task]), task))


def make_trace(rates, horizon=120.0, seed=42, arrival_process="deterministic_periodic"):
    generators = {
        "deterministic_periodic": periodic_arrivals,
        "poisson": poisson_arrivals,
    }
    if arrival_process not in generators:
        raise ValueError("Unknown arrival process")
    return tuple(generators[arrival_process](rates, horizon, seed))


def trace_digest(trace):
    return hashlib.sha256(json.dumps(trace, separators=(",", ":")).encode()).hexdigest()


def expand_routes(deployment, routing):
    aliases = {f"{m}#{i}": m for m, n in sorted(deployment.items()) for i in range(n)}
    routes = {
        t: {
            f"{m}#{i}": q / deployment[m]
            for m, q in sorted(row.items())
            if q > 0
            for i in range(deployment[m])
        }
        for t, row in sorted(routing.items())
    }
    return aliases, routes
