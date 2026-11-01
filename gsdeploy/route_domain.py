"""One finite rational routing domain shared by all search backends.

Fine denominators extend two-family rows only; they never multiply the
three-or-more-family composition grid. Capacity witnesses are separate.
"""

import itertools
from fractions import Fraction


def validate_pair_denominators(value, known_tasks):
    if not isinstance(value, dict) or set(value) - set(known_tasks):
        raise ValueError("route_pair_denominators_by_task must map known tasks to lists")
    for task, denominators in value.items():
        if not isinstance(denominators, list) or any(
            type(d) is not int or d < 2 for d in denominators
        ):
            raise ValueError(f"route_pair_denominators_by_task.{task} must contain integers >=2")


def compositions(total, size):
    if size == 1:
        yield (total,)
    elif size > 1:
        for first in range(total + 1):
            for tail in compositions(total - first, size - 1):
                yield (first,) + tail


def fraction_rows(models, K, pair_denominators=()):
    """Yield unique tuples ``(model, Fraction)`` in a deterministic order.

    Base domain: all nonnegative family shares with a common denominator
    1..K. Extension: for every unordered family pair and declared denominator
    d, all positive splits u/d, (d-u)/d, u=1..d-1. Pure rows stay in the base
    domain. Family roles do not restrict any pair. Deduplication is exact.
    """
    if type(K) is not int or not 1 <= K <= 16:
        raise ValueError("route_grid must be an integer in [1,16]")
    validate_pair_denominators({"task": list(pair_denominators)}, {"task"})
    models = tuple(models)
    if len(set(models)) != len(models):
        raise ValueError("routing models must be unique")
    seen = set()
    for denominator in range(1, K + 1):
        for size in range(1, min(denominator, len(models)) + 1):
            for active in itertools.combinations(models, size):
                for extra in compositions(denominator - size, size):
                    row = tuple((m, Fraction(u + 1, denominator))
                                for m, u in zip(active, extra))
                    if row not in seen:
                        seen.add(row)
                        yield row
    for denominator in sorted(set(pair_denominators)):
        for first, second in itertools.combinations(models, 2):
            for units in range(1, denominator):
                row = ((first, Fraction(units, denominator)),
                       (second, Fraction(denominator - units, denominator)))
                if row not in seen:
                    seen.add(row)
                    yield row


def domain_description(K, pair_denominators):
    return {
        "route_domain_schema": "base_grid_plus_pair_grids_v1",
        "route_grid": K,
        "route_pair_denominators_by_task": {
            t: sorted(set(ds)) for t, ds in sorted(pair_denominators.items())
        },
        "pair_extension": "all positive u/d splits for every supported family pair; no role restriction",
        "deduplication": "exact rational shares before multiplication by workload",
    }
