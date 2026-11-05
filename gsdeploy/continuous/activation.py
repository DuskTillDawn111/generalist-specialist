"""Flow activation is separate from deployment residency."""

import itertools
import math


def model_loads(routing):
    models = sorted({m for row in routing.values() for m in row})
    return {m: math.fsum(row.get(m, 0) for row in routing.values()) for m in models}


def active_models(routing):
    return tuple(m for m, rate in model_loads(routing).items() if rate > 0)


def deployment_choices(problem, routing):
    active = set(active_models(routing))
    return {m: tuple(n for n in ns if n > 0 or m not in active)
            for m, ns in problem.states.items()}


def deployments(problem, routing):
    choices = deployment_choices(problem, routing)
    for states in itertools.product(*choices.values()):
        yield {m: n for m, n in zip(choices, states) if n}
