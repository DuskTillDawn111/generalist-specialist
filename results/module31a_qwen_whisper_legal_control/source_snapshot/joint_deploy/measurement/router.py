"""Smooth weighted round robin; deterministic, task-specific state."""

class Router:
    def __init__(self, routing_rps):
        self.weights = {task: {m: rate for m, rate in sorted(route.items()) if rate > 0}
                        for task, route in routing_rps.items()}
        self.deficits = {task: {m: 0.0 for m in route} for task, route in self.weights.items()}

    def choose(self, task):
        weights, deficits = self.weights[task], self.deficits[task]
        for model, weight in weights.items():
            deficits[model] += weight
        chosen = max(weights, key=lambda model: deficits[model])
        deficits[chosen] -= sum(weights.values())
        return chosen
