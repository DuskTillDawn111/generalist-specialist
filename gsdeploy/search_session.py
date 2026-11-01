"""Isolated digest memoization around unmodified frozen mathematical sources.

Only a private module namespace sees the memo. Each session owns a deep-frozen
JSON input tree; ordinary mutable public inputs always retain their full gates.
This module does not fit models, change scores/bounds, or invoke GPU APIs.
"""

import copy
import importlib
import itertools
import math
from pathlib import Path
import sys
import types
import uuid


def _reject(*args, **kwargs):
    raise TypeError("Search session inputs are immutable")


class FrozenDict(dict):
    __slots__ = ("_session_token",)
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _reject
    __setattr__ = __delattr__ = _reject

    def __init__(self, *args, **kwargs):
        if hasattr(self, "_session_token"):
            _reject()
        dict.__init__(self, *args, **kwargs)

    def __deepcopy__(self, memo):
        return self


class FrozenList(list):
    __slots__ = ("_session_token",)
    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = _reject
    __iadd__ = __imul__ = __setattr__ = __delattr__ = _reject

    def __init__(self, *args):
        if hasattr(self, "_session_token"):
            _reject()
        list.__init__(self, *args)

    def __deepcopy__(self, memo):
        return self


def freeze(value, counter=None):
    """Freeze a copied JSON tree; tokens are unique only within its session."""
    counter = (uuid.uuid4().hex, itertools.count()) if counter is None else counter
    if isinstance(value, dict):
        frozen = FrozenDict({key: freeze(v, counter) for key, v in value.items()})
    elif isinstance(value, list):
        frozen = FrozenList(freeze(v, counter) for v in value)
    elif value is None or type(value) in (str, bool, int, float):
        return value
    else:
        raise TypeError("Search session expects JSON input, got " + type(value).__name__)
    object.__setattr__(frozen, "_session_token", (counter[0], next(counter[1])))
    return frozen


class DigestMemo:
    """Cache exact original digests using immutable nodes and value keys.

    Temporary dict/list keys are rebuilt by structure. They never use object
    identity, so ephemeral allocation/id reuse cannot return an old digest.
    """
    def __init__(self, original):
        self.original, self.cache = original, {}
        self.hits = self.misses = 0

    def key(self, value):
        if isinstance(value, (FrozenDict, FrozenList)):
            return ("frozen", value._session_token)
        if isinstance(value, dict):
            return ("dict", tuple(sorted((self.key(k), self.key(v)) for k, v in value.items())))
        if isinstance(value, (list, tuple)):
            return ("list", tuple(self.key(v) for v in value))
        if value is None:
            return ("null",)
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("Nonfinite digest input")
            return ("float", value.hex())
        if type(value) in (str, bool, int):
            return (type(value).__name__, value)
        raise TypeError("Unsupported digest input: " + type(value).__name__)

    def __call__(self, value):
        key = self.key(value)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        result = self.original(value)
        self.cache[key] = result
        return result


class SearchSession:
    """Load unchanged sources privately, validate once, freeze, then memoize.

    ``config`` is a plain JSON config dictionary with an inline catalog. A new
    session revalidates every input; no cache is shared with previous sessions.
    All ordinary public prediction/validation functions remain untouched.
    """
    def __init__(self, source_root, config, *, models=None):
        self.source_root = Path(source_root).resolve()
        self.prefix = "_gsdeploy_search_" + uuid.uuid4().hex
        package = types.ModuleType(self.prefix)
        package.__path__ = [str(self.source_root / "gsdeploy")]
        package.__package__ = self.prefix
        sys.modules[self.prefix] = package
        self.closed = False
        try:
            for name in ("quality", "calibration", "calibration_v2", "config", "prediction",
                         "surrogate_bounds", "optimizer", "milp", "system"):
                self.module(name)
            raw = copy.deepcopy(config)
            source_fields = self.module("config").SystemConfig.__dataclass_fields__
            if "route_pair_denominators_by_task" not in source_fields:
                if raw.get("route_pair_denominators_by_task"):
                    raise ValueError("SOURCE_ROUTE_DOMAIN_UNSUPPORTED: frozen config has no pair routing domain")
                raw.pop("route_pair_denominators_by_task", None)
            if raw["catalog"].get("performance_mode") == "replay_transfer_v1":
                self.module("replay_transfer")
            # Original full gates run without memo on copied ordinary values.
            self.module("config").SystemConfig(**raw)
            self.validate(raw)
            if models is not None:
                for field in ("models", "capacity_factors", "memory_table_mib"):
                    raw["catalog"][field] = {m: raw["catalog"][field][m] for m in models}
                self.validate(raw)
            self.input = freeze(raw)
            self.config = self.module("config").SystemConfig(**self.input)
            self.memo = DigestMemo(self.module("quality").quality_digest)
            original = self.module("quality").quality_digest
            for name, module in tuple(sys.modules.items()):
                if name.startswith(self.prefix + "."):
                    for key, value in tuple(vars(module).items()):
                        if value is original:
                            setattr(module, key, self.memo)
        except BaseException:
            self.close()
            raise

    def module(self, name):
        if self.closed:
            raise RuntimeError("Search session is closed")
        return importlib.import_module(self.prefix + "." + name)

    def validate(self, raw):
        mode = raw["catalog"].get("performance_mode")
        if mode == "calibrated_surrogate":
            return self.module("calibration").validated_spec(raw["catalog"], raw["workload"])
        if mode == "calibrated_replay_v2":
            return self.module("calibration_v2").validated_replay_spec(raw["catalog"], raw["workload"])
        if mode == "replay_transfer_v1":
            return self.module("replay_transfer").validated_transfer_spec(raw["catalog"], raw["workload"])
        raise ValueError("Search sessions require an explicitly validated calibrated predictor")

    def context(self):
        c = self.config
        return dict(horizon=c.horizon_seconds, seed=c.random_seed, arrival_process=c.arrival_process,
                    drain_seconds=c.drain_seconds, energy_reference_watts=c.energy_reference_watts,
                    startup_amortization_seconds=c.startup_amortization_seconds)

    def predict(self, deployment, routing):
        c = self.config
        return self.module("prediction").predict_scenarios(c.catalog, deployment, routing,
            c.workload, c.gpu_memory_mib, c.latency_reference_ms, c.weights, **self.context())

    def optimizer(self, algorithm, **options):
        c = self.config
        kwargs = dict(search_max_states=c.search_max_states, search_policy=c.search_policy,
                      milp_capacity_witness=c.milp_capacity_witness, **self.context())
        pair_denominators = getattr(c, "route_pair_denominators_by_task", {})
        if pair_denominators:
            kwargs["route_pair_denominators_by_task"] = pair_denominators
        if algorithm == "milp":
            cls = self.module("milp").JointMilpOptimizer
            kwargs.update(milp_time_limit_seconds=c.milp_time_limit_seconds,
                milp_max_iterations=c.milp_max_iterations, milp_max_columns=c.milp_max_columns,
                milp_absolute_gap=c.milp_absolute_gap)
        elif algorithm in {"dfs", "plain_dfs"}:
            cls = self.module("optimizer").UtilityOptimizer
            kwargs.update(dfs_time_limit_seconds=c.milp_time_limit_seconds,
                          dfs_bound_pruning=algorithm != "plain_dfs")
        else:
            raise ValueError("Unknown algorithm: " + str(algorithm))
        kwargs.update(options)
        if (kwargs.get("route_pair_denominators_by_task")
                and "route_pair_denominators_by_task" not in c.__dataclass_fields__):
            raise ValueError("SOURCE_ROUTE_DOMAIN_UNSUPPORTED: frozen optimizer has no pair routing domain")
        return cls(c.catalog, c.workload, c.gpu_memory_mib, c.latency_reference_ms, c.weights, **kwargs)

    def stats(self):
        return dict(digest_hits=self.memo.hits, digest_misses=self.memo.misses,
                    digest_entries=len(self.memo.cache), private_module_prefix=self.prefix,
                    source_root=str(self.source_root))

    def close(self):
        for name in tuple(sys.modules):
            if name == self.prefix or name.startswith(self.prefix + "."):
                del sys.modules[name]
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
