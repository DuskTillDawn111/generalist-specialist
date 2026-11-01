"""CPU-only resource contracts and launch preparation.

An allocation is a request, never evidence that isolation took effect. This
module neither starts MPS nor creates CUDA contexts. Unimplemented controls
fail closed at launch, while remaining inspectable in experiment packets.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


RESOURCE_SCHEMA = "gsdeploy.resource_configuration.v1"
MODES = {"shared", "mps_cap", "mps_static_sm", "green_context", "request_admission"}
def _mps_keys(environment):
    return {k for k in environment if isinstance(k, str) and k.startswith("CUDA_MPS_")}


def identity(value):
    """Stable finite JSON identity; also used for candidate/cache separation."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError(f"Expected keys {sorted(required)}; optional {sorted(optional)}")


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_resources(document, aliases):
    """Require explicit per-process assignments, including idle residents."""
    _keys(document, {"schema", "mode", "device", "allocations"}, {"mps_session", "sm_capacity", "groups"})
    if (document["schema"] != RESOURCE_SCHEMA or not isinstance(document["mode"], str)
            or document["mode"] not in MODES):
        raise ValueError("Unknown resource schema/mode")
    if not aliases or any(not isinstance(a, str) or not a for a in aliases):
        raise ValueError("Nonempty worker aliases required")
    device = document["device"]
    if not isinstance(device, str) or not device or device != device.strip() or "," in device or device == "-1":
        raise ValueError("Exactly one explicit CUDA device required")
    allocations = document["allocations"]
    if not isinstance(allocations, dict) or set(allocations) != set(aliases):
        raise ValueError("Allocations must cover exactly all resident worker aliases")
    mode = document["mode"]
    if mode == "mps_cap":
        session = document.get("mps_session")
        _keys(session, {"pipe_directory", "log_directory"})
        for path in session.values():
            if not isinstance(path, str) or not Path(path).is_absolute():
                raise ValueError("MPS session paths must be absolute")
        if session["pipe_directory"] == session["log_directory"]:
            raise ValueError("MPS pipe and log directories must differ")
    elif "mps_session" in document:
        raise ValueError("MPS session is only supported for mps_cap")
    used = set()
    if mode in {"mps_static_sm", "green_context"}:
        _integer(document.get("sm_capacity"), "sm_capacity")
    elif "sm_capacity" in document:
        raise ValueError("SM capacity is only meaningful for explicit SM sets")
    if mode == "request_admission":
        groups = document.get("groups")
        if not isinstance(groups, dict) or not groups:
            raise ValueError("Request admission requires nonempty named groups")
        covered = set()
        for name, group in groups.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Admission group names must be nonempty strings")
            _keys(group, {"aliases", "max_active"})
            members = group["aliases"]
            if (not isinstance(members, list) or not members
                    or any(not isinstance(a, str) for a in members)
                    or len(set(members)) != len(members) or set(members) - set(aliases)):
                raise ValueError("Admission groups require distinct resident aliases")
            _integer(group["max_active"], "group max_active")
            covered.update(members)
        if covered != set(aliases):
            raise ValueError("Admission groups must cover all residents, including backgrounds")
    elif "groups" in document:
        raise ValueError("Groups are only meaningful for request admission")
    for alias, allocation in allocations.items():
        if mode == "shared":
            _keys(allocation, set())
        elif mode == "mps_cap":
            _keys(allocation, {"active_thread_percentage"})
            percentage = allocation["active_thread_percentage"]
            if type(percentage) is not int or not 0 < percentage <= 100:
                raise ValueError(f"Invalid MPS percentage for {alias}")
            # These are per-client ceilings, NOT additive exclusive SM shares.
        elif mode == "request_admission":
            _keys(allocation, {"max_inflight"})
            _integer(allocation["max_inflight"], "max_inflight")
        else:
            _keys(allocation, {"sm_ids"})
            ids = allocation["sm_ids"]
            if not isinstance(ids, list) or not ids:
                raise ValueError("Explicit nonempty SM sets required")
            for sm in ids:
                _integer(sm, "SM id", 0)
                if sm >= document["sm_capacity"] or sm in used:
                    raise ValueError("SM sets must be disjoint and within declared capacity")
                used.add(sm)
    normalized = copy.deepcopy(document)
    if mode in {"mps_static_sm", "green_context"}:
        for allocation in normalized["allocations"].values():
            allocation["sm_ids"].sort()
    if mode == "request_admission":
        for group in normalized["groups"].values():
            group["aliases"].sort()
    return normalized


def prepare_registry(registry, resources):
    """Bind requested controls to already-expanded workers without launching.

    SM identities are design declarations until a platform adapter proves the
    supported granularity and realized mapping. No cap-to-speed extrapolation.
    """
    resources = validate_resources(resources, registry)
    result = copy.deepcopy(registry)
    mode = resources["mode"]
    blockers = [] if mode in {"shared", "request_admission"} else ["PLATFORM_RESOURCE_CONTROL_NOT_VERIFIED"]
    if mode in {"mps_static_sm", "green_context"}:
        blockers.append("RESOURCE_CONTEXT_ADAPTER_NOT_IMPLEMENTED")
    for alias, entry in result.items():
        env = entry.setdefault("env", {})
        if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
            raise ValueError("Worker environment must contain string keys and values")
        if _mps_keys(env):
            raise ValueError("Configure MPS only through the resource contract")
        if "resource_binding" in entry:
            raise ValueError("Worker registry already has a resource binding")
        if env.get("CUDA_VISIBLE_DEVICES", resources["device"]) != resources["device"]:
            raise ValueError("Worker CUDA device conflicts with resource contract")
        env["CUDA_VISIBLE_DEVICES"] = resources["device"]
        if mode == "mps_cap":
            env.update(CUDA_MPS_PIPE_DIRECTORY=resources["mps_session"]["pipe_directory"],
                       CUDA_MPS_LOG_DIRECTORY=resources["mps_session"]["log_directory"],
                       CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=format(
                           resources["allocations"][alias]["active_thread_percentage"], ".12g"))
        entry["resource_binding"] = {
            "schema": "gsdeploy.worker_resource_binding.v1", "alias": alias,
            "configuration_sha256": identity(resources), "mode": mode,
            "requested": copy.deepcopy(resources["allocations"][alias]),
            "expected_environment": copy.deepcopy({k: v for k, v in env.items()
                                                     if k in _mps_keys(env) or k == "CUDA_VISIBLE_DEVICES"}),
            "launch_blockers": list(blockers), "effective_isolation_verified": False,
            **({"admission_configuration": resources} if mode == "request_admission" else {}),
        }
    return result


def worker_environment(config, parent):
    """Resolve child environment; explicit studies cannot inherit hidden caps."""
    binding = config.get("resource_binding")
    env = dict(parent)
    if binding is not None:
        if binding.get("schema") != "gsdeploy.worker_resource_binding.v1":
            raise ValueError("Unknown worker resource binding")
        if binding.get("launch_blockers"):
            raise ValueError("Resource launch blocked: " + "; ".join(binding["launch_blockers"]))
        if binding.get("mode") not in {"shared", "request_admission"}:
            raise ValueError("Controlled resource launch requires a verified platform adapter")
        for key in _mps_keys(env):
            env.pop(key, None)
        expected = binding["expected_environment"]
        actual = {k: v for k, v in config.get("env", {}).items()
                  if k in _mps_keys(config.get("env", {})) or k == "CUDA_VISIBLE_DEVICES"}
        if actual != expected or _mps_keys(actual):
            raise ValueError("Worker resource environment changed after preparation")
    env.update(config.get("env", {}))
    env["PYTHONUNBUFFERED"] = "1"
    return env
