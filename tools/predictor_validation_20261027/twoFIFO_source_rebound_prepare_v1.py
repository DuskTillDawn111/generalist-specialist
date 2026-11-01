"""CPU-only preparation for a source-rebound TwoFIFO training campaign.

The previous campaign is immutable evidence.  This module reads its twelve
plans, copies their frozen source tree to a new output, replaces the two
drifted gsdeploy sources with the current working-tree bytes, and rebinds the
context/plan identities.  It deliberately has no execute or GPU entry point.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import uuid
from pathlib import Path

from .stage_response_features_v1 import receipt

ROOT = Path(__file__).resolve().parents[2]
PAIR = (ROOT / "gsdeploy/milp.py", ROOT / "gsdeploy/optimizer.py")
OLD_SCHEMA = "gsdeploy.original_twoFIFO_owned_training_campaign.v3"
SCHEMA = "gsdeploy.twoFIFO_source_rebound_campaign.v1"
PLAN_SCHEMA = OLD_SCHEMA + ".attempt"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _pair_receipts():
    return [receipt(path) for path in PAIR]


def _old_campaign(path):
    path = Path(path).resolve()
    campaign = read(path)
    accepted = ((campaign.get("schema") == OLD_SCHEMA and campaign.get("status") == "CPU_PREPARED_NOT_COLLECTED")
        or (campaign.get("schema") == SCHEMA and campaign.get("status") == "CPU_PREPARED_SOURCE_REBOUND_NOT_COLLECTED"))
    require(accepted and campaign.get("no_retries") is True and campaign.get("GPU_started") is False,
            "Immutable CPU-prepared source campaign required")
    refs = campaign.get("attempts")
    require(type(refs) is list and len(refs) == 12, "Exactly twelve original attempts required")
    plans = []
    for ref in refs:
        require(receipt(ref["path"]) == ref, "Original campaign plan receipt changed")
        plan = read(ref["path"])
        require(plan.get("schema") == PLAN_SCHEMA and plan.get("split") == "training"
                and plan.get("no_retries") is True and plan.get("GPU_started", False) is False,
                "Original training attempt is not a CPU-prepared plan")
        require(plan.get("context", {}).get("path"), "Original plan lacks context binding")
        require(receipt(plan["context"]["path"]) == plan["context"],
                "Original context receipt changed")
        plans.append((ref, plan))
    return path, campaign, plans


def _replace_context(old_context, old_context_path, source_root, pair_frozen, rebound_id):
    context = copy.deepcopy(old_context)
    old_current = {str(path): ref for path, ref in zip(PAIR, pair_frozen)}
    replaced = {str(path): False for path in PAIR}
    for item in context.get("execution_sources", []):
        original = item.get("original_path")
        if original in old_current:
            item["frozen"] = copy.deepcopy(old_current[original])
            replaced[original] = True
    require(all(replaced.values()), "Context has no exact current source bindings to rebound")
    context["sources"] = [item["frozen"] for item in context["execution_sources"]]
    context["source_rebound"] = {
        "schema": SCHEMA + ".context_binding",
        "old_context": receipt(old_context_path),
        "current_sources": _pair_receipts(),
        "frozen_sources": pair_frozen,
        "validation_labels_read": False,
        "GPU_started": False,
        "source_rebound_id": rebound_id,
    }
    context["source_rebound_context_sha256"] = digest(
        {key: value for key, value in context.items() if key != "source_rebound_context_sha256"})
    return context


def _rebound_sources(old_plan, pair_refs, source_root):
    """Update only the two known drifted direct source receipts."""
    sources = copy.deepcopy(old_plan["sources"])
    by_path = {item["path"]: item for item in pair_refs}
    seen = set()
    for item in sources:
        if item.get("path") in by_path:
            item.update(by_path[item["path"]])
            seen.add(item["path"])
    require(seen == set(by_path), "Original plan lacks one of the drifted source receipts")
    return sources


def prepare(campaign_path, output, *, source_rebound_id):
    require(isinstance(source_rebound_id, str) and len(source_rebound_id) == 32
        and all(c in '0123456789abcdef' for c in source_rebound_id),
        'Explicit random 128-bit lowercase hexadecimal source rebound nonce required')
    old_path, old_campaign, old_plans = _old_campaign(campaign_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    old_source_root = old_path.parent / "source-freeze"
    require(old_source_root.is_dir(), "Original source-freeze tree required")
    source_root = output / "source-freeze"
    shutil.copytree(old_source_root, source_root)

    # Freeze the exact current files into the new tree.  No old artifact is
    # edited, and every plan receives the same pair receipts.
    pair_frozen = []
    for source in PAIR:
        target = source_root / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        pair_frozen.append(receipt(target))
    pair_refs = _pair_receipts()

    old_context_path = Path(old_plans[0][1]["context"]["path"]).resolve()
    old_context = read(old_context_path)
    rebound_id = source_rebound_id
    context = _replace_context(old_context, old_context_path, source_root, pair_frozen, rebound_id)
    context_path = output / "context.json"
    save(context_path, context)
    context_ref = receipt(context_path)
    attempts = []
    for old_ref, old_plan in old_plans:
        plan = copy.deepcopy(old_plan)
        plan["execution_id"] = uuid.uuid5(uuid.NAMESPACE_URL,
            rebound_id + ":" + plan["attempt_id"]).hex
        plan["context"] = context_ref
        plan["source_freeze"] = [
            copy.deepcopy(item) for item in plan["source_freeze"]
        ]
        for item in plan["source_freeze"]:
            original = Path(item["original_path"])
            target = source_root / original.relative_to(ROOT)
            require(target.is_file(), "Copied source-freeze file missing")
            item["frozen"] = receipt(target)
        require(all(any(item.get("original_path") == str(source) for item in plan["source_freeze"])
                    for source in PAIR), "Plan source-freeze pair missing")
        plan["sources"] = _rebound_sources(plan, pair_refs, source_root)
        plan["GPU_started"] = False
        plan["source_rebound"] = {
            "schema": SCHEMA + ".plan_binding",
            "old_plan": old_ref,
            "old_campaign": receipt(old_path),
            "current_sources": pair_refs,
            "frozen_sources": pair_frozen,
            "context": context_ref,
            "validation_labels_read": False,
            "GPU_started": False,
            "source_rebound_id": rebound_id,
        }
        plan["execution_context_sha256"] = digest(
            {key: value for key, value in plan.items() if key != "execution_context_sha256"})
        destination = output / plan["attempt_id"]
        destination.mkdir()
        plan_path = destination / "plan.json"
        save(plan_path, plan)
        attempts.append(receipt(plan_path))

    campaign = {
        "schema": SCHEMA,
        "status": "CPU_PREPARED_SOURCE_REBOUND_NOT_COLLECTED",
        "old_campaign": receipt(old_path),
        "context": context_ref,
        "current_sources": pair_refs,
        "frozen_sources": pair_frozen,
        "attempts": attempts,
        "attempt_count": 12,
        "no_retries": True,
        "GPU_started": False,
        "numerical_fit_run": False,
        "validation_labels_read": False,
        "execute_entrypoint": None,
        "source_rebound_id": rebound_id,
        "preparation_source": receipt(__file__),
    }
    save(output / "campaign.json", campaign)
    audit(output / "campaign.json")
    return campaign


def audit(path):
    campaign_path = Path(path).resolve()
    campaign = read(campaign_path)
    require(campaign.get("schema") == SCHEMA
            and campaign.get("status") == "CPU_PREPARED_SOURCE_REBOUND_NOT_COLLECTED"
            and campaign.get("attempt_count") == 12
            and campaign.get("no_retries") is True
            and campaign.get("GPU_started") is False
            and campaign.get("numerical_fit_run") is False
            and campaign.get("validation_labels_read") is False
            and campaign.get("execute_entrypoint") is None,
            "Exact CPU-only source-rebound campaign required")
    old_ref = campaign["old_campaign"]
    require(receipt(old_ref["path"]) == old_ref, "Original campaign changed")
    _, old_campaign, old_plans = _old_campaign(old_ref["path"])
    context_ref = campaign["context"]
    require(receipt(context_ref["path"]) == context_ref, "Rebound context changed")
    context = read(context_ref["path"])
    require(context.get("source_rebound", {}).get("source_rebound_id") == campaign.get("source_rebound_id"),
        "Context rebound nonce mismatch")
    pair_refs = campaign["current_sources"]
    pair_frozen = campaign["frozen_sources"]
    require(pair_refs == _pair_receipts(), "Current gsdeploy source SHA changed")
    for current, frozen in zip(PAIR, pair_frozen):
        require(receipt(frozen["path"]) == frozen
                and frozen["sha256"] == hashlib.sha256(current.read_bytes()).hexdigest(),
                "Rebound frozen source does not match current source")
    require(context.get("source_rebound", {}).get("current_sources") == pair_refs,
            "Context current-source binding missing")
    require(context.get("source_rebound", {}).get("frozen_sources") == pair_frozen,
            "Context frozen-source binding missing")
    require(context.get("source_rebound", {}).get("old_context") == old_plans[0][1]["context"],
            "Context old-context binding changed")
    require(context.get("source_rebound_context_sha256") == digest(
        {key: value for key, value in context.items() if key != "source_rebound_context_sha256"}),
        "Rebound context identity drift")
    refs = campaign.get("attempts")
    require(type(refs) is list and len(refs) == 12, "Exact twelve rebound plans required")
    ids = []
    for index, ref in enumerate(refs):
        require(receipt(ref["path"]) == ref, "Rebound plan receipt changed")
        plan = read(ref["path"])
        old_ref_for_plan, old_plan = old_plans[index]
        require(plan.get("schema") == PLAN_SCHEMA and plan.get("context") == context_ref
                and plan.get("source_rebound", {}).get("current_sources") == pair_refs
                and plan.get("source_rebound", {}).get("frozen_sources") == pair_frozen
                and plan.get("source_rebound", {}).get("old_plan") == old_ref_for_plan
                and plan.get("source_rebound", {}).get("source_rebound_id") == campaign.get("source_rebound_id")
                and plan.get("GPU_started") is False and plan.get("no_retries") is True,
                "Plan context/source rebound binding mismatch")
        unchanged = {key: value for key, value in plan.items()
                     if key not in {"execution_id", "context", "source_freeze", "sources",
                                    "source_rebound", "execution_context_sha256", "GPU_started"}}
        old_unchanged = {key: value for key, value in old_plan.items()
                         if key not in {"execution_id", "context", "source_freeze", "sources",
                                        "execution_context_sha256", "GPU_started", "source_rebound"}}
        require(unchanged == old_unchanged, "Rebound plan changed immutable workload semantics")
        require(plan.get("execution_context_sha256") == digest(
            {key: value for key, value in plan.items() if key != "execution_context_sha256"}),
                "Plan execution context identity drift")
        ids.append(plan["attempt_id"])
        for current in pair_refs:
            require(any(item.get("path") == current["path"] and item.get("sha256") == current["sha256"]
                        for item in plan.get("sources", [])),
                    "Plan source closure lacks current source receipt")
    require(len(set(ids)) == 12, "Rebound attempt IDs must be unique")
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="op", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--campaign", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--source-rebound-id", required=True)
    check = sub.add_parser("audit")
    check.add_argument("--campaign", required=True)
    args = parser.parse_args()
    result = prepare(args.campaign, args.output, source_rebound_id=args.source_rebound_id) if args.op == "prepare" else audit(args.campaign)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
