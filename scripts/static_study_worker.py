"""Unified retained worker; --check is CPU-only, --execute explicitly loads GPU."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def load_module(name, path):
    definition = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--source-root", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--pool")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    pp = spec.get("adapter") == "retained_ppocrv6"
    tasks = ["ocr"] if pp else (["text", "image", "ocr", "asr"] if spec["candidate"]["slot"].startswith("generalist")
                                else [spec["candidate"]["slot"]])
    if args.check:
        count = 0
        adapter = None
        if pp:
            source = Path(spec["adapter_source"])
            if hashlib.sha256(source.read_bytes()).hexdigest() != spec["adapter_source_sha256"]:
                raise ValueError("Changed retained PP adapter source")
        else:
            from gsdeploy.backends.compact import CompactAdapter
            adapter = CompactAdapter(json.loads(Path(spec["resources"]["protocol"]).read_text()),
                                     json.loads(Path(spec["resources"]["classmap"]).read_text()))
        if args.pool:
            for line in Path(args.pool).read_text().splitlines():
                item = json.loads(line)
                if item["task_type"] not in tasks:
                    continue
                if adapter:
                    candidate = spec.get("task_candidates", {}).get(item["task_type"], spec["candidate"])
                    adapter.build(candidate, item["sample"])
                count += 1
        print(json.dumps(dict(status="CPU_READY_GPU_UNVERIFIED", tasks=tasks, inputs_checked=count, gpu_started=False)))
        return
    from gsdeploy.backends.compact_memory import allocator_environment
    if any(os.environ.get(k) != v for k, v in allocator_environment(spec["execution"]).items()):
        raise ValueError("Frozen allocator environment was not applied before worker startup")
    resources = dict(spec["resources"])
    resources["request_output"] = str(Path(resources["request_output"]) / str(os.getpid()))
    if pp:
        source = Path(spec["adapter_source"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != spec["adapter_source_sha256"]:
            raise ValueError("Changed retained PP adapter source")
        adapter = load_module("static_retained_pp_adapter", source).Adapter(spec, resources)
    else:
        from gsdeploy.backends.compact_native import ProfileAdapter
        adapter = ProfileAdapter(spec, resources)
    # Keep current runtime wire protocol while native inference comes from frozen source.
    protocol = load_module("static_runtime_protocol", Path(__file__).resolve().parents[1] / "gsdeploy/runtime/protocol.py")

    def handle(request):
        if request["task_type"] not in tasks or request["sample"]["task"] != request["task_type"]:
            raise ValueError("Unsupported or mismatched worker task")
        if not pp:
            adapter.executor.candidate = spec.get("task_candidates", {}).get(request["task_type"], spec["candidate"])
        return adapter.run(request["sample"])

    protocol.run_worker(spec["candidate"]["slot"], handle,
                        dict(model=spec["candidate"]["model"], tasks=tasks, spec_sha256=hashlib.sha256(Path(args.spec).read_bytes()).hexdigest()))


if __name__ == "__main__":
    main()
