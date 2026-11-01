"""Explicit native worker entry; importing or --help never loads a model."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('GPU execution is disabled; explicit --execute is required')
    from .compact_native import ProfileAdapter
    from ..runtime.protocol import run_worker
    spec = json.loads(Path(args.spec).read_text())
    from .compact_memory import allocator_environment
    if any(os.environ.get(k) != v for k, v in allocator_environment(spec['execution']).items()):
        raise ValueError('Frozen allocator environment was not applied before the worker started')
    adapter = ProfileAdapter(spec, spec['resources'])
    tasks = adapter.executor.adapter.supported_tasks(spec['candidate'])
    run_worker(spec['candidate']['slot'], lambda request: adapter.run(request['sample']),
               {'model': spec['candidate']['model'], 'tasks': list(tasks),
                'revision': spec['candidate']['revision'], 'protocol_sha256': adapter.executor.adapter.policy_sha256})


if __name__ == '__main__':
    main()
