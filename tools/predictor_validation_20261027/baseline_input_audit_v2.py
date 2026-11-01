"""Read-only CPU adaptation of existing raw kernel profiles or Nsight exports.

Writes a new exclusive result directory. Does not invoke a profiler, GPU API,
model backend, or fit a model from unsupported/missing evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gsdeploy.forecasting import kernel_baseline
from gsdeploy.forecasting.kernel_baseline import (
    MANIFEST_SCHEMA, adapt_nsys_trace, adapt_raw_kernel_profiles,
)


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def audit(path, output_dir, *, manifest_path=None, raw=False):
    path, output_dir = Path(path).resolve(), Path(output_dir).resolve()
    data = json.loads(path.read_text())
    manifest = json.loads(Path(manifest_path).read_text()) if manifest_path else dict(
        schema=MANIFEST_SCHEMA, run_id='read_only_unbound_trace_audit',
        timing_basis='profiled_isolated')
    sources = [receipt(path)]
    implementation_sources = [receipt(__file__), receipt(kernel_baseline.__file__)]
    if manifest_path:
        sources.append(receipt(manifest_path))
    if raw:
        if manifest.get('measurement_source') != sources[0]:
            raise ValueError('Raw measurement source must exactly bind the input JSON receipt')
        adapted = adapt_raw_kernel_profiles(data, manifest)
    else:
        # Verify source bytes, not only equality of self-reported JSON receipts.
        for source in data.get('sources', []):
            if receipt(source['path']) != source:
                raise ValueError('Existing trace source receipt no longer matches bytes')
            sources.append(source)
        if not manifest_path and data.get('sources'):
            manifest['measurement_source'] = data['sources'][0]
        adapted = adapt_nsys_trace(data, manifest)
    required = ('split', 'evidence_kind', 'hardware_id', 'execution_context_sha256',
               'isolation_evidence', 'profiler_id', 'profiler_perturbed',
               'perturbation_flags', 'configurations') + (() if raw else ('kernel_bindings',))
    missing = [name for name in required if name not in manifest or
               (manifest[name] is None and name != 'profiler_id')]
    report = dict(schema='gsdeploy.kernel_baseline_input_audit.v2', sources=sources,
                  implementation_sources=implementation_sources,
                  adapter_result=adapted, missing_capture_fields=missing,
                  trained_model_created=False, gpu_work_performed=False,
                  natural_timing_validated=False, deployment_qualified=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir/'audit.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (output_dir/'output-manifest.json').write_text(json.dumps(
        dict(outputs=[receipt(output)]), indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--trace', type=Path)
    group.add_argument('--raw', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.raw or args.trace, args.output_dir,
                   manifest_path=args.manifest, raw=args.raw is not None)
    adapted = report['adapter_result']
    print(json.dumps(dict(accepted=len(adapted['rows']), rejected=len(adapted['rejected_rows']),
                          trained_model_created=False)))


if __name__ == '__main__':
    main()
