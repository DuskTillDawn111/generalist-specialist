"""Fit a separate unvalidated causal idle-energy law to verified training receipts."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.fit_activation_native import load_bundle
from gsdeploy.forecasting.native_training import digest, stage_layout
from gsdeploy.forecasting.native_energy import fit_idle_age_energy, NativeIdleAgeEnergy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    fit = json.loads(args.fit.read_text())
    profiles = {p['id']: p for p in fit['profiles']['profiles']}
    runs = []
    for doc, receipt in zip(fit['normalized_runs'], fit['source_receipts']):
        bundle = load_bundle(Path(receipt.rsplit(':sha256:', 1)[0]).parent)
        if bundle['source'] != receipt or doc['run_id'] != bundle['raw']['run_id']:
            raise ValueError('Original training energy receipts changed')
        raw = bundle['raw']
        prefix_ends = [s['end_ns'] for r in raw['priming_request_records']
                       for s in stage_layout(r) if s['kind'] == 'gpu']
        if not prefix_ends or max(prefix_ends) > raw['observation_start_ns']:
            raise ValueError('Observed complete priming prefix required for causal idle age')
        runs.append(dict(run_id=doc['run_id'], split=doc['split'],
                         window_ns=doc['observation_window_ns'],
                         initial_last_native_end_ns=max(prefix_ends),
                         spans=[dict(alias=r['alias'], profile_id=r['profile_id'], **s)
                                for r in doc['requests'] for s in r['segments']],
                         telemetry_errors=raw['telemetry_errors'], board_samples=raw['board_samples']))
    spec, diagnostic = fit_idle_age_energy(profiles, runs)
    result = copy.deepcopy(fit)
    result.update(parent_energy_fit_sha256=digest(fit), idle_energy_diagnostic=diagnostic,
                  idle_energy_independent_accuracy_qualified=False)
    if spec is not None:
        spec['execution_context_sha256'] = fit['profiles']['execution_context_sha256']
        result['profiles']['native_idle_age_energy'] = spec
        law = NativeIdleAgeEnergy(spec)
        from gsdeploy.forecasting.native_energy import activity_windows
        comparisons = []
        from gsdeploy.energy import integrate_power
        for run in runs:
            windows = activity_windows(profiles, run['spans'], *run['window_ns'])
            predicted, _ = law.integrate(windows, run['initial_last_native_end_ns'])
            actual = integrate_power(run['board_samples'], *run['window_ns'], max_gap_seconds=.25)
            comparisons.append(dict(run_id=run['run_id'], observed_joules=actual, predicted_joules=predicted,
                                    relative_error=abs(predicted-actual)/actual))
        result['idle_energy_training_comparisons'] = comparisons
    with args.output.open('x') as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False); out.write('\n')
    print(json.dumps(dict(output=str(args.output), diagnostic={k:v for k,v in diagnostic.items() if k != 'blocks_detail'},
                          training_comparisons=result.get('idle_energy_training_comparisons'),
                          admission_enabled=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
