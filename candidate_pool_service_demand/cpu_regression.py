"""Existing CPU regressions plus demand gates, without repeating 7B campaigns."""
import json
from pathlib import Path
import unittest

from candidate_pool_audit.provenance import sha
from .prepare import ROOT, OUT, read, write, reference

EXCLUDED = {
    'deployment_search.tests.test_integer_replicas.ValidationCampaignTests.test_A_B_C_D',
    'deployment_search.tests.test_integer_replicas.ValidationCampaignTests.test_12_20_30_no_enumeration_and_fixed_runtime_budget',
    'deployment_core.tests.test_milp_seeder.OracleAndScalabilityTests.test_20_and_30_models_without_deployment_enumeration',
    'deployment_search.tests.test_search.SearchTests.test_pools_scale_without_enumeration_and_improve',
}


def flatten(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from flatten(test)
        else:
            yield test


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for directory in ('deployment_core/tests', 'deployment_search/tests'):
        # core/tests is a namespace directory without __init__.py. Import test
        # modules by name instead of changing the protected historical package.
        for path in sorted((ROOT / directory).glob('test_*.py')):
            name = '.'.join(path.relative_to(ROOT).with_suffix('').parts)
            suite.addTests(loader.loadTestsFromName(name))
    suite.addTests(loader.loadTestsFromName('candidate_pool_service_demand.test_service_demand'))
    tests = list(flatten(suite))
    skipped = [test.id() for test in tests if test.id() in EXCLUDED]
    if set(skipped) != EXCLUDED:
        raise RuntimeError('Expected exact four campaign/scalability wrappers to be excluded')
    # Prior successful 7B campaigns remain valid only if ALL their code is unchanged.
    protected = read(OUT / 'before_hashes.json')['files']
    relevant = {name: ref for name, ref in protected.items() if name.startswith(('deployment_core/', 'deployment_search/')) and name.endswith('.py')}
    changed = [name for name, ref in relevant.items() if sha(ROOT / name) != ref['sha256']]
    if changed:
        raise RuntimeError('7B code changed; previous campaign evidence is no longer applicable')
    priors = [ROOT / 'results/integer_replica_search_foundation' / name
              for name in ('core_regression.log', 'search_regression.log')]
    if not all(prior.read_text().rstrip().endswith('OK') for prior in priors):
        raise RuntimeError('Prior existing regression did not pass')
    with (OUT / 'cpu_tests.log').open('w') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.TestSuite(t for t in tests if t.id() not in EXCLUDED))
    value = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors),
        'prior_campaign_test_methods_not_rerun': skipped,
        'retained_prior_evidence': [reference(prior) for prior in priors], 'unchanged_7B_source_files_checked': len(relevant),
        'reason_not_rerun': 'User explicitly prohibited rerunning A–E and 20/30 synthetic scalability campaigns; these four unit-test wrappers invoke campaign/scalability runs.',
        'log': reference(OUT / 'cpu_tests.log')}
    write(OUT / 'cpu_regression_validation.json', value)
    print(json.dumps(value), flush=True)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
