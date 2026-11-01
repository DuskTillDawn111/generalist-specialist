"""Explicit recovery producers for the shared complete-cohort resource scorer."""
import argparse

from . import twoFIFO_recovery_generated_v1 as generated
from . import twoFIFO_recovery_resource_v1 as resource
from . import twoFIFO_source_rebound_resource_rescore_v1 as scorer


def run_all(completed_report_path, resource_model_path, output, **kwargs):
    return scorer.run_all(completed_report_path, resource_model_path, output,
                          generated_api=generated, resource_api=resource, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-report', required=True)
    parser.add_argument('--resource-model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scenario', choices=['mean', 'max'], default='mean')
    parser.add_argument('--skip-measured-labels', action='store_true')
    args = parser.parse_args()
    print(run_all(args.completed_report, args.resource_model, args.output,
                  scenario=args.scenario, include_measured_labels=not args.skip_measured_labels))
