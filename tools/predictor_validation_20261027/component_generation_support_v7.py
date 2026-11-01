"""Rederive exact generation bindings for original-context DAG components.

This module fits no model. Each component retains its original catalog path,
review schema and DAG receipt. Positive unsupported work mass remains UNKNOWN.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from .generation_dag_coverage_v6 import select_scenarios
from .generation_work_v4 import actual_work, input_content_identity
from .generation_work_v5 import ExactContentGenerationWorkProvider
from .kernel_domain_bridge_review_v3 import load_report as load_v3
from .small_training_domain_bridge_review_v4 import load_report as load_v4
from .multi_native_composite_v5 import MultiNativeProvider, checked
from .native_collect import receipt, save
from .native_execution_projection_v5 import native_execution_projection
from .startup_prediction_v6 import public_catalog

SCHEMA = 'gsdeploy.component_exact_content_generation_dag_support.v7'
REVIEW_LOADERS = {
    'gsdeploy.training_collection_strict_review.v3': load_v3,
    'gsdeploy.startup_training_collection_strict_review.v4': load_v4,
}
PUBLIC_FEATURES = ('alias', 'pool_id', 'family', 'task', 'input_tokens',
                   'protocol_sha256', 'effective_generation', 'input_content_sha256')


def load_light_review(ref):
    report = checked(ref)
    loader = REVIEW_LOADERS.get(report.get('schema'))
    if loader is None:
        raise ValueError('Exact v3 or startup-aware v4 strict training review required')
    report, plan, raw = loader(ref['path'], 'light-kernel')
    if (report.get('training_only') is not True or report.get('issues')
            or raw.get('kind') != 'measured' or raw.get('status') != 'completed'
            or raw.get('not_for_training') or any(j.get('split') != 'training' for j in plan['jobs'])):
        raise ValueError('Completed exclusively training light evidence required')
    return report, plan, raw


def derive_components(adapter, light_reviews):
    """Recompute public input features/work from each exact original light raw."""
    if set(light_reviews) != set(adapter.components):
        raise ValueError('One explicit original light review per component required')
    components, keys = [], set()
    for cid, component in adapter.components.items():
        entry = component['entry']; spec = entry['spec']; dag = checked(spec['dag'])
        report, plan, raw = load_light_review(light_reviews[cid])
        if report['raw'] not in dag.get('sources', []):
            raise ValueError('Light descriptor raw must be an original DAG source')
        light_context = checked(spec['light_context']); natural_context = checked(spec['natural_context'])
        light = native_execution_projection(light_context)
        natural = native_execution_projection(natural_context)
        if (report['context'] != spec['light_context']
                or raw['execution_context_sha256'] != entry['original_light_context_sha256']
                or plan['execution_context_sha256'] != entry['original_light_context_sha256']
                or dag['execution_context_sha256'] != entry['original_light_context_sha256']
                or light['original_full_context_sha256'] != entry['original_light_context_sha256']
                or natural['original_full_context_sha256'] != entry['original_natural_context_sha256']):
            raise ValueError('Generation descriptors must retain original component execution identities')
        light_pools, light_catalog = public_catalog(light_context)
        pools, catalog = public_catalog(natural_context)
        # Equal bytes at another path do not replace the component's receipt.
        if light_catalog['frozen'] != catalog['frozen'] or light_pools != pools:
            raise ValueError('Component light/natural exact public catalog receipts differ')
        rows = {}; planned = Counter()
        for job in plan['jobs']:
            planned.update((job['id'], alias, job['tasks'][alias], pool)
                           for alias, pool in job['inputs'].items())
        observed = Counter((r['profile_job_id'], r['model'], r['task'], r['pool_id'])
                           for r in raw['request_records'])
        if observed != planned:
            raise ValueError('Exact nonduplicated original training cohort required')
        for row in raw['request_records']:
            key = row['model'], row['task'], row['pool_id'], row['request_id']
            if key in rows or row.get('split') != 'training' or row.get('status') != 'completed':
                raise ValueError('Unique completed training descriptor source required')
            rows[key] = row
        descriptors = []
        for profile in dag['profiles']:
            key = profile['alias'], profile['task'], profile['pool_id']
            if key in keys or adapter.routes.get(key) != cid:
                raise ValueError('Unique exact component route required')
            keys.add(key)
            row = rows[(*key, profile['request_id'])]
            if profile['source_sqlite'] != report['owners'][profile['alias']]['sqlite']:
                raise ValueError('Generation template must bind original strict owner SQLite')
            pool = pools[profile['pool_id']]
            if pool['task_type'] != profile['task']:
                raise ValueError('Public content task differs from exact DAG route')
            response = row['worker_response']; ocr = profile['task'] == 'ocr'
            work = dict(kind='NON_AUTOREGRESSIVE_OCR_SEPARATE') if ocr else actual_work(response, row['request_id'])
            descriptors.append(dict(component_id=cid, alias=profile['alias'], family=profile['family'],
                task=profile['task'], pool_id=profile['pool_id'], input_content_sha256=input_content_identity(pool),
                input_tokens=response.get('input_tokens'), protocol_sha256=response.get('protocol_sha256'),
                effective_generation=copy.deepcopy(response.get('effective_generation')), work=work,
                native_execution_projection_sha256=natural['projection_sha256'],
                original_light_context_sha256=entry['original_light_context_sha256'],
                original_natural_context_sha256=entry['original_natural_context_sha256'],
                source_sqlite=profile['source_sqlite'],
                template=dict(dag=spec['dag'], alias=profile['alias'], task=profile['task'], pool_id=profile['pool_id'])))
        components.append(dict(component_id=cid, dag=spec['dag'], light_review=light_reviews[cid],
            light_raw=report['raw'], light_context=spec['light_context'], natural_context=spec['natural_context'],
            public_input_catalog=catalog['frozen'], descriptors=descriptors))
    if keys != set(adapter.plans) or Counter(k[1] for k in keys) != dict(text=6, image=2, asr=2, ocr=2):
        raise ValueError('Exactly the original twelve four-task routes required')
    return components


def implementation_sources():
    return [receipt(Path(__file__).with_name(name)) for name in (
        'component_generation_support_v7.py', 'generation_dag_coverage_v5.py',
        'generation_dag_coverage_v6.py', 'generation_work_v4.py', 'generation_work_v5.py',
        'kernel_domain_bridge_review_v3.py', 'small_training_domain_bridge_review_v4.py',
        'startup_prediction_v6.py', 'native_execution_projection_v5.py')]


def build(composite_ref, generation_ref, light_reviews, aliases):
    adapter = MultiNativeProvider(checked(composite_ref), aliases, host_scenario='host_issue_depth_supply')
    document = dict(schema=SCHEMA, composite=composite_ref, generation_model=generation_ref,
        light_reviews=copy.deepcopy(light_reviews), components=derive_components(adapter, light_reviews),
        implementation_sources=implementation_sources(), training_only=True,
        future_actual_output_used=False, unseen_input_generalization_claimed=False,
        physical_accuracy_validated=False, deployment_qualified=False, GPU_started=False)
    # Execute every generation lookup before publishing a support document.
    provider = ComponentGenerationSupport(document, adapter, allow_exact_content_generation_hypothesis=True)
    rows = [dict(request_id='support:' + str(i), alias=a, task=t, pool_id=p)
            for i, (a, t, p) in enumerate(adapter.plans)]
    provider.check(rows)
    return document


class ComponentGenerationSupport:
    def __init__(self, document, adapter, *, allow_exact_content_generation_hypothesis=False):
        if not allow_exact_content_generation_hypothesis:
            raise ValueError('Explicit exact-content generation conditioning opt-in required')
        if (document.get('schema') != SCHEMA or document.get('training_only') is not True
                or document.get('future_actual_output_used') is not False
                or document.get('implementation_sources') != implementation_sources()):
            raise ValueError('Source-bound component generation support required')
        contract = checked(document['composite'])
        if (contract['composite_execution_sha256'] != adapter.execution_context_sha256
                or contract['components'] != [c['entry'] for c in adapter.components.values()]):
            raise ValueError('Generation support must bind this exact original composite')
        expected = derive_components(adapter, document['light_reviews'])
        if document.get('components') != expected:
            raise ValueError('Generation descriptors differ from rederived original training evidence')
        self.adapter = adapter
        self.public_bindings = {}
        self.providers = {}
        self.templates = {}
        model = checked(document['generation_model'])
        for component in expected:
            cid = component['component_id']; templates = component['descriptors']
            projections = {t['native_execution_projection_sha256'] for t in templates}
            if len(projections) != 1:
                raise ValueError('One original natural projection per component required')
            projection = projections.pop()
            if projection not in self.providers:
                self.providers[projection] = ExactContentGenerationWorkProvider(
                    model, projection, allow_exact_content_hypothesis=True)
            self.templates[cid] = templates
            for binding in templates:
                key = binding['alias'], binding['task'], binding['pool_id']
                self.public_bindings[key] = copy.deepcopy(binding)

    def check(self, rows):
        results = []
        for row in rows:
            key = row['alias'], row['task'], row['pool_id']
            if key not in self.public_bindings:
                raise UnknownEvidence('NO_EXACT_COMPONENT_GENERATION_PUBLIC_BINDING')
            binding = self.public_bindings[key]; expected = binding['template']
            plan = self.adapter.plans[key]
            if (self.adapter.routes[key] != binding['component_id']
                    or plan['source_sqlite'] != binding['source_sqlite']
                    or any(plan[k] != expected[k] for k in ('alias', 'task', 'pool_id'))
                    or self.adapter.component(row)['entry']['spec']['dag'] != expected['dag']):
                raise ValueError('Selected generation DAG/owner/task/input differs from executed original plan')
            if row['task'] == 'ocr':
                results.append(dict(request_id=row['request_id'], kind='NON_AUTOREGRESSIVE_OCR_SEPARATE',
                    input_content_sha256=binding['input_content_sha256'], selected_template=expected,
                    future_actual_output_used=False))
                continue
            public = {k: copy.deepcopy(binding[k]) for k in PUBLIC_FEATURES}
            projection = binding['native_execution_projection_sha256']
            public.update(native_execution_projection_sha256=projection, request_id=row['request_id'])
            result = select_scenarios(self.providers[projection], public, self.templates[binding['component_id']])
            if (not result['unconditional_forecast_complete'] or len(result['outcomes']) != 1
                    or result['covered_probability_mass'] != 1.
                    or result['outcomes'][0]['scenario']['weight'] != 1.):
                raise UnknownEvidence('COMPLETE_SINGLE_EXACT_CONTENT_WORK_SCENARIO_REQUIRED_FOR_CURRENT_DAG_ENGINE')
            selected = result['outcomes'][0]
            if selected['template'] != expected:
                raise ValueError('Selected generation template differs from exact executed component DAG')
            results.append(dict(request_id=row['request_id'], public_features=public,
                work=selected['scenario'], selected_template=selected['template'], coverage=result,
                input_token_feature_basis='Immutable exact public content and source-bound training tokenization',
                future_actual_output_used=False))
        return dict(scope='EXACT_KNOWN_ORIGINAL_COMPONENT_CONTENT_AND_GENERATION_WORK', rows=results,
            all_declared_requests_retained=True, unknown_scenario_mass_dropped=False,
            unseen_input_generalization_claimed=False, future_actual_output_used=False,
            physical_accuracy_validated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specification', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); spec = checked(receipt(args.specification))
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Exclusive component support output required')
    save(output, build(spec['composite'], spec['generation_work'], spec['light_reviews'], spec['aliases']))


if __name__ == '__main__':
    main()
