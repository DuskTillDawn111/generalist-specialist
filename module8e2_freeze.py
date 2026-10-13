"""Module 8E-2 preregistration and CPU feasibility only. No serving imports."""
import argparse
import ast
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/module8e2_heldout_freeze'
E1 = ROOT / 'results/module8e1_method_revision'
B = ROOT / 'results/module8b_quality_pool_finalization'
C = ROOT / 'results/module8c_final_real_search'
D = ROOT / 'results/module8d_baseline_freeze'
TASKS = ['text', 'image', 'ocr', 'asr', 'visual_qa', 'lid']
SEED = 20261004


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def freeze(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text() == content, 'Immutable artifact differs: ' + name
    else:
        path.write_text(content)
    sidecar = Path(str(path) + '.sha256')
    if sidecar.exists():
        assert sidecar.read_text() == sha(path) + '\n'
    else:
        sidecar.write_text(sha(path) + '\n')
    return ref(path)


def verify_file(path):
    assert sha(path) == Path(str(path) + '.sha256').read_text().strip(), str(path)


def event(stage, prerequisites):
    path = OUT / ('stage_' + stage + '.json')
    if path.exists():
        verify_file(path)
        assert read(path)['prerequisites'] == prerequisites
        return
    freeze(path.name, {'stage': stage, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                      'prerequisites': prerequisites, 'new_GPU_runs': 0,
                      'heldout_runtime_results_read': 0})


def ast_method(path, cls, method):
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)


def verify_sources():
    for name in ['method_v1_freeze.json', 'method_v2_spec.json', 'method_v2_source_hashes.json']:
        verify_file(E1 / name)
    v1 = read(E1 / 'method_v1_freeze.json')
    v2 = read(E1 / 'method_v2_source_hashes.json')
    records = list(v1['sources'].values()) + list(v1['archived_sources'].values())
    records += [v1['wrapper'], v2['spec'], v2['shared_v1_sources']]
    records += list(v2['sources'].values())
    definitions = read(D / 'baseline_definitions.json')
    verify_file(D / 'baseline_definitions.json')
    records += list(definitions['definition_inputs'].values())
    for r in records:
        assert sha(r['path']) == r['sha256'], r['path']
    checks = {}
    for method in ['_seed', '_routing', '_key', '_memory', '_memory_increment',
                   '_candidate_rejection', '_warm_start']:
        a = ast_method(ROOT / 'deployment_search/integer_search.py', 'IntegerGuidedDeploymentSearch', method)
        b = ast_method(ROOT / 'method_v2/integer_search.py', 'MethodV2', method)
        checks[method] = ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    def suffix(node):
        loop = next(n for n in node.body if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Name) and n.target.id == 'offset')
        index = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'best' for t in n.targets))
        return ast.dump(ast.Module(body=loop.body[index:], type_ignores=[]), include_attributes=False)
    checks['runtime_evaluation_accept_rollback_stopping'] = suffix(ast_method(
        ROOT / 'deployment_search/search.py', 'GuidedDeploymentSearch', 'search')) == suffix(ast_method(
        ROOT / 'method_v2/engine.py', 'RevisedGuidedSearch', 'search'))
    from method_v1 import config
    from method_v2 import frozen_config
    checks['same_config'] = asdict(config()) == asdict(frozen_config())
    assert all(checks.values()), checks
    assert read(E1 / 'module8e1_gate.json')['MODULE8E2_ALLOWED']
    return v1, definitions, checks


def preregister():
    import numpy as np
    import scipy
    v1, definitions, checks = verify_sources()
    dev = read(E1 / 'development_workload_designation.json')
    matrix = read(B / 'final_benchmark_workload_matrix.json')
    vectors = {p['name']: p['lambda_rps'] for p in matrix['proposals']}
    assert vectors == dev['lambda_vectors'] and len(vectors) == 6
    freeze('development_workload_exclusion.json', {
        'DEVELOPMENT_WORKLOAD_IDS': dev['workloads'], 'lambda_vectors': vectors,
        'lambda_sha256': {n: digest(x) for n, x in vectors.items()},
        'workload_record_sha256': {p['name']: digest(p) for p in matrix['proposals']},
        'matrix': ref(B / 'final_benchmark_workload_matrix.json'),
        'designation': ref(E1 / 'development_workload_designation.json'),
        'excluded_from_heldout': True, 'renaming_or_lambda_retuning_as_heldout_allowed': False,
        'final_v2_performance_advantage_claim_allowed': False})
    source = ROOT / 'results/candidate_pool_v2_service_demand/updated_service_demand_catalog.json'
    reference_ms = read(source)['models']['qwen']['service_time_ms']
    current = read(ROOT / 'results/candidate_pool_v2_1/service_demand_v2_1.json')
    assert reference_ms == current['models']['qwen']['service_time_ms']
    s_ref = {t: reference_ms[t] / 1000 for t in TASKS}
    # No RNG draw, seeder, candidate generation, or runtime inspection before these files exist.
    freeze('heldout_generation_spec.json', {
        'module': '8E-2', 'frozen_before_RNG_draw_and_planning': True, 'task_order': TASKS,
        'reference': {'model': 'qwen', 's_ref_seconds': s_ref, 's_ref_ms': reference_ms,
                      'source': ref(source), 'Module8A_generator_source': ref(ROOT / 'quality_workload_audit/demand.py'),
                      'choice_basis': 'Reuse the uniform six-task Module 8A frozen Qwen standalone reference; no method/candidate/runtime outcomes enter its selection.',
                      'admission_caveat': 'Qwen OCR is a normalization reference only; the unchanged quality policy disallows its deployed OCR path. A predeclared Qwen + admitted PaddleOCR feasibility witness repairs OCR.',
                      'replacement_after_generation_allowed': False},
        'sampling': {'distribution': 'Dirichlet', 'concentration': [1., 1., 1., 1., 1., 1.],
                     'draw_count': 3, 'dominant_task_requirement': None,
                     'minimum_active_share': .01,
                     'minimum_active_share_formula': 'alpha[k] = 0.01 + (1 - 6*0.01)*raw_dirichlet[k]',
                     'minimum_active_share_policy': 'Apply the affine simplex floor to every draw, without rejection. All six tasks stay active.',
                     'resampling_allowed': False},
        'intensity': {'reference_capacity': 1., 'L_definition': 'L = sum(lambda[k]*s_ref[k]) / 1.0',
                      'rho_max_unchanged': .8, 'moderate_L': .4, 'high_L': .72,
                      'moderate_fraction_of_reference_planning_limit': .5,
                      'high_fraction_of_reference_planning_limit': .9,
                      'choice_basis': 'Fixed 50% and 90% of the existing rho_max=0.8 single-reference planning bound, before sampling; independent of search behavior.',
                      'feasibility_witness': 'One Qwen for its five admitted tasks, one PaddleOCR for OCR. Frozen repaired OCR s0 <= Qwen OCR s_ref, both single capacity=1; each family demand <= L < rho_max for every simplex draw. Witness memory checked after workload hashes.',
                      'limitation': 'High approaches the reference planning bound; it need not saturate a faster mixed deployment or the true GPU.'},
        'generation_formula': 'D_ref=L*1.0; lambda[k]=D_ref*alpha[k]/s_ref[k]; raw_request_share[k]=lambda[k]/sum(lambda); reference_share[k]=lambda[k]*s_ref[k]/sum(lambda*s_ref)',
        'collision_rule': 'Exact development lambda or demand-share collision => preserve samples, close gate, no redraw. No tuned variants of development targets.',
        'infeasible_handling': 'Preserve all six workloads and hashes; mark failed conditions and HELDOUT_GPU_CAMPAIGN_READY=false. No redraw, rescale, removal, replacement, or candidate-based rejection.',
        'planning_allowed_checks': ['capability', 'quality_admission', 'service_demand', 'memory', 'seed_existence'],
        'planning_candidate_generation_allowed': False,
        'new_GPU_runs_allowed_in_8E2': False, 'heldout_runtime_read_allowed': False})
    freeze('heldout_rng_freeze.json', {
        'HELDOUT_MIX_SEED': SEED, 'engine': 'numpy.random.Generator(numpy.random.PCG64(20261004))',
        'draw_call': 'rng.dirichlet(np.ones(6), size=3), one call, task order from generation spec',
        'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
        'python_version': sys.version, 'python_executable': sys.executable,
        'global_rng_policy': 'No project-wide workload-simplex RNG policy. Existing per-method random-neighbor seeds and timed-run seeds have different scopes and remain unchanged.',
        'RNG_stream_used_for_execution_schedule': False,
        'timed_search_seeds': [7101, 7202, 7303], 'confirmation_seeds': [8101, 8202]})
    protocol = read(C / 'demand_balanced/protocol.json')
    freeze('method_comparison_freeze.json', {
        'methods': {'M1': {'method': 'method_v1', 'entrypoint': 'method_v1.create',
                           'definition': ref(E1 / 'method_v1_freeze.json'),
                           'neighborhood': 'original x±1', 'Top_B': 'original exact delta_z order'},
                    'M2': {'method': 'method_v2', 'entrypoint': 'method_v2.MethodV2',
                           'definition': ref(E1 / 'method_v2_spec.json'),
                           'sources': ref(E1 / 'method_v2_source_hashes.json'),
                           'neighborhood': 'targeted SWAP / prospective replacement closure plus unchanged base',
                           'Top_B': 'candidate-class diversified Top-2'},
                    'B2': {'definition_source': ref(D / 'baseline_definitions.json'),
                           'rule_exact_copy': definitions['B2'], 'implementation_source': ref(ROOT / 'module8d_baseline_freeze/definitions.py'),
                           'search_allowed': False}},
        'same_M1_M2': ['workload', 'cold MILP seed generation', 'MILP witness/deployment', 'candidate pool',
                       'quality admission', 'empirical replica capacity', 'memory model', 'canonical routing',
                       'RuntimeEvaluator', 'runtime objective', 'measurement seeds and settings', 'failure policy'],
        'search_config': v1['config'], 'search_budget': v1['search_budget'],
        'predictor': None, 'deployment_enumeration': 0,
        'seeder': {'source': ref(ROOT / 'deployment_core/milp_seeder.py'),
                   'backend_source': ref(ROOT / 'deployment_core/empirical_milp.py'),
                   'RNG_seed': None, 'rule': 'Existing deterministic SciPy/HiGHS minimum empirical memory then minimum count; no extra solver seed introduced.',
                   'campaign_rule': 'Both cold entry points must reproduce the frozen witness exactly; mismatch blocks execution without choosing a preferred seed.'},
        'frozen_inputs': definitions['definition_inputs'],
        'catalog_loader': ref(ROOT / 'module8c_final_real_search/common.py'),
        'runtime_context_template': ref(C / 'demand_balanced/runtime_context.json'),
        'runtime_evaluator': ref(ROOT / 'integer_replica_pilot/evaluator.py'),
        'runtime_harness': ref(ROOT / 'integer_replica_pilot/runtime.py'),
        'runtime_objective': ref(ROOT / 'deployment_search/real_objective.py'),
        'runtime_objective_base': ref(ROOT / 'deployment_search/scoring.py'),
        'measurement_settings': protocol['measurement_settings'], 'search_seeds': protocol['search_seeds'],
        'confirmation_seeds': protocol['confirmation_seeds'], 'confirmation_order': protocol['confirmation_order'],
        'confirmation_influences_search': False,
        'failure_policy': protocol['failure_policy'], 'DQA': protocol['DQA'],
        'execution_environment_overrides': definitions['measurement_protocol']['execution_environment_overrides'],
        'hardware': 'Same frozen RTX 4090, 49140 MiB; verify recorded hardware/environment/weights/adapters before launch. Drift closes gate; no second hardware.',
        'equal_budget_static_verified': all(checks.values()), 'fresh_AST_equality_checks': checks,
        'ROUTING_REVISION_PENDING': True,
        'routing_scope': 'M1/M2 retain identical canonical routing. B2 retains its 8D-1 fixed task-selected routing. No routing revision.',
        'secondary_reporting': ['weighted mean E2E', 'weighted task P95 (not pooled P95)', 'stability',
                                'completion ratio', 'peak VRAM', 'N_G', 'N_S', 'full deployment vector',
                                'gap_to_B2_v1', 'gap_to_B2_v2'],
        'mechanism_reporting': {
            'M2_each_iteration': ['all base candidates including infeasible/tabu', 'targeted swaps and origins',
                                 'exact replacement closures', 'Class P', 'Class R', 'Top-B composition and rank',
                                 'runtime evaluated with logical and raw IDs', 'accepted', 'rollback reason', 'budget/stopping'],
            'beneficial_replacement_definition': 'An actually evaluated SWAP/replacement that passes the frozen objective against its pre-transition incumbent; separately flag actual acceptance and final-confirmation improvement.',
            'M1_attribution': 'Match exact deployment AND routing against all v1 traces; record not_generated, generated_later (iteration indices), rank_too_low, or budget/stopping_missed. Multiple applicable flags allowed; do not invent v1 runtime for unmatched states.',
            'scope_caution': 'Different incumbent paths limit causal attribution. Structural access and accepted improvement are evidence, not proof of an unmeasured v1 counterfactual.',
            'no_extra_runtime_for_mechanism': True},
        'analysis_data_source': 'Independent final confirmations (two reps), never choose best repetition; search aggregates separately support transition/mechanism evidence.',
        'aggregation': 'Unchanged evaluator aggregate: worst stability across reps; equal-repetition arrival-weighted means and weighted task P95; memory objective uses frozen planning memory; observed peak VRAM reported separately.',
        'campaign_scope': ['M1', 'M2', 'frozen B2 only'], 'GPU_campaign_started': False})
    freeze('tie_policy.json', {
        'objective_class': 'NoiseAwareRuntimeObjective(mean_tolerance_ms=1.0,min_relative_improvement=0.05)',
        'sources': [ref(ROOT / 'deployment_search/scoring.py'), ref(ROOT / 'deployment_search/real_objective.py')],
        'stability_order_best_first': ['steady', 'recoverable_congested', 'unstable'],
        'same_steady_order': ['weighted mean E2E with existing 1ms tolerance', 'weighted task P95', 'frozen empirical memory'],
        'min_relative_improvement': .05, 'mean_tolerance_ms': 1.,
        'pairwise_rule': 'Call the frozen objective in BOTH directions with final-confirmation aggregates and actual planning memories. WIN if M2 improves M1 only; LOSS if M1 improves M2 only; TIE if neither; both true is a protocol error/blocker.',
        'same_nonsteady_rule': 'Existing policy compares equal nonsteady states as no improvement => TIE, even when reported latencies differ. Do not introduce a new nonsteady tie-break.',
        'noise_gate_caveat': 'The existing 5% MEAN gain gate also applies when base comparator returns p95_improved; memory_improved bypasses this gate. Preserve exact boundary behavior.',
        'missing_or_invalid_confirmation': 'INCOMPLETE; no WIN/TIE/LOSS, no numeric imputation, aggregate status withheld and gate closed.',
        'memory_metric': 'frozen empirical planning memory, not observed peak VRAM', 'GPU_results_seen': False})
    freeze('success_criteria.json', {
        'frozen_before_GPU': True, 'unit': 'all six workloads, equal weight; two intensities of one mixture are correlated, not six independent random mixture samples',
        'prerequisite': 'All six valid final confirmations and complete source/equal-budget/order/sharing audits. Otherwise report INCOMPLETE/PROTOCOL_INVALID, withhold Revision validation status and B2_GAP_STATUS.',
        'variables': {'W': 'M2 primary WIN count', 'T': 'TIE count', 'L': 'LOSS count',
                      'SR': 'number of lower final stability ranks for M2 vs M1',
                      'severe': 'any same-steady relative mean regression >=20%',
                      'clear_regression': 'any same-steady relative mean regression >=10%',
                      'activated': 'at least one runtime-evaluated Class R SWAP or replacement',
                      'helped': 'at least one accepted objective-beneficial SWAP/replacement, with documented v1 access limitation, on a workload whose final-confirmation primary result is WIN'},
        'ordered_rules': [
            {'status': 'NOT_SUPPORTED', 'if': 'L>=3 OR SR>=2 OR severe', 'meaning': 'Systematic primary losses or clear severe/stability degradation.'},
            {'status': 'STRONG_SUPPORT', 'if': 'W>=3 AND W>L AND L<=1 AND SR==0 AND NOT clear_regression AND helped', 'meaning': 'Overall primary advantage with actual mechanism help and no clear systematic regression.'},
            {'status': 'SUPPORT', 'if': 'W>L AND W>=1 AND activated AND SR<=1 AND NOT severe', 'meaning': 'Positive but limited primary improvement; ties allowed, mechanism activated.'},
            {'status': 'NEUTRAL', 'if': 'otherwise after no NOT_SUPPORTED rule', 'meaning': 'Mostly tied, weak mechanism evidence, or mixed results without systematic degradation; no supported advantage claim.'}],
        'B2_is_not_revision_success_input': True,
        'B2_gap': {
            'per_workload': 'Report pairwise outcome vs B2 and signed mean gap g=mean_method/mean_B2-1 only when both final-confirmed steady and mean_B2>0; also P95 gap separately. Otherwise numeric gap=null with reason.',
            'aggregate': 'All six conditions retained. If all three methods are steady on all six, use equal-workload mean excess gaps G1=mean(max(g1,0)), G2=mean(max(g2,0)). No cherry-picked steady subset.',
            'ordered_status_rules': [
                'CLOSED if M2 has zero primary LOSS vs B2 across all six (ties allowed); distinguish already_closed_v1.',
                'Otherwise, if all-six steady metric eligible: REDUCED if G1>0 and (G1-G2)/G1>=0.05; WORSENED if G2>G1 and (G1==0 or (G2-G1)/G1>=0.05); else UNCHANGED.',
                'Otherwise use primary LOSS counts vs B2 across all six: REDUCED if M2 count < M1; WORSENED if >; UNCHANGED if equal. Label objective-count fallback and report stability differences.'],
            'statuses': ['CLOSED', 'REDUCED', 'UNCHANGED', 'WORSENED'],
            'no_requirement_to_beat_B2_everywhere': True},
        'ROUTING_REVISION_PENDING': True,
        'claim_scope': 'Only neighborhood + candidate-selection revision under frozen canonical routing, existing task data/pool and one hardware. No routing-solved, statistical significance, dynamic, ablation, second-hardware, or development-workload final advantage claim.'})
    freeze('measurement_sharing_policy.json', {
        'enabled': True, 'identity_exact_no_rounding': ['full integer deployment vector', 'exact routing rates/identity',
            'model revisions/weight hashes/adapters/precision/backend', 'environment/hardware/context hash',
            'workload hash and exact lambda', 'full protocol/settings/aggregation', 'phase contract', 'repetition seed'],
        'identity_source': ref(ROOT / 'integer_replica_pilot/evaluator.py'),
        'search_phase': 'M1/M2/B2 may share identical search raw runs only for seeds 7101/7202/7303 with full identity match.',
        'confirmation_phase': 'M1/M2 may share across methods within the SAME role (seed or final) and repetition seed. Never share seed with final within a method, even if deployment/routing coincide. B2 fixed confirmations remain a separate fixed role. No search-confirmation sharing.',
        'logical_records': ['method', 'workload_hash', 'logical_slot', 'phase', 'role', 'repetition_seed',
                            'deployment', 'routing_hash', 'measurement_identity_hash', 'physical_raw_id',
                            'raw_sha256', 'shared_with', 'cache_hit', 'budget_charge'],
        'cached_logical_budget_charge': 'Every selected logical search evaluation consumes one of the method\'s six slots, including shared/cache hits; no freed slots or repetitions.',
        'raw_donor_rule': 'Earliest compatible scheduled slot, then physical ID lexically; never select donors by metrics.',
        'candidate_information_firewall': 'Each method commits its own candidates/ranks/Top-B using its own prior returned measurements. At each round both Top-B lists are hash-committed before any round runtime values are released. Another method\'s cached state/value is invisible until this method independently selects that exact state.',
        'shared_raw_is_one_sample': True, 'sample_reporting': 'Report logical references and distinct physical runs separately; do not treat shared method measurements as independent evidence.',
        'invalid_measurement': 'Preserve raw/error, stop campaign; no reuse as valid runtime, no automatic retry or budget growth.',
        'cross_workload_sharing_allowed': False})
    event('preregistered', {n: ref(OUT / n) for n in ['development_workload_exclusion.json',
        'heldout_generation_spec.json', 'heldout_rng_freeze.json', 'method_comparison_freeze.json',
        'tie_policy.json', 'success_criteria.json', 'measurement_sharing_policy.json']})
    print('Preregistered generation/reference/RNG/intensities/methods/analysis before sampling.', flush=True)


def generate():
    import numpy as np
    verify_file(OUT / 'stage_preregistered.json')
    for r in read(OUT / 'stage_preregistered.json')['prerequisites'].values():
        assert sha(r['path']) == r['sha256']
    spec = read(OUT / 'heldout_generation_spec.json')
    rng_spec = read(OUT / 'heldout_rng_freeze.json')
    assert np.__version__ == rng_spec['numpy_version']
    rng = np.random.Generator(np.random.PCG64(rng_spec['HELDOUT_MIX_SEED']))
    raw = rng.dirichlet(np.array(spec['sampling']['concentration']), size=3)
    floor = spec['sampling']['minimum_active_share']
    shares = floor + (1 - len(TASKS) * floor) * raw
    dev = read(OUT / 'development_workload_exclusion.json')
    s_ref = spec['reference']['s_ref_seconds']
    dev_shares = {n: {t: x[t]*s_ref[t]/math.fsum(x[k]*s_ref[k] for k in TASKS)
                      for t in TASKS} for n, x in dev['lambda_vectors'].items()}
    workloads = []
    for i, alpha in enumerate(shares, 1):
        alpha = {t: float(alpha[j]) for j, t in enumerate(TASKS)}
        for intensity in ['moderate', 'high']:
            load = spec['intensity'][intensity + '_L']
            demand = load * spec['intensity']['reference_capacity']
            rates = {t: demand * alpha[t] / s_ref[t] for t in TASKS}
            total = math.fsum(rates.values())
            actual_demand = math.fsum(rates[t]*s_ref[t] for t in TASKS)
            collisions = [n for n, x in dev['lambda_vectors'].items() if x == rates or dev_shares[n] == alpha]
            workloads.append({'id': f'heldout_mix_{i:02d}_{intensity}', 'mixture_id': i,
                'intensity': intensity, 'lambda_rps': rates, 'total_rps': total,
                'raw_request_share': {t: rates[t]/total for t in TASKS},
                'reference_demand_share': alpha,
                'recomputed_reference_demand_share': {t: rates[t]*s_ref[t]/actual_demand for t in TASKS},
                'raw_dirichlet_share': {t: float(raw[i-1, j]) for j, t in enumerate(TASKS)},
                'total_reference_demand_seconds_per_second': demand,
                'reference_normalized_load_L': load, 's_ref_seconds': s_ref,
                'RNG_seed': SEED, 'generation_formula': spec['generation_formula'],
                'generation_spec_sha256': sha(OUT / 'heldout_generation_spec.json'),
                'development_collisions': collisions, 'planning_used_to_generate': False})
    freeze('heldout_workloads.json', {'task_order': TASKS, 'workloads': workloads,
        'mixtures': 3, 'intensities_per_mixture': 2, 'resamples': 0, 'planning_results_used': False,
        'heldout_runtime_results_read': 0})
    freeze('heldout_workload_hashes.json', {
        'serialization': 'SHA-256 of UTF-8 json.dumps(record,sort_keys=True,separators=(comma,colon),allow_nan=False); lambda hash uses the exact lambda dict. Whole-file hash covers pretty JSON bytes.',
        'whole_file': ref(OUT / 'heldout_workloads.json'),
        'workloads': {w['id']: {'workload_sha256': digest(w), 'lambda_sha256': digest(w['lambda_rps']),
                              'demand_share_sha256': digest(w['reference_demand_share'])} for w in workloads}})
    event('workloads_hashed', {'preregistration': ref(OUT / 'stage_preregistered.json'),
                              'workloads': ref(OUT / 'heldout_workloads.json'),
                              'hashes': ref(OUT / 'heldout_workload_hashes.json')})
    print('Generated and hashed exactly 3 mixtures x 2 intensities; zero resamples.', flush=True)


def sanity():
    verify_file(OUT / 'stage_workloads_hashed.json')
    verify_file(OUT / 'heldout_workloads.json')
    from module8c_final_real_search.common import inputs
    from deployment_core import Workload
    from deployment_core.milp_seeder import MilpDeploymentSeeder, validate_seed
    from deployment_core.replica_calibration import ReplicaPlanningPolicy
    from deployment_search.routing import solve_canonical_routing
    from deployment_search.result import jsonable
    from deployment_search.tests.fixtures import forbid_enumeration
    catalog, hardware, roles, cal = inputs()
    policy = ReplicaPlanningPolicy(catalog, 'empirical', 'empirical', cal)
    bounds, _ = policy.bounds(hardware)
    admission = read(B / 'quality_admission_matrix.json')
    admitted = {(r['task'], r['model']) for r in admission['rows'] if r['quality_admissible']}
    formal = sorted(read(ROOT / 'results/candidate_pool_v2_1/candidate_pool_v2_1.json')['formal_model_ids'])
    hashes = read(OUT / 'heldout_workload_hashes.json')
    rows = []
    with forbid_enumeration() as guards:
        for w in read(OUT / 'heldout_workloads.json')['workloads']:
            assert digest(w) == hashes['workloads'][w['id']]['workload_sha256']
            workload = Workload(w['lambda_rps'])
            seed = MilpDeploymentSeeder().solve(catalog, workload, hardware, .8, replica_bounds=bounds,
                capacity_mode='empirical', memory_mode='empirical', calibration=cal)
            errors = validate_seed(seed, catalog, workload, hardware, .8, bounds, planning_policy=policy)
            route = None if not seed.feasible else solve_canonical_routing(seed.deployment, workload,
                catalog, .8, replica_bounds=bounds, capacity_mode='empirical', calibration=cal)
            if route is None: errors.append('seed canonical route missing')
            witness_x = {m: int(m in ['qwen', 'paddleocr']) for m in catalog}
            witness_route = {(t, m): w['lambda_rps'][t] if m == ('paddleocr' if t == 'ocr' else 'qwen') else 0.
                             for t in TASKS for m in catalog}
            def check_state(x, routing):
                pressure = {m: math.fsum(routing[t, m]*catalog[m].service_time_ms.get(t, 0)/1000
                             for t in TASKS)/policy.capacity(m, x[m]) if x[m] else 0. for m in catalog}
                quality = all((t, m) in admitted and x[m] > 0 for (t, m), value in routing.items() if value > 0)
                conserved = all(math.isclose(math.fsum(routing[t, m] for m in catalog), w['lambda_rps'][t],
                               rel_tol=1e-7, abs_tol=1e-12) for t in TASKS)
                memory = policy.deployment_memory(x)
                return {'quality_admissible': quality, 'demand_conserved': conserved,
                        'max_pressure': max(pressure.values()), 'family_pressure': pressure,
                        'memory_mib': memory, 'memory_feasible': memory <= hardware.gpu_memory_mib,
                        'service_demand_feasible': max(pressure.values()) <= .8 + 1e-9}
            witness = check_state(witness_x, witness_route)
            # Exact Module 8D-1 deterministic B2 rule, fixed one-replica state; no search/ranking.
            selected = {}
            for task in workload.active_tasks:
                specialists = [m for m in catalog if roles[m] == 'specialist' and task in catalog[m].supported_tasks]
                options = specialists or [m for m in catalog if roles[m] == 'generalist' and task in catalog[m].supported_tasks]
                if not options:
                    errors.append('B2 uncovered task: ' + task)
                else:
                    selected[task] = min(options, key=lambda m: (catalog[m].service_time_ms[task], m))
            b2_x = {m: int(m in set(selected.values())) for m in catalog}
            b2_route = {(t, m): w['lambda_rps'][t] if selected.get(t) == m else 0.
                        for t in TASKS for m in catalog}
            b2_check = check_state(b2_x, b2_route)
            for label, checks in [('reference_witness', witness), ('B2_fixed_state', b2_check)]:
                for key in ['quality_admissible', 'demand_conserved', 'memory_feasible', 'service_demand_feasible']:
                    if not checks[key]: errors.append(label + ': ' + key)
            capability = all(any(t in p.supported_tasks for p in catalog.values()) for t in workload.active_tasks)
            if not capability: errors.append('capability')
            rows.append({'workload': w['id'], 'workload_sha256': digest(w), 'feasible': not errors,
                'errors': errors, 'capability_feasible': capability, 'seed_exists': seed.feasible,
                'seed': jsonable(seed), 'seed_full_vector': {m: (seed.deployment or {}).get(m, 0) for m in formal},
                'canonical': jsonable(route), 'reference_witness': witness,
                'B2': {'deployment': {m: b2_x.get(m, 0) for m in formal}, 'selected_task_model': selected,
                       'routing_rps': {t: {m: b2_route[t, m] for m in catalog} for t in TASKS},
                       'checks': b2_check}, 'candidate_generation': 0, 'runtime_evaluations': 0})
        enumeration_calls = sum(g.call_count for g in guards)
    assert enumeration_calls == 0
    freeze('planning_sanity_check.json', {'status': 'PASS' if all(r['feasible'] for r in rows) else 'FAIL',
        'workloads': rows, 'workloads_hashed_before_planning': ref(OUT / 'stage_workloads_hashed.json'),
        'method_candidate_generation': 0, 'deployment_enumeration': enumeration_calls,
        'GPU_runs': 0, 'heldout_runtime_results_read': 0,
        'infeasible_handling_applied': 'Preserve samples and close gate; no resampling/rescaling.',
        'scope': 'CPU MILP seed + canonical LP and deterministic fixed B2/witness feasibility only; no planning advantage selection.',
        'bounds': bounds, 'hardware_memory_mib': hardware.gpu_memory_mib})
    event('sanity_complete', {'hashes': ref(OUT / 'heldout_workload_hashes.json'),
                              'sanity': ref(OUT / 'planning_sanity_check.json')})
    print('CPU planning sanity: ' + read(OUT / 'planning_sanity_check.json')['status'], flush=True)


def finish():
    verify_sources()
    verify_file(OUT / 'stage_sanity_complete.json')
    workloads = read(OUT / 'heldout_workloads.json')['workloads']
    schedule = []
    # Freeze logical request slots: future state identities depend on the unchanged adaptive search.
    for wi, w in enumerate(workloads):
        order = ['M1', 'M2'] if wi % 2 == 0 else ['M2', 'M1']
        slots = []
        for method in order:
            slots.append({'phase': 'search', 'role': 'seed', 'method': method, 'logical_state_index': 0, 'repetitions': [7101, 7202, 7303]})
        # B2 has no decision process. Alternate its position in the initial block by workload index.
        slots.insert(wi % 3, {'phase': 'search', 'role': 'fixed', 'method': 'B2', 'logical_state_index': 0, 'repetitions': [7101, 7202, 7303]})
        for iteration, ranks in [(1, [1, 2]), (2, [1, 2]), (3, [1])]:
            turn = order if iteration % 2 == 1 else list(reversed(order))
            for rank in ranks:
                for method in turn:
                    slots.append({'phase': 'search', 'role': 'candidate', 'method': method,
                                  'iteration': iteration, 'Top_B_rank': rank,
                                  'logical_state_index': (iteration - 1)*2 + rank,
                                  'repetitions': [7101, 7202, 7303]})
        for role, index in read(OUT / 'method_comparison_freeze.json')['confirmation_order']:
            turn = order if index == 1 else list(reversed(order))
            for method in turn:
                slots.append({'phase': 'confirmation', 'role': role, 'method': method,
                              'repetition': index, 'seed': [8101, 8202][index - 1]})
        for index, seed in enumerate([8101, 8202], 1):
            slots.insert(len(slots)-2 if index == 1 else len(slots),
                         {'phase': 'confirmation', 'role': 'fixed', 'method': 'B2', 'repetition': index, 'seed': seed})
        schedule.append({'workload': w['id'], 'workload_sha256': digest(w), 'slots': slots})
    freeze('execution_order.json', {
        'frozen_before_GPU': True, 'schedule_type': 'Deterministic interleaved logical-state blocks',
        'workload_order': [w['id'] for w in workloads], 'workloads': schedule,
        'state_identity_binding': 'Adaptive candidate vectors cannot be named before runtime; bind each predeclared slot to that method\'s independently committed exact Top-B state. Never reorder slots based on metrics.',
        'round_barrier': 'Both methods finish ranking/hash-committing their round using only their prior local metrics before either receives any new round measurement; commit <=Top2 even though iteration3 has only one remaining runtime slot.',
        'repetition_granularity': 'Each search slot is an atomic three-repetition state block, sequential seeds; interleave method state blocks as listed.',
        'unused_slot_rule': 'A stopped method, absent rank, or exhausted budget leaves a logged SKIPPED slot. Never transfer slots, resample, replace candidates or start another workload early to fill gaps.',
        'cache_hit_rule': 'Log a logical evaluation at its scheduled position; physical donor earliest exact-compatible scheduled request. Deliver only at the receiving slot, after independent selection; no opportunistic prefetch.',
        'B2_insertion_rule': 'Initial search block position=workload_index mod 3 among two seed slots; fixed confirmation rep1 before the final two method confirmation slots, rep2 after them.',
        'confirmation_barrier': 'Both methods have frozen final deployment/routing and trace hash before any confirmation; confirmations cannot alter search or final selection.',
        'timing_controls': 'Reuse existing worker startup/warmup/drain/shutdown and serial single-GPU execution; no overlapping GPU jobs, outcome-triggered cooldowns or retries. Record timestamps/telemetry and every skipped/failed slot.',
        'physical_order': 'Ordered list after removing logged SKIPPED slots and exact identity cache hits; preserved logical order is authoritative.',
        'scheduler_implementation': 'Campaign execution adapter must implement this contract around unchanged engines/evaluator; this module freezes the protocol and does not build or launch a GPU runner.'})
    logical_search = 6 * (6 + 6 + 1)
    confirmations = 6 * (2 + 2 + 1)
    no_sharing_runs = logical_search * 3 + confirmations * 2
    physical_states = 6 * (6 + 6 + 1 - 1)
    search_runs = physical_states * 3
    confirm_runs = 6 * (4 + 4 + 2 - 2)
    freeze('gpu_budget_estimate.json', {
        'heldout_workloads': 6, 'M1_states_per_workload_including_seed': 6,
        'M2_states_per_workload_including_seed': 6, 'B2_fixed_states_per_workload': 1,
        'search_repetitions_per_state': 3, 'confirmation_repetitions_per_seed_or_final': 2,
        'B2_protocol': 'Same 8D-1 three fixed-state search-phase reps plus two independent fixed confirmations.',
        'worst_case_logical_evaluations': {'search_state_evaluations': logical_search,
            'confirmation_state_role_evaluations': confirmations, 'total_aggregate_requests': logical_search + confirmations,
            'logical_timed_run_references': no_sharing_runs},
        'without_sharing': {'physical_deployment_routing_workload_states': logical_search,
            'search_timed_runs': logical_search*3, 'confirmation_timed_runs': confirmations*2,
            'total_timed_runs': no_sharing_runs},
        'strict_sharing_worst_case': {'physical_deployment_routing_workload_states': physical_states,
            'physical_search_state_bundles': physical_states,
            'physical_confirmation_state_role_bundles': 6*(2+2+1-1),
            'search_timed_runs': search_runs, 'confirmation_timed_runs': confirm_runs,
            'worst_case_timed_runs': search_runs + confirm_runs,
            'guaranteed_savings': {'common_seed_search_runs': 6*3, 'common_seed_confirmation_runs': 6*2}},
        'sharing_assumptions': 'Same frozen cold seed, routing, context, all exact identity fields for M1/M2. All five other search states per method and B2 are distinct in worst case. M1/M2 final confirmations do not overlap in worst case. Seed/final independence retained within methods.',
        'identity_mismatch_rule': 'Stop and close gate; do not silently change to no-sharing protocol or extend budget.',
        'further_sharing': 'Exact candidate/final/B2 overlaps reduce physical runs only, never logical budget or independent sample count.',
        'time_estimate': {'measurement_window_seconds': 210,
            'worst_case_timed_window_hours': (search_runs+confirm_runs)*210/3600,
            'warmup_plus_window_hours_excluding_startup_drain_shutdown': (search_runs+confirm_runs)*220/3600},
        'actual_GPU_runs': 0, 'campaign_started': False})
    sanity_result = read(OUT / 'planning_sanity_check.json')
    conditions = {
        'generation_independent_of_v2_runtime': read(OUT / 'heldout_workloads.json')['heldout_runtime_results_read'] == 0,
        'six_workloads_frozen_and_hashed': len(workloads) == 6 and all(digest(w) == read(OUT / 'heldout_workload_hashes.json')['workloads'][w['id']]['workload_sha256'] for w in workloads),
        'methods_M1_M2_B2_frozen': True,
        'equal_budget_verified': read(OUT / 'method_comparison_freeze.json')['equal_budget_static_verified'],
        'success_criteria_frozen': (OUT / 'success_criteria.json').exists(),
        'execution_order_frozen': (OUT / 'execution_order.json').exists(),
        'measurement_sharing_policy_frozen': (OUT / 'measurement_sharing_policy.json').exists(),
        'planning_feasible': sanity_result['status'] == 'PASS',
        'development_exclusion_and_no_collision': all(not w['development_collisions'] for w in workloads)}
    for path in OUT.glob('*.json'):
        if path.name != 'artifact_manifest.json': verify_file(path)
    blockers = [k for k, v in conditions.items() if not v]
    freeze('heldout_campaign_gate.json', {'module': '8E-2', 'HELDOUT_GPU_CAMPAIGN_READY': not blockers,
        'conditions': conditions, 'blockers': blockers, 'ROUTING_REVISION_PENDING': True,
        'GPU_campaign_started': False, 'new_GPU_runs': 0, 'heldout_runtime_results_read': 0,
        'readiness_scope': 'Workload/protocol preregistration ready, not an execution authorization. This task stops here. Future runner must verify all identities, seed reproducibility, scheduler/sharing/firewall and budget contract before GPU launch.',
        'forbidden_in_this_task': ['heldout GPU search', 'method_v1/v2 changes', 'routing revision',
            'quality/candidate-pool/model changes', 'additional baselines', 'ablation', 'dynamic', 'second hardware']})
    event('complete', {'gate': ref(OUT / 'heldout_campaign_gate.json'),
                       'execution_order': ref(OUT / 'execution_order.json'),
                       'budget': ref(OUT / 'gpu_budget_estimate.json')})
    lines = ['# Module 8E-2 — Held-out workload and validation protocol freeze', '',
        f"HELDOUT_GPU_CAMPAIGN_READY = {str(not blockers).lower()}", '',
        '已完成 workload/protocol 预注册；新 GPU runs=0，held-out runtime result reads=0。v1/v2、routing、quality policy、candidate pool 和 models 均未修改。', '',
        '原六个 workload 永久列入 DEVELOPMENT_WORKLOAD_IDS；保存完整 record/lambda/hash 与来源，不可重命名、微调后用于 held-out 或最终 v2 advantage claim。', '',
        '沿用 Module 8A 的统一 Qwen standalone s0 reference（ms 转 s）。Qwen OCR 只用于 demand 归一化，不作为可部署的质量合格 OCR path。全 task feasibility witness 固定为 Qwen + PaddleOCR；不根据 planning 更换 reference。', '',
        '生成顺序证据：stage_preregistered.json → stage_workloads_hashed.json → stage_sanity_complete.json → stage_complete.json。生成与协议文件先落盘并 hash；单次 PCG64(20261004) Dirichlet([1]*6) 抽 3 个样本，再统一 affine floor alpha=0.01+0.94*draw；没有 rejection/resampling。', '',
        'moderate L=0.4，high L=0.72，分别是既有 rho_max=0.8 的 50% / 90%。L=sum(lambda*s_ref)，lambda=L*alpha/s_ref。High 接近 reference planning 边界，不声称真实 GPU 或所有混合 deployment 饱和。', '',
        '|Workload|L|Total rps|Seed max pressure|B2 max pressure|Planning|',
        '|---|---:|---:|---:|---:|---|']
    for w, r in zip(workloads, sanity_result['workloads']):
        z = r['canonical']['max_pressure'] if r['canonical'] else None
        lines.append(f"|{w['id']}|{w['reference_normalized_load_L']:.2f}|{w['total_rps']:.6f}|{z}|{r['B2']['checks']['max_pressure']:.6f}|{'PASS' if r['feasible'] else 'FAIL'}|")
    lines += ['', '各 workload 保存 exact lambda、raw request share、reference demand share、D_ref、seed、公式及 SHA-256。只进行 capability/quality/service-demand/memory/seed 检查；method candidates=0，enumeration=0，runtime evaluation=0。不可行按预注册规则保留样本并关闭 gate。', '',
        'M1/M2 共用 cold MILP seed 与同一 frozen 12-model formal pool（11 个有效 quality-admitted family）、empirical capacity/memory、canonical routing、RuntimeEvaluator 和 NoiseAwareRuntimeObjective(1ms,5%)。Top-B=2、iterations<=3、unique states<=6（含 seed）、3 reps/state、patience=1、seed/final 各2 confirmations，predictor=None。重新检查源码 hashes、配置与 seeder/routing/accept/rollback/stopping AST 等价。', '',
        'B2 严格沿用 8D-1：quality-admissible Specialist min s0，否则 Generalist min s0，model ID 解 ties，每个所选 model 单 replica；100% task traffic 到所选 model。其 fixed routing 与 M1/M2 canonical routing 的既有差异明确保留。', '',
        '执行顺序按 workload 索引交错 method 的 seed / 每轮 Top-B state block，B2 按固定索引插入。未来 adaptive state 无法事前指定 vector，因此冻结 logical slot 绑定规则：每轮双方先独立 commit Top-B，再返回测量。跳过停止/空位，不转移预算，不根据结果改顺序。search slot 内3 reps 原子执行；GPU serial。', '',
        '共享只允许 workload/deployment/routing/model/environment/protocol/phase/seed 全部 exact 相同的 raw measurement。跨 method 的共同 seed/search 可共享，confirmation 仅同 role 跨 method 共享；同 method seed/final 独立，即使 state 相同。共享仍分别记 logical evaluation 并计预算，不增加独立样本。', '',
        'Primary 使用 independent final-confirmation aggregates，双向调用既有 objective 冻结 WIN/TIE/LOSS。steady > recoverable_congested > unstable；同 steady mean/P95/planning memory；同非 steady 按既有政策 TIE，不新增比较规则。P95 改善仍受既有5% mean gain gate。success_criteria.json 将 STRONG_SUPPORT/SUPPORT/NEUTRAL/NOT_SUPPORTED 和独立 B2_GAP_STATUS 的计数、退化阈值、mechanism 条件全部预注册。未完成或 protocol invalid 时不输出成功判断。', '',
        'Mechanism 记录每轮 base/swaps/closures/Class P/R/Top-B/测量/接受/rollback，并对匹配 state 检查 v1 not generated / generated later / rank too low / budget or stopping missed；不增加 counterfactual GPU 测量。', '',
        '最坏 logical search evaluations=78，confirmation role evaluations=30，总 aggregate requests=108，logical timed-run references=294。无共享最坏78个物理 state、294 timed runs；共同 seed 的严格共享保证最坏72个物理 state、216 search +48 confirmation=264 timed runs（timed window 15.4h，含10s warmup 16.13h，尚不含启动/drain/shutdown）。进一步 exact overlap 只降低物理测量。', '',
        'ROUTING_REVISION_PENDING = true。Held-out 仅验证 neighborhood + candidate-selection revision，不能声称 routing 已解决。只有3个独立 mixture；其两档强度相关。任务数据/质量评估样本仍沿用已有冻结来源，不能声称 unseen task data 或第二硬件泛化。', '',
        f"Blockers: {blockers}", '', '本阶段到此停止，没有开始 held-out campaign。']
    summary = OUT / 'module8e2_summary.md'
    content = '\n'.join(lines) + '\n'
    if summary.exists(): assert summary.read_text() == content
    else: summary.write_text(content)
    Path(str(summary) + '.sha256').write_text(sha(summary) + '\n')
    artifacts = {p.name: ref(p) for p in sorted(OUT.iterdir()) if p.is_file()
                 and not p.name.endswith('.sha256') and p.name != 'artifact_manifest.json'}
    freeze('artifact_manifest.json', {'module': '8E-2', 'artifacts': artifacts,
        'generator_source': ref(Path(__file__)), 'all_outputs_immutable': True, 'GPU_runs': 0,
        'heldout_runtime_results_read': 0})
    print(json.dumps({'HELDOUT_GPU_CAMPAIGN_READY': not blockers, 'blockers': blockers,
                      'physical_states_worst_case': physical_states,
                      'timed_runs_worst_case': search_runs + confirm_runs, 'GPU_runs': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['preregister', 'generate', 'sanity', 'finish'])
    stage = parser.parse_args().stage
    {'preregister': preregister, 'generate': generate, 'sanity': sanity, 'finish': finish}[stage]()
