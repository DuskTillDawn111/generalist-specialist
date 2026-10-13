"""Regression and discrete-capacity witness tests; no GPU execution."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from deployment_core import Hardware, Workload, MilpDeploymentSeeder, Deployment
from deployment_core.milp_seeder import validate_seed
from deployment_core.replica_calibration import ReplicaCalibration, ReplicaPlanningPolicy, load_replica_calibration
from deployment_core.routing_solver import solve_capacity_lp
from deployment_search import IntegerGuidedDeploymentSearch, SearchConfig, solve_canonical_routing
from deployment_search.integer_foundation import model, binary_compatibility
from candidate_pool_service_demand.dry_run import load_completed_catalog


def fixture(catalog, real, g2=None, memory2=None):
    names = set(catalog)
    factors = {m: {0: 0., 1: 1., 2: (g2 or {}).get(m, real.capacity_factors[m][2])} for m in names}
    memory = {m: {0: 0., 1: catalog[m].memory_mib, 2: (memory2 or {}).get(m, 2*catalog[m].memory_mib)} for m in names}
    return ReplicaCalibration(factors, memory, {m: 2 for m in names},
        {m: {'replica_scaling_status': 'declared_synthetic_validation_fixture'} for m in names}, real.source_hashes).validate(catalog)


def competition(real):
    c = {'qwen': model('qwen', 3., image=500.), 'resnet50': model('resnet50', 1., image=1000.)}
    data = fixture(c, real)
    bounds, x, w, h = {m: 2 for m in c}, {m: 1 for m in c}, Workload({'image': 1.2}), Hardware(9.)
    outputs = {}
    for mode in ('linear', 'empirical'):
        result = IntegerGuidedDeploymentSearch().search(w, c, h, SearchConfig(dry_run=True),
            initial_deployment=x, replica_bounds=bounds, capacity_mode=mode,
            memory_mode='empirical', calibration=data)
        ups = sorted((candidate for candidate in result.iterations[0]['candidates'] if candidate.mutation.operation == 'SCALE_UP'), key=lambda candidate: candidate.rank)
        outputs[mode] = {'winner': ups[0].mutation.model, 'z': result.canonical.max_pressure,
            'scale_ups': [{'model': a.mutation.model, 'delta_z': a.delta_z, 'rank': a.rank, 'memory_mib': a.memory_mib} for a in ups],
            'runtime_evaluations': result.runtime_evaluation_count}
    return outputs


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog, cls.hardware, cls.roles, *_ = load_completed_catalog()
        cls.real = load_replica_calibration(cls.catalog)

    def test_full_precision_source_and_no_pending_factors(self):
        data = json.loads(Path(self.real.source_hashes['replica_scaling_metrics.json']['path']).read_text())
        for m in ('qwen','whisper-small','resnet50'):
            pair = next(p for p in data['scaling_comparisons'] if p['model_id'] == m and p['load_level'] == 'high')
            self.assertEqual(self.real.capacity_factors[m][2], pair['paired_ratios']['throughput_gain']['mean'])
            self.assertNotEqual(self.real.capacity_factors[m][2], round(self.real.capacity_factors[m][2], 3))
        for m in set(self.catalog)-{'qwen','whisper-small','resnet50'}:
            self.assertEqual(self.real.evidence_bounds[m], 1)
            self.assertNotIn(2, self.real.capacity_factors[m])
            self.assertEqual(self.real.model_evidence[m]['replica_scaling_status'], 'pending')

    def test_memory_single_anchor_and_observed_double(self):
        p = ReplicaPlanningPolicy(self.catalog, 'empirical', calibration=self.real)
        self.assertEqual(p.memory('qwen',1), self.catalog['qwen'].memory_mib)
        self.assertEqual(p.memory('qwen',2), self.real.model_evidence['qwen']['observed_7c1_double_idle_mib']['mean'])
        self.assertNotEqual(p.memory('qwen',2), 2*self.catalog['qwen'].memory_mib)

    def test_evidence_bounds_do_not_change_physical_bounds(self):
        p = ReplicaPlanningPolicy(self.catalog, 'empirical', calibration=self.real)
        bounds, refs = p.bounds(self.hardware)
        self.assertEqual(refs['resnet50']['R_mem'], 81)
        self.assertEqual((refs['resnet50']['R_evidence'], bounds['resnet50']), (2,2))
        self.assertGreater(refs['paddleocr']['R_mem'], 1)
        self.assertEqual(bounds['paddleocr'], 1)
        bad = dict(bounds, phi4mm=2)
        with self.assertRaisesRegex(ValueError, 'evidence'):
            p.bounds(self.hardware, bad)
        with self.assertRaises(ValueError):
            p.capacity('phi4mm',2)
        with self.assertRaises(ValueError):
            p.memory('resnet50',3)
        with self.assertRaises(ValueError):
            solve_canonical_routing({'phi4mm':2}, Workload({}), self.catalog, capacity_mode='empirical', calibration=self.real)

    def test_case_A_exact_binary_seed_routing_search_serialization(self):
        c = {m:self.catalog[m] for m in ('qwen','resnet50','whisper-small')}
        cal = ReplicaCalibration({m:self.real.capacity_factors[m] for m in c}, {m:self.real.memory_table_mib[m] for m in c}, {m:2 for m in c})
        b = {m:1 for m in c}; w = Workload({'asr': .15,'image': .4,'visual_qa':.2})
        seeds = [MilpDeploymentSeeder().solve(c,w,self.hardware,replica_bounds=b,capacity_mode=mode,calibration=cal) for mode in ('linear','empirical')]
        a,z = [asdict(s) for s in seeds]
        a.pop('solver_time_ms'); z.pop('solver_time_ms'); self.assertEqual(a,z)
        routes = [solve_canonical_routing(seeds[0].deployment,w,c,replica_bounds=b,capacity_mode=mode,calibration=cal) for mode in ('linear','empirical')]
        self.assertEqual(asdict(routes[0]), asdict(routes[1]))
        # x<=1 gives identical fixed-deployment LP even when R=2 is available.
        wide={m:2 for m in c}
        wide_routes=[solve_canonical_routing(seeds[0].deployment,w,c,replica_bounds=wide,capacity_mode=mode,calibration=cal) for mode in ('linear','empirical')]
        self.assertEqual(asdict(wide_routes[0]),asdict(wide_routes[1]))
        results = [IntegerGuidedDeploymentSearch().search(w,c,self.hardware,SearchConfig(dry_run=True),replica_bounds=b,capacity_mode=mode,calibration=cal).to_dict() for mode in ('linear','empirical')]
        for r in results:r['seed'].pop('solver_time_ms')
        self.assertEqual(results[0],results[1])
        self.assertTrue(binary_compatibility()['passed'])

    def test_case_B_qwen_marginal_capacity_and_feasibility(self):
        c = {'qwen': model('qwen',2.,asr=1000.)}; cal=fixture(c,self.real,memory2={'qwen':3.5})
        p=ReplicaPlanningPolicy(c,'empirical',calibration=cal);g=p.capacity('qwen',2)
        self.assertEqual(g-1.,self.real.capacity_factors['qwen'][2]-1.)
        self.assertLess(g-1.,.25)
        seed=MilpDeploymentSeeder().solve(c,Workload({'asr':.9}),Hardware(5.),capacity_mode='empirical',calibration=cal)
        self.assertEqual(seed.deployment,{'qwen':2});self.assertEqual(seed.objective_value,3.5)
        self.assertEqual(validate_seed(seed,c,Workload({'asr':.9}),Hardware(5.),.8,{'qwen':2},planning_policy=p),[])
        self.assertEqual(seed.count_selection,{'qwen':{0:0,1:0,2:1}})
        self.assertAlmostEqual(seed.utilization['qwen'], .9/g)
        with patch('deployment_core.empirical_milp.milp', side_effect=RuntimeError('No solver in validator')):
            self.assertEqual(validate_seed(seed,c,Workload({'asr':.9}),Hardware(5.),.8,{'qwen':2},planning_policy=p),[])
        high=Workload({'asr':1.4})
        self.assertTrue(MilpDeploymentSeeder().solve(c,high,Hardware(5.),binary_mode=False).feasible)
        self.assertFalse(MilpDeploymentSeeder().solve(c,high,Hardware(5.),capacity_mode='empirical',calibration=cal).feasible)
        self.assertIsNone(solve_canonical_routing({'qwen':2},high,c,capacity_mode='empirical',calibration=cal))
        self.assertIsNone(solve_capacity_lp(Deployment({'qwen':2}),c,high,capacity_mode='empirical',calibration=cal))

    def test_case_C_resnet_near_linear_capacity(self):
        c={'resnet50':model('resnet50',1.,image=1000.)};cal=fixture(c,self.real);w=Workload({'image':1.2})
        linear=solve_canonical_routing({'resnet50':2},w,c,replica_bounds={'resnet50':2})
        empirical=solve_canonical_routing({'resnet50':2},w,c,capacity_mode='empirical',calibration=cal)
        self.assertLess(abs(empirical.max_pressure/linear.max_pressure-1),.02)
        self.assertAlmostEqual(empirical.max_pressure,1.2/self.real.capacity_factors['resnet50'][2])

    def test_case_D_competition_changes_order_without_runtime(self):
        out=competition(self.real)
        self.assertEqual(out['linear']['winner'],'qwen');self.assertEqual(out['empirical']['winner'],'resnet50')
        self.assertEqual(out['linear']['runtime_evaluations']+out['empirical']['runtime_evaluations'],0)

    def test_discrete_memory_gates_and_scale_down_actual_saving(self):
        c={'qwen':model('qwen',2.,asr=1000.)};cal=fixture(c,self.real,memory2={'qwen':3.5})
        h,w=Hardware(3.75),Workload({'asr':.9})
        # R_mem=floor(3.75/2)=1 remains the mandatory physical ledger bound.
        self.assertFalse(MilpDeploymentSeeder().solve(c,w,h,capacity_mode='empirical',calibration=cal).feasible)
        result=IntegerGuidedDeploymentSearch().search(Workload({'asr':.2}),c,Hardware(5.),SearchConfig(dry_run=True),
            initial_deployment={'qwen':2},capacity_mode='empirical',calibration=cal)
        down=next(a for a in result.iterations[0]['candidates'] if a.mutation.operation=='SCALE_DOWN')
        self.assertEqual(down.memory_freed_mib,1.5);self.assertEqual(down.memory_mib,2.)

    def test_changed_aggregate_rejected_by_frozen_hash(self):
        evidence=Path(self.real.source_hashes['replica_scaling_metrics.json']['path']).parent
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)
            import shutil
            for name in ('artifact_manifest.json','replica_scaling_metrics.json','memory_scaling.json','planning_surrogate_validation.json','replica_scaling_plan.json','measurement_validation.json'):
                shutil.copy2(evidence/name,target/name)
            data=json.loads((target/'replica_scaling_metrics.json').read_text());data['scaling_comparisons'][0]['load_level']='tampered'
            (target/'replica_scaling_metrics.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                load_replica_calibration(self.catalog,target)

    def test_real_catalog_defaults_empirical_and_no_runtime_call(self):
        evaluator=unittest.mock.Mock();evaluator.evaluate.side_effect=AssertionError('GPU/runtime forbidden')
        result=IntegerGuidedDeploymentSearch(evaluator).search(Workload({'visual_qa':.2}),self.catalog,self.hardware,SearchConfig(dry_run=True),calibration=self.real)
        self.assertEqual(result.capacity_mode,'empirical');self.assertEqual(result.runtime_evaluation_count,0)
        evaluator.evaluate.assert_not_called()


class DiscreteDiagnosticTests(unittest.TestCase):
    def test_complete_direct_neighbors_preserve_guided_pruning(self):
        from .foundation import planning_run, complete_neighbor_diagnostic
        catalog,hardware,roles,records,_=load_completed_catalog()
        cal=load_replica_calibration(catalog)
        row=json.loads(Path('results/deployment_search_multiworkload/benchmark_workloads.json').read_text())['workloads'][0]
        state=dict.fromkeys(catalog,0);state.update(qwen=1,resnet50=1);state['whisper-small']=1
        guided=planning_run(row,catalog,hardware,roles,records,cal,'empirical',initial=state)
        direct=complete_neighbor_diagnostic(guided,catalog,hardware,roles,cal,'empirical')
        self.assertLessEqual(direct['candidate_count'],2*len(catalog))
        for m in ('resnet50','whisper-small'):
            candidate=next(a for a in direct['candidates'] if a['model_id']==m and a['new_replica_count']==2)
            self.assertTrue(candidate['feasibility'])
            self.assertFalse(candidate['included_by_production_pressure_task_guidance'])
            self.assertAlmostEqual(candidate['delta_z'],0.)
        self.assertEqual(direct['runtime_evaluations'],0)

    def test_small_discrete_milp_oracle(self):
        from .foundation import small_discrete_oracle
        catalog,*_=load_completed_catalog()
        result=small_discrete_oracle(load_replica_calibration(catalog))
        self.assertTrue(result['passed'])
        self.assertTrue(all(case['passed'] for case in result['cases']))

if __name__ == '__main__':unittest.main()
