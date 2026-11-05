"""Tiny exhaustive checks for new bounds, exact columns and training plumbing."""

import copy
from dataclasses import replace
import itertools
from pathlib import Path
import unittest

from gsdeploy.config import SystemConfig
from gsdeploy.continuous import Budget, ImmutableContinuousProblem, SessionPredictorOracle
from gsdeploy.continuous.contracts import TraceSpec
from gsdeploy.continuous.candidate_bounds import ContinuousDeploymentBounds
from gsdeploy.continuous.multi_candidate import JointDeploymentCandidates, MultiCandidateContinuousRefinement
from gsdeploy.continuous.adaptive_candidate import AdaptiveCandidateRefinementOptimizer
from gsdeploy.forecasting.lifecycle import OBSERVATION_SCHEMA
from gsdeploy.forecasting.profiles import ProfileProvider, context_key
from gsdeploy.forecasting.context_law import SCHEMA as LAW_SCHEMA
from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.paper_training import OBS_SCHEMA, export_segments, fit_config
from gsdeploy.paper_forecast import MODE, canonical

ROOT=Path(__file__).resolve().parents[1]


def config():
    return SystemConfig.load(ROOT/'configs/paper_fifo_cpu_fixture.json')


def training(entry):
    """Explicit analytical test records, never GPU observations."""
    context=entry['profiles']['execution_context_sha256']
    docs=[]
    for i in range(2):
        source=f'cpu-test-run-{i}'
        life=dict(schema=OBSERVATION_SCHEMA,id=source+':request',split='training',kind='cpu_fixture',
            execution_context_sha256=context,family='g',task='text',pool_id='x',
            permit_scope='not_applicable',clock='host_perf_counter_ns',source=source,
            timestamps=dict(handler_end_ns=80_000_000,parent_receipt_ns=85_000_000,
                            permit_release_ns=None,worker_next_ready_ns=100_000_000),
            delays=dict(parent_receipt_ms=5.,permit_release_ms=None,worker_next_ready_ms=20.))
        docs.append(dict(schema=OBS_SCHEMA,split='training',kind='cpu_fixture',source=source,run_id=source,
            execution_context_sha256=context,clock='host_perf_counter_ns',same_host_verified=True,
            observation_window_ns=[0,110_000_000],
            roster_complete=True,requests=[dict(request_id='request',alias='g#0',profile_id='g',
                segments=[dict(segment_id='gpu',start_ns=0,end_ns=80_000_000)])],
            resource_observations=[dict(id=source+':idle',members={},board_watts=20.,peak_memory_mib=20.,
                    start_ns=100_000_000,end_ns=110_000_000,
                    scope='whole_board_and_all_resident_workers'),
                dict(id=source+':active',members={'g#0':dict(profile_id='g',segment_id='gpu')},
                    start_ns=0,end_ns=80_000_000,
                    board_watts=60.,peak_memory_mib=20.,scope='whole_board_and_all_resident_workers')],
            lifecycle_observations=[life]))
    return docs


class PaperBoundsTests(unittest.TestCase):
    def test_bound_dominates_complete_predictions_including_overlap_and_endpoints(self):
        p=ImmutableContinuousProblem(config())
        with SessionPredictorOracle(p) as oracle:
            bounder=ContinuousDeploymentBounds(p,oracle)
            self.assertTrue(bounder.available)
            for deployment in ({'g':1},{'s':1},{'g':1,'s':1}):
                proof=bounder.evaluate(deployment)
                self.assertIsNotNone(proof['upper_bound'])
                for ratio in ([0.,.1,.314159,.5,.9,1.] if len(deployment)>1 else [None]):
                    route={'text':{m:2.*(ratio if m=='s' else 1-ratio) for m in deployment}} if ratio is not None else {'text':{next(iter(deployment)):2.}}
                    pred=oracle.session.predict(deployment,route)
                    self.assertTrue(pred['eligible'])
                    self.assertGreaterEqual(proof['upper_bound']+1e-10,pred['U'])
                    self.assertFalse(proof['physical_bound'])

    def test_unknown_contexts_are_not_infeasibility_proofs_and_idle_memory_is_kept(self):
        c=config();entry=c.catalog[MODE]['deployments'][-1]
        entry['profiles']['contexts']={k:v for k,v in entry['profiles']['contexts'].items() if len(v['members'])<2}
        p=ImmutableContinuousProblem(c)
        with SessionPredictorOracle(p) as oracle:
            proof=ContinuousDeploymentBounds(p,oracle).evaluate({'g':1,'s':1})
            self.assertFalse(proof['safe_pruned'])
            self.assertEqual(proof['memory_mib'],40.)

    def test_true_memory_and_rho_necessities_can_prune(self):
        c=config();c=replace(c,gpu_memory_mib=30.)
        p=ImmutableContinuousProblem(c)
        with SessionPredictorOracle(p) as oracle:
            proof=ContinuousDeploymentBounds(p,oracle).evaluate({'g':1,'s':1})
            self.assertEqual(proof['reason'],'ALL_CONTEXT_MEMORY_FLOOR')
        c=config()
        c.catalog[MODE]['deployments'][0]['capacity_service_ms']['g']['text']=500.
        p=ImmutableContinuousProblem(c)
        with SessionPredictorOracle(p) as oracle:
            proof=ContinuousDeploymentBounds(p,oracle).evaluate({'g':1})
            self.assertEqual(proof['reason'],'ORIGINAL_RHO_SIMPLEX_INFEASIBLE')


class ColumnReuseTests(unittest.TestCase):
    def test_residual_candidates_reuse_columns_but_not_feedback(self):
        p=ImmutableContinuousProblem(config())
        outputs=[]
        for enabled in (False,True):
            with SessionPredictorOracle(p) as oracle:
                ex=JointDeploymentCandidates(p,oracle,Budget(8,100,64),cache_columns=enabled)
                cs=ex.generate(3)
                outputs.append([(c['deployment'],c['grid_routing'],c['grid_utility']) for c in cs])
                self.assertEqual(len({canonical(c['deployment']) for c in cs}),3)
                if enabled:
                    self.assertEqual(ex.column_cache.hits,2)
                    self.assertEqual(ex.column_cache.misses,1)
                    self.assertEqual(len(ex.solves[0]['excluded_deployments']),0)
                    self.assertEqual(len(ex.solves[2]['excluded_deployments']),2)
        self.assertEqual(outputs[0],outputs[1])

    def test_multi_and_adaptive_run_new_predictor_without_deployment_resolve_in_refinement(self):
        for cls in (MultiCandidateContinuousRefinement,AdaptiveCandidateRefinementOptimizer):
            raw=cls(config(),k=3,seconds=5,max_predictions=64,max_routes=32).solve()
            self.assertIsNotNone(raw['winner'])
            self.assertAlmostEqual(raw['winner']['utility'],-3.0157777777777777)


class TrainingTests(unittest.TestCase):
    def test_training_pipeline_reuses_work_fit_and_lifecycle_without_qualification(self):
        c=config();entry=c.catalog[MODE]['deployments'][0]
        raw,report=fit_config(c,{canonical(entry['deployment']):training(entry)})
        self.assertEqual(report['deployments'][0]['status'],'FITTED_DECLARED_CONTEXTS_ONLY')
        self.assertFalse(report['validation_labels_read'])
        fitted=SystemConfig(**raw)
        p=ImmutableContinuousProblem(fitted,replica_options={'g':[1],'s':[]})
        with SessionPredictorOracle(p) as oracle:
            value=oracle.session.predict({'g':1},{'text':{'g':2.}})
            self.assertTrue(value['eligible'])
            self.assertFalse(value['independent_accuracy_qualified'])

    def test_validation_or_unknown_background_cannot_enter_training(self):
        c=config();entry=c.catalog[MODE]['deployments'][0]
        for field,value in [('split','validation'),('roster_complete',False),('same_host_verified',False)]:
            d=training(entry)[0];d[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                export_segments(d,entry['profiles'],{'g#0':'g'})

    def test_training_cannot_silently_ignore_unknown_deployment(self):
        c=config();entry=c.catalog[MODE]['deployments'][0]
        with self.assertRaises(ValueError):
            fit_config(c,{canonical(entry['deployment']):training(entry),'unknown':training(entry)})

    def test_owned_permit_release_is_mandatory_for_complete_fit(self):
        c=config();entry=c.catalog[MODE]['deployments'][0]
        entry['lifecycle']['profiles'][0]['permit_scope']='request_admission'
        docs=training(entry)
        for d in docs:d['lifecycle_observations'][0]['permit_scope']='request_admission'
        raw,report=fit_config(c,{canonical(entry['deployment']):docs})
        self.assertEqual(report['deployments'][0]['status'],'PARTIAL_TRAINING_LAWS')

    def test_missing_ready_or_resources_is_not_filled_from_handler_end(self):
        c=config();entry=c.catalog[MODE]['deployments'][0];docs=training(entry)
        for d in docs:
            d['resource_observations']=[]
            d['lifecycle_observations'][0]['timestamps']['worker_next_ready_ns']=None
            d['lifecycle_observations'][0]['delays']['worker_next_ready_ms']=None
        raw,report=fit_config(c,{canonical(entry['deployment']):docs})
        self.assertEqual(report['deployments'][0]['status'],'PARTIAL_TRAINING_LAWS')
        p=ImmutableContinuousProblem(SystemConfig(**raw),replica_options={'g':[1],'s':[]})
        with SessionPredictorOracle(p) as oracle:
            pred=oracle.session.predict({'g':1},{'text':{'g':2.}})
            self.assertIsNone(pred['U'])
            self.assertTrue(pred['performance_unknown'])

    def test_missing_all_lifecycle_and_partial_runs_stay_unknown(self):
        c=config();entry=c.catalog[MODE]['deployments'][0];docs=training(entry)
        docs[1]['requests']=[]
        # An empty recording with no observed work is rejected explicitly.
        with self.assertRaises(ValueError):
            fit_config(c,{canonical(entry['deployment']):docs})
        docs=training(entry)
        for d in docs:
            d['lifecycle_observations']=[]
        raw,report=fit_config(c,{canonical(entry['deployment']):docs})
        self.assertEqual(report['deployments'][0]['status'],'PARTIAL_TRAINING_LAWS')
        self.assertIsNone(raw['catalog'][MODE]['deployments'][0]['lifecycle']['profiles'][0]['scenarios']['mean']['worker_next_ready_ms'])

    def test_resource_window_and_lifecycle_must_match_stage_clock(self):
        c=config();entry=c.catalog[MODE]['deployments'][0]
        for corrupt in ('resource','handler_end'):
            docs=training(entry)
            if corrupt=='resource':
                docs[0]['resource_observations'][0]['start_ns']=70_000_000
            else:
                docs[0]['lifecycle_observations'][0]['timestamps']['handler_end_ns']-=1
            with self.subTest(corrupt=corrupt),self.assertRaises(ValueError):
                fit_config(c,{canonical(entry['deployment']):docs})

    def test_duplicate_observations_in_one_run_are_not_independent(self):
        c=config();entry=c.catalog[MODE]['deployments'][0];docs=training(entry)
        alternate=copy.deepcopy(entry['profiles']['profiles'][0]);alternate.update(id='g-other',pool_id='y')
        entry['profiles']['profiles'].append(alternate)
        life=copy.deepcopy(entry['lifecycle']['profiles'][0]);life['pool_id']='y'
        entry['lifecycle']['profiles'].append(life)
        second=copy.deepcopy(docs[0]['requests'][0]);second['request_id']='request2'
        for segment in second['segments']:
            segment['start_ns']+=120_000_000;segment['end_ns']+=120_000_000
        docs[0]['requests'].append(second);docs[0]['observation_window_ns'][1]=220_000_000
        later=copy.deepcopy(docs[0]['lifecycle_observations'][0]);later['id']=docs[0]['run_id']+':request2'
        later['timestamps']={k:v+120_000_000 if v is not None else None for k,v in later['timestamps'].items()}
        docs[0]['lifecycle_observations'].append(later)
        docs[1]['requests'][0]['profile_id']='g-other'
        docs[1]['resource_observations'][1]['members']['g#0']['profile_id']='g-other'
        docs[1]['lifecycle_observations'][0]['pool_id']='y'
        raw,report=fit_config(c,{canonical(entry['deployment']):docs})
        rejected=report['deployments'][0]['rejected_progress_groups']
        self.assertTrue(any(g.get('reason')=='INSUFFICIENT_INDEPENDENT_PROGRESS_RUNS' for g in rejected))
        means=raw['catalog'][MODE]['deployments'][0]['lifecycle']['profiles']
        self.assertTrue(all(p['scenarios']['mean']['worker_next_ready_ms'] is None for p in means))


def interpolation_config():
    """Unseen mid/mid context bracketed by four analytical donor contexts."""
    c=config();doc=c.catalog[MODE]['deployments'][-1]['profiles']
    originals=copy.deepcopy(doc['profiles'])
    doc['profiles']=[];doc['contexts']={}
    for p in originals:
        for suffix,demand in (('low',.1),('mid',.3),('high',.5),('outside',.9)):
            q=copy.deepcopy(p)
            q['id']=p['id']+suffix
            q['pool_id']='x' if suffix=='mid' else suffix
            q['segments'][0]['demand']=dict(sm=demand,dram=demand,l2=demand)
            doc['profiles'].append(q)
    for g,s in [('mid','low'),('mid','high'),('low','mid'),('high','mid'),('mid',None),(None,'mid'),(None,None)]:
        members={f'{m}#0':dict(profile_id=m+suffix,segment_id='gpu') for m,suffix in [('g',g),('s',s)] if suffix}
        high='high' in (g,s)
        doc['contexts'][context_key(members)]=dict(members=members,evidence='cpu-analytical-donors',scenarios={
            'mean':dict(progress={a:.5 if high else 1. for a in members},
                        board_watts=100. if high else 60. if members else 20.,peak_memory_mib=40.)})
    doc['context_interpolation']=dict(schema=LAW_SCHEMA,method='convex_observed_resource_features',scenarios=['mean'])
    return c


class ContextInterpolationTests(unittest.TestCase):
    def provider(self,c):
        d=c.catalog[MODE]['deployments'][-1]['profiles']
        return ProfileProvider(d,d['execution_context_sha256'],allow_cpu_fixture=True)

    def active(self,g='mid',s='mid'):
        return {m+'#0':dict(profile_id=m+v,segment_id='gpu',task='text',pool_id='x' if v=='mid' else v)
                for m,v in [('g',g),('s',s)]}

    def test_unseen_context_changes_progress_and_keeps_resident_resources(self):
        provider=self.provider(interpolation_config())
        value,evidence=provider._context(self.active(),{'g#0':{'family':'g'},'s#0':{'family':'s'}},'mean')
        self.assertAlmostEqual(value['progress']['g#0'],.75)
        self.assertAlmostEqual(value['progress']['s#0'],.75)
        self.assertAlmostEqual(value['board_watts'],80.)
        self.assertEqual(value['peak_memory_mib'],40.)
        self.assertIn('"physical_bound": false',evidence)

    def test_no_extrapolation_missing_counters_or_scenario_fill(self):
        for corrupt in ('outside','counters','scenario','exact_missing'):
            c=interpolation_config();doc=c.catalog[MODE]['deployments'][-1]['profiles']
            active=self.active()
            scenario='mean'
            if corrupt=='outside':active=self.active(s='outside')
            if corrupt=='counters':
                next(p for p in doc['profiles'] if p['id']=='smid')['segments'][0]['demand']['sm']=None
            if corrupt=='scenario':scenario='tail'
            if corrupt=='exact_missing':
                members={a:{k:r[k] for k in ('profile_id','segment_id')} for a,r in active.items()}
                doc['contexts'][context_key(members)]=dict(members=members,evidence='explicit-partial',scenarios={
                    'other':dict(progress={a:1. for a in active},board_watts=None,peak_memory_mib=None)})
            with self.subTest(corrupt=corrupt),self.assertRaises(UnknownEvidence):
                self.provider(c)._context(active,{'g#0':{'family':'g'},'s#0':{'family':'s'}},scenario)

    def test_stage_bound_covers_interpolated_full_fifo(self):
        c=interpolation_config();p=ImmutableContinuousProblem(c,
            traces=[TraceSpec('burst',11,((0.,'text'),(0.,'text')))])
        with SessionPredictorOracle(p) as oracle:
            proof=ContinuousDeploymentBounds(p,oracle).evaluate({'g':1,'s':1})
            pred=oracle.predict(p,{'g':1,'s':1},{'text':{'g':1.,'s':1.}},p.train_traces[0],Budget())
            self.assertTrue(pred['eligible'])
            self.assertGreaterEqual(proof['upper_bound'],pred['U'])
            self.assertIn('convex_context_law',str(pred['forecast']))


if __name__=='__main__':unittest.main()
