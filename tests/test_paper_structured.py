"""Bounded CPU counterexamples for structured paper/FIFO optimization.

Fixtures are analytical software checks; none is GPU accuracy evidence.
"""

import copy
from dataclasses import replace
import unittest

from gsdeploy.continuous import (Budget, ContinuousRoutingEvaluator,
    ImmutableContinuousProblem, SessionPredictorOracle)
from gsdeploy.continuous.contracts import TraceSpec
from gsdeploy.continuous.paper_bounds import PaperDeploymentBounds
from gsdeploy.continuous.paper_structured import PaperStructuredOptimizer, CapacityAwareTransferSearch
from gsdeploy.continuous.refinement import FixedDeploymentRoutingEvaluator
from gsdeploy.forecasting.observed_state import (new_observed_state,
    advance_observed_state, feedback_event)
from gsdeploy.forecasting.profiles import context_key
from gsdeploy.paper_forecast import MODE, PaperFifoSession, canonical
from gsdeploy.paper_capture import CAPTURE_SCHEMA, normalize_capture
from gsdeploy.paper_training import fit_config
from gsdeploy.profiled_resource_search import ProfiledResourceConfigurationSearch
from gsdeploy.resources import RESOURCE_SCHEMA, identity
from gsdeploy.system import DeploymentSystem
from tests.test_paper_completion import config, training


def predict(session,c,x,r,**kwargs):
    context=DeploymentSystem(c).context();context.update(kwargs)
    return session.evaluate(x,r,c.workload,c.gpu_memory_mib,c.latency_reference_ms,
                            c.weights,**context)


def multitask_config():
    """A single G FIFO serving simultaneous text/image, with READY delays."""
    c=config();data=c.catalog
    data['models']={'g':data['models']['g']}
    data['capacity_factors']={'g':data['capacity_factors']['g']}
    data['memory_table_mib']={'g':data['memory_table_mib']['g']}
    data['quality']['paths']={'g':data['quality']['paths']['g']}
    model=data['models']['g']
    model['supported_tasks']=model['executable_tasks']=['text','image']
    model['service_time_ms']['image']=2.
    data['quality']['paths']['g']['image']={**data['quality']['paths']['g']['text'],'metric':'accuracy'}
    spec=data[MODE];entry=spec['deployments'][0];spec['deployments']=[entry]
    spec['input_order']['image']=['x']
    entry['capacity_service_ms']['g']['image']=80.
    profile=copy.deepcopy(entry['profiles']['profiles'][0]);profile.update(id='g-image',task='image')
    entry['profiles']['profiles'].append(profile)
    life=copy.deepcopy(entry['lifecycle']['profiles'][0]);life['task']='image'
    entry['lifecycle']['profiles'].append(life)
    members={'g#0':dict(profile_id='g-image',segment_id='gpu')}
    entry['profiles']['contexts'][context_key(members)]=dict(members=members,evidence='CPU fixture',scenarios={
        'mean':dict(progress={'g#0':1.},board_watts=60.,peak_memory_mib=20.)})
    return replace(c,workload={'text':1.,'image':1.},latency_reference_ms={'text':10.,'image':10.})


def resource_candidates():
    """Two declared fixture controls, each with its own execution laws."""
    rows=[]
    for name,mode in [('shared','shared'),('capped','mps_cap')]:
        c=config();bindings={}
        for entry in c.catalog[MODE]['deployments']:
            aliases={f'{m}#{i}' for m,n in entry['deployment'].items() for i in range(n)}
            resource=dict(schema=RESOURCE_SCHEMA,mode=mode,device='0',allocations={
                a:{} if mode=='shared' else dict(active_thread_percentage=50) for a in aliases})
            if mode=='mps_cap':resource['mps_session']=dict(pipe_directory='/tmp/gsdeploy-test-mps-pipe',
                                                         log_directory='/tmp/gsdeploy-test-mps-log')
            entry['resources']=resource
            context=entry['profiles']['execution_context_sha256']+'-'+name
            entry['profiles']['execution_context_sha256']=entry['lifecycle']['execution_context_sha256']=context
            # Explicit test laws, not a cap-to-speed assumption. The test
            # happens to give capped configurations a slower handler.
            if mode=='mps_cap':
                for p in entry['profiles']['profiles']:
                    p['segments'][0]['duration_ms']['mean']*=2
            bindings[canonical(entry['deployment'])]=dict(execution_context_sha256=context,
                resource_configuration_sha256=identity(resource),evidence='CPU analytical fixture only')
        rows.append(dict(id=name,config=c,profile_bindings=bindings))
    return rows


def busy_case():
    c=config();s=PaperFifoSession(c.catalog);entry,engine=s.entries[canonical({'g':1})]
    context=entry['profiles']['execution_context_sha256']
    state=new_observed_state(engine.aliases,context)
    arrivals=[dict(sequence=i+1,request_id=rid,alias='g#0',task='text',pool_id='x',
        arrival_ns=at,observed_time_ns=at) for i,(rid,at) in enumerate([('a',0),('b',10_000_000)])]
    event=feedback_event(1,'START',dict(request_id='a',alias='g#0',perf_ns=0),0,context)
    state=advance_observed_state(state,40_000_000,arrivals=arrivals,feedback=[event])
    queues={'g#0':[dict(request_id='a',task='text',pool_id='x',arrival_ns=0,
        forecast_cursor=dict(profile_id='g',segment_index=0,remaining_fraction=.5,
                             observed_at_ns=40_000_000,evidence='CPU explicit observed cursor')),
        dict(request_id='b',task='text',pool_id='x',arrival_ns=10_000_000)]}
    future=[dict(request_id='c',task='text',pool_id='x',arrival_ns=80_000_000)]
    return c,s,state,queues,future


def capture_fixture(entry,index=0):
    normalized=training(entry)[index];rid='request'
    row=dict(request_id=rid,model='g#0',task='text',pool_id='x',split='training',status='completed',
        worker_response=dict(start_perf_ns=0,end_perf_ns=80_000_000),
        response_reader_event_enqueue_time_ns=85_000_000,completion_time_ns=85_000_000,
        execution_events=[dict(request_id=rid,perf_ns=t,received_time_ns=t,event=edge,kind='gpu',segment_id='gpu')
                          for t,edge in [(0,'start'),(80_000_000,'end')]],
        previous_handler_next_request_ready=dict(request_id=rid,perf_ns=100_000_000,received_time_ns=101_000_000,
            clock='host_perf_counter_ns',boundary='previous_handler_next_request_ready',
            scope='after_response_flush_before_next_read',pid=123))
    context=entry['profiles']['execution_context_sha256']
    return dict(schema=CAPTURE_SCHEMA,split='training',kind='cpu_fixture',source=normalized['source'],
        run_id=normalized['run_id'],execution_context_sha256=context,roster_complete=True,resident_aliases=['g#0'],
        resource_configuration_sha256=None,observation_window_ns=normalized['observation_window_ns'],
        resource_observations=normalized['resource_observations'],clock_contract=dict(handler='host_perf_counter_ns',
            runtime='host_perf_counter_ns',worker_next_ready='host_perf_counter_ns',same_host_verified=True,
            source='explicit CPU common-clock fixture'),raw=dict(split='training',kind='cpu_fixture',
                execution_context_sha256=context,admission_enabled=False,request_records=[row]))


class FifoCacheTests(unittest.TestCase):
    def test_mapping_reuse_recomputes_quality_and_rho(self):
        c=config();s=PaperFifoSession(c.catalog);x={'g':1,'s':1}
        a=predict(s,c,x,{'text':{'g':.7,'s':1.3}})
        b=predict(s,c,x,{'text':{'g':.8,'s':1.2}})
        self.assertEqual(a['request_mapping'],b['request_mapping'])
        self.assertEqual(a['forecast'],b['forecast'])
        self.assertNotEqual(a['Q_norm'],b['Q_norm']);self.assertNotEqual(a['rho'],b['rho'])
        self.assertEqual(s.engine_calls,1);self.assertEqual(s.fifo_hits,1)
        a['forecast']['finishes'].clear()
        again=predict(s,c,x,{'text':{'g':.7,'s':1.3}})
        self.assertEqual(len(again['forecast']['finishes']),2)

    def test_same_mapping_cannot_inherit_eligibility(self):
        c=config();c.catalog[MODE]['deployments'][-1]['capacity_service_ms']['g']['text']=1100.
        s=PaperFifoSession(c.catalog);x={'g':1,'s':1}
        a=predict(s,c,x,{'text':{'g':.7,'s':1.3}})
        b=predict(s,c,x,{'text':{'g':.8,'s':1.2}})
        self.assertTrue(a['eligible']);self.assertFalse(b['eligible'])
        self.assertEqual(s.engine_calls,1);self.assertEqual(s.fifo_hits,1)

    def test_mapping_residents_and_window_are_in_cache_key(self):
        c=config();s=PaperFifoSession(c.catalog)
        for x,r,kw in [({'g':1,'s':1},{'text':{'g':.7,'s':1.3}},{}),
                       ({'g':1,'s':1},{'text':{'g':1.5,'s':.5}},{}),
                       ({'g':1},{'text':{'g':2.}},{}),
                       ({'g':1},{'text':{'g':2.}},{'horizon':1.1})]:
            predict(s,c,x,r,**kw)
        self.assertEqual(s.engine_calls,4);self.assertEqual(s.fifo_hits,0)

    def test_disabled_and_bounded_fifo_cache(self):
        for field in ('fifo_cache_size','fifo_cache_max_bytes'):
            c=config();c.catalog[MODE][field]=0;s=PaperFifoSession(c.catalog)
            for g in (.7,.8):predict(s,c,{'g':1,'s':1},{'text':{'g':g,'s':2-g}})
            self.assertEqual(s.engine_calls,2);self.assertEqual(s.fifo_cache_bytes,0)
        c=config();c.catalog[MODE]['fifo_cache_size']=1;s=PaperFifoSession(c.catalog)
        for g in (.7,1.5,.8):predict(s,c,{'g':1,'s':1},{'text':{'g':g,'s':2-g}})
        self.assertEqual(s.engine_calls,3);self.assertEqual(len(s.fifo_cache),1)


class SharedFifoBoundTests(unittest.TestCase):
    def test_exact_route_bound_dominates_exhaustive_small_routes(self):
        for arrivals in (((0.,'text'),(.5,'text')),((0.,'text'),(0.,'text'))):
            p=ImmutableContinuousProblem(config(),traces=[TraceSpec('case',42,arrivals)])
            with SessionPredictorOracle(p) as oracle:
                bound=PaperDeploymentBounds(p,oracle)
                for x in ({'g':1},{'s':1},{'g':1,'s':1}):
                    generic=bound.evaluate(x)
                    for ratio in ([0.,.1,.314159,.5,.732051,1.] if len(x)==2 else [None]):
                        route={'text':{m:2*(ratio if m=='s' else 1-ratio) for m in x}} if ratio is not None else {'text':{next(iter(x)):2.}}
                        proof=bound.evaluate(x,routing=route)
                        pred=oracle.predict(p,x,route,p.train_traces[0],Budget(5,4,4))
                        self.assertTrue(pred['eligible'])
                        self.assertGreaterEqual(proof['upper_bound']+1e-9,pred['U'])
                        self.assertLessEqual(proof['upper_bound'],generic['upper_bound']+1e-8)
                        floors=proof['trace_floors'][0]['request_floors']
                        for floor,row in zip(floors,pred['request_mapping']):
                            self.assertEqual(floor['alias'],row['alias'])
                            self.assertLessEqual(floor['finish_floor_ns'],pred['forecast']['finishes'][row['request_id']]['finish_ns'])

    def test_one_fifo_across_tasks_and_ready_not_receipt_serializes(self):
        c=multitask_config();trace=TraceSpec('two-tasks',42,((0.,'image'),(0.,'text')))
        p=ImmutableContinuousProblem(c,traces=[trace]);route={'image':{'g':1.},'text':{'g':1.}}
        with SessionPredictorOracle(p) as oracle:
            proof=PaperDeploymentBounds(p,oracle).evaluate({'g':1},routing=route)
            pred=oracle.predict(p,{'g':1},route,trace,Budget())
            floors=proof['trace_floors'][0]['request_floors']
            self.assertTrue(pred['eligible'])
            self.assertAlmostEqual(floors[0]['finish_floor_ns']/1e6,85.,places=3)
            self.assertAlmostEqual(floors[1]['finish_floor_ns']/1e6,185.,places=3)
            self.assertGreaterEqual(proof['upper_bound'],pred['U'])

    def test_drain_proof_and_missing_contexts_do_not_invent_evidence(self):
        c=replace(multitask_config(),horizon_seconds=.1,drain_seconds=0.)
        p=ImmutableContinuousProblem(c,traces=[TraceSpec('burst',42,((0.,'image'),(0.,'text')))])
        route={'image':{'g':1.},'text':{'g':1.}}
        with SessionPredictorOracle(p) as oracle:
            proof=PaperDeploymentBounds(p,oracle).evaluate({'g':1},routing=route)
            self.assertTrue(proof['safe_pruned']);self.assertEqual(proof['feasibility'],'INFEASIBLE')
        c=config();doc=c.catalog[MODE]['deployments'][-1]['profiles']
        doc['contexts']={k:v for k,v in doc['contexts'].items() if not v['members']}
        p=ImmutableContinuousProblem(c)
        with SessionPredictorOracle(p) as oracle:
            proof=PaperDeploymentBounds(p,oracle).evaluate({'g':1,'s':1},routing={'text':{'g':1.,'s':1.}})
            self.assertFalse(proof['safe_pruned']);self.assertNotEqual(proof['feasibility'],'INFEASIBLE')
            pred=oracle.session.predict({'g':1,'s':1},{'text':{'g':1.,'s':1.}})
            self.assertTrue(pred['performance_unknown'])

    def test_bounded_and_milp_bounds_match_exhaustive_and_save_calls(self):
        c=replace(config(),gpu_memory_mib=30.);p=ImmutableContinuousProblem(c);values=[]
        for inner in ('enumeration','bounded','milp_bounds'):
            with SessionPredictorOracle(p) as oracle:
                e=ContinuousRoutingEvaluator(p,oracle,inner=inner,budget=Budget(15,20,10))
                raw=e.evaluate({'text':{'g':1.,'s':0.}})
                values.append((raw['best_feasible_utility'],e.budget.counts['predictions']))
                self.assertTrue(raw['certified'],(inner,raw['status'],raw['termination_reason']))
                if inner=='milp_bounds':
                    self.assertTrue(raw['shared_fifo_bounds_enabled'])
                    self.assertTrue(any(r.get('pruned') for r in raw['milp_rounds']))
        self.assertEqual(values[0][0],values[1][0]);self.assertEqual(values[0][0],values[2][0])
        self.assertLess(values[1][1],values[0][1]);self.assertLess(values[2][1],values[0][1])


class StructuredSearchTests(unittest.TestCase):
    def test_unknown_interior_does_not_abort_endpoint_search_or_unload(self):
        c=config();doc=c.catalog[MODE]['deployments'][-1]['profiles']
        doc['contexts']={k:v for k,v in doc['contexts'].items() if len(v['members'])<2}
        p=ImmutableContinuousProblem(c,traces=[TraceSpec('burst',42,((0.,'text'),(0.,'text')))])
        with SessionPredictorOracle(p) as oracle:
            entry=oracle.session.module('paper_forecast').session(oracle._config.catalog).entries[canonical({'g':1,'s':1})][0]
            e=FixedDeploymentRoutingEvaluator(p,oracle,{'g':1,'s':1},Budget(5,32,24))
            raw=CapacityAwareTransferSearch(e,entry).solve({'text':{'g':.5,'s':.5}})
            self.assertGreater(raw['unknown_routes'],0);self.assertIsNotNone(raw['winner'])
            self.assertEqual(raw['winner']['deployment'],{'g':1,'s':1})
            self.assertEqual(raw['winner']['prediction']['traces'][0]['prediction']['safe_memory_mib'],40.)

    def test_single_joint_seed_no_later_deployment_milp_and_retained_incumbent(self):
        raw=PaperStructuredOptimizer(config(),seconds=15,max_routes=24,max_predictions=40).solve()
        self.assertIsNotNone(raw['winner']);self.assertIsNotNone(raw['initial_winner'],raw['initial_joint_result']['status'])
        self.assertGreaterEqual(raw['winner']['utility'],raw['initial_winner']['utility'])
        self.assertEqual(raw['deployment_milp_calls_after_seed'],0)
        self.assertTrue(all(s['counts']['milp']==0 for s in raw['stages']))
        self.assertLessEqual(raw['budget']['outer_evaluations'],24)
        self.assertLessEqual(raw['budget']['full_prediction_calls'],40)
        self.assertEqual(raw['deployment_domain']['profiled_task_covers'],3)
        self.assertEqual(raw['deployment_domain']['unsupported_task_covers'],0)
        self.assertFalse(raw['global_optimality']);self.assertFalse(raw['final_gain_eligible'])

    def test_no_joint_path_and_truncation_are_explicit(self):
        c=config();c.catalog[MODE]['deployments']=c.catalog[MODE]['deployments'][:2]
        raw=PaperStructuredOptimizer(c,seconds=5,max_routes=8,max_predictions=16,
            joint_seed=False,max_candidates=1).solve()
        self.assertEqual(raw['budget']['milp_solve_calls'],0)
        self.assertEqual(raw['deployment_domain']['unsupported_task_covers'],1)
        self.assertEqual(raw['deployment_domain']['candidate_limit_truncated'],1)
        self.assertEqual(raw['deployment_domain']['unvisited_profiled'],1)
        self.assertIsNotNone(raw['winner'])

    def test_tiny_time_limit_keeps_declared_domain_and_uncertified_result(self):
        raw=PaperStructuredOptimizer(config(),seconds=1e-8,max_routes=1,max_predictions=1).solve()
        self.assertEqual(raw['deployment_domain']['profiled_task_covers'],3)
        self.assertEqual(raw['deployment_domain']['unsupported_task_covers'],0)
        self.assertEqual(raw['deployment_domain']['bounds_unprocessed'],3)
        self.assertEqual(raw['status'],'TIME_LIMIT');self.assertFalse(raw['global_optimality'])

    def test_de_lp_budget_exit_still_preserves_best_and_no_unload(self):
        raw=PaperStructuredOptimizer(config(),seconds=5,max_routes=3,max_predictions=8,
            method='de',joint_seed=False).solve()
        self.assertIsNotNone(raw['winner']);self.assertEqual(raw['budget']['milp_solve_calls'],0)
        self.assertLessEqual(raw['budget']['outer_evaluations'],3)
        for stage in raw['stages']:
            for h in stage['history']:
                self.assertEqual(h['deployment'],stage['deployment'])
                for shares in h['routing_shares'].values():self.assertAlmostEqual(sum(shares.values()),1.)


class ResourceDecisionTests(unittest.TestCase):
    def test_selection_uses_complete_u_and_launch_binding_preserves_commands(self):
        raw=ProfiledResourceConfigurationSearch(resource_candidates(),seconds=8,max_routes=24,
            max_predictions=40,joint_seed=False).solve()
        self.assertEqual(raw['winner']['configuration_id'],'shared')
        self.assertEqual(raw['evaluated_configurations'],2)
        self.assertFalse(raw['execution_packet']['runnable_registry'])
        self.assertFalse(raw['effective_isolation_verified'])
        original={a:dict(command=['python','actual_worker.py'],env={'TASK_TOKEN':'kept'})
                  for a in raw['execution_packet']['resources']['allocations']}
        bound=ProfiledResourceConfigurationSearch.prepare_launch_registry(raw,original)
        for alias in original:
            self.assertEqual(bound[alias]['command'],original[alias]['command'])
            self.assertEqual(bound[alias]['env']['TASK_TOKEN'],'kept')
        self.assertNotIn('resource_binding',next(iter(original.values())))

    def test_rejects_objective_rho_and_profile_resource_identity_changes(self):
        for corrupt in ('weights','capacity','binding','context_reuse'):
            rows=resource_candidates();c=rows[1]['config']
            if corrupt=='weights':rows[1]['config']=replace(c,latency_reference_ms={'text':11.})
            elif corrupt=='capacity':c.catalog[MODE]['deployments'][0]['capacity_service_ms']['g']['text']=79.
            elif corrupt=='binding':next(iter(rows[1]['profile_bindings'].values()))['resource_configuration_sha256']='bad'
            else:
                for entry,old in zip(c.catalog[MODE]['deployments'],rows[0]['config'].catalog[MODE]['deployments']):
                    context=old['profiles']['execution_context_sha256']
                    entry['profiles']['execution_context_sha256']=entry['lifecycle']['execution_context_sha256']=context
                    rows[1]['profile_bindings'][canonical(entry['deployment'])]['execution_context_sha256']=context
            with self.subTest(corrupt=corrupt),self.assertRaises(ValueError):
                ProfiledResourceConfigurationSearch(rows)

    def test_mps_selection_keeps_platform_verification_blocker(self):
        raw=ProfiledResourceConfigurationSearch(resource_candidates()[1:],seconds=5,max_routes=12,
            max_predictions=24,joint_seed=False).solve()
        self.assertIn('PLATFORM_RESOURCE_CONTROL_NOT_VERIFIED',raw['execution_packet']['launch_blockers'])
        self.assertFalse(raw['effective_isolation_verified'])


class BusyContinuationTests(unittest.TestCase):
    def test_busy_cursor_queue_and_future_use_same_provider_and_ready_lifecycle(self):
        c,s,state,queues,future=busy_case()
        result=s.forecast_continuation({'g':1},queues,state,now_ns=40_000_000,deadline_ns=1_000_000_000,
            drain_deadline_ns=2_000_000_000,future_arrivals=future,routing={'text':{'g':2.}})
        self.assertEqual(result['status'],'FORECAST_COMPLETE');self.assertIsNone(result['U'])
        finishes=result['forecast']['finishes']
        for rid,finish in [('a',85_000_000),('b',185_000_000),('c',285_000_000)]:
            self.assertEqual(finishes[rid]['finish_ns'],finish)

    def test_missing_progress_or_pending_member_is_unknown(self):
        for corrupt in ('cursor','cohort'):
            c,s,state,queues,future=busy_case()
            if corrupt=='cursor':queues['g#0'][0].pop('forecast_cursor')
            else:queues['g#0'].pop()
            raw=s.forecast_continuation({'g':1},queues,state,now_ns=40_000_000,deadline_ns=1_000_000_000,
                drain_deadline_ns=2_000_000_000)
            self.assertEqual(raw['status'],'UNKNOWN');self.assertIsNone(raw['U'])

    def test_stale_observer_and_undeployed_future_route_rejected(self):
        c,s,state,queues,future=busy_case()
        for now,route in [(41_000_000,{'text':{'g':2.}}),(40_000_000,{'text':{'s':2.}})]:
            with self.assertRaises(ValueError):
                s.forecast_continuation({'g':1},queues,state,now_ns=now,deadline_ns=1_000_000_000,
                    drain_deadline_ns=2_000_000_000,future_arrivals=future,routing=route)


class CaptureAdapterTests(unittest.TestCase):
    def test_native_records_normalize_and_fit_same_full_fifo_laws(self):
        c=config();entry=c.catalog[MODE]['deployments'][0]
        captures=[capture_fixture(entry,i) for i in range(2)];before=copy.deepcopy(captures)
        normalized=[normalize_capture(row,entry) for row in captures]
        fitted,report=fit_config(c,{canonical(entry['deployment']):normalized})
        self.assertEqual(captures,before)
        self.assertEqual(report['deployments'][0]['status'],'FITTED_DECLARED_CONTEXTS_ONLY')
        from gsdeploy.config import SystemConfig
        f=SystemConfig(**fitted);s=PaperFifoSession(f.catalog)
        self.assertAlmostEqual(predict(s,f,{'g':1},{'text':{'g':2.}})['U'],-3.0157777777777777)

    def test_rejects_missing_gap_clock_roster_and_resource_mismatch(self):
        c=config();entry=c.catalog[MODE]['deployments'][0]
        for corrupt in ('missing','gap','clock','roster','resource','split','order'):
            capture=capture_fixture(entry);row=capture['raw']['request_records'][0]
            if corrupt=='missing':row.pop('execution_events')
            elif corrupt=='gap':row['execution_events'][0]['perf_ns']=1
            elif corrupt=='clock':capture['clock_contract']['same_host_verified']=False
            elif corrupt=='roster':capture['resident_aliases'].append('s#0')
            elif corrupt=='resource':capture['resource_configuration_sha256']='different'
            elif corrupt=='split':row['split']='validation'
            else:row['execution_events'].reverse()
            with self.subTest(corrupt=corrupt),self.assertRaises(ValueError):normalize_capture(capture,entry)

    def test_missing_ready_remains_partial_not_replaced_with_end(self):
        c=config();entry=c.catalog[MODE]['deployments'][0];docs=[]
        for i in range(2):
            capture=capture_fixture(entry,i)
            capture['raw']['request_records'][0].pop('previous_handler_next_request_ready')
            docs.append(normalize_capture(capture,entry))
        raw,report=fit_config(c,{canonical(entry['deployment']):docs})
        self.assertEqual(report['deployments'][0]['status'],'PARTIAL_TRAINING_LAWS')
        self.assertIsNone(raw['catalog'][MODE]['deployments'][0]['lifecycle']['profiles'][0]['scenarios']['mean']['worker_next_ready_ms'])


if __name__=='__main__':unittest.main()
