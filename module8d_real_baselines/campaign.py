import argparse, fcntl, json, os, traceback
from pathlib import Path
from unittest.mock import patch
from deployment_core import Workload
from deployment_search import IntegerGuidedDeploymentSearch, SearchConfig
from deployment_search.real_objective import NoiseAwareRuntimeObjective
from deployment_search.runtime_cache import save_json,file_hash
from deployment_search.tests.fixtures import forbid_enumeration
from integer_replica_pilot.evaluator import aggregate, PilotExecutionBlocked
from .common import ROOT,F,C,OUT,read,ref,inputs,prepare,verify_inputs
from .evidence import EvidenceBank,SharedEvidenceEvaluator


def evaluator(directory,workload,bank,owner,budget):
    d=read(F/'baseline_definitions.json');catalog=inputs()[0]
    return SharedEvidenceEvaluator(directory/'search_cache',catalog,C/workload/'runtime_context.json',
        d['measurement_protocol']['settings'],d['measurement_protocol']['measurement_seeds'],budget,
        bank,owner,workload)


def nested_route(state,catalog):
    return {(t,m):state['routing_rps'][t][m] for t in state['routing_rps'] for m in catalog}


def fixed_job(job,bank):
    baseline,name=job['baseline'],job['workload'];owner=f'{baseline}/{name}';directory=OUT/baseline.lower()/name
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'status.json').exists():return
    state=next(s for s in read(F/'baseline_deployments.json')['states'] if s['baseline']==baseline and s['workload']==name)
    ev=evaluator(directory,name,bank,owner,1);catalog=inputs()[0]
    x={m:state['deployment'][m] for m in catalog};workload=Workload(state['lambda_rps']);routing=nested_route(state,catalog)
    # Check frozen explicit route and vector, do not invoke a seeder or optimizer here.
    identity=ev.identity(x,workload,routing)
    try:
        runtime=ev.evaluate(x,workload,routing)
        reps=[bank.confirmation(directory,owner,name,identity,'state',i,set()) for i in (1,2)]
        assert len({r['raw_result_path'] for r in reps})==2
        confirm=aggregate(reps,'independent_confirmation',directory/'confirmation/state_aggregate.json',x)
        save_json(directory/'confirmation/state_aggregate.json',confirm)
        save_json(directory/'result.json',{'baseline':baseline,'workload':name,'state':state,
            'identity':identity,'runtime':runtime,'confirmation':confirm,'measurement_source':ev.journal[0]['measurement_source'],
            'source_paths':runtime.raw_result_paths,'confirmation_sources':reps,'no_search':True})
        save_json(directory/'status.json',{'status':'complete','owner':owner,'search':False})
    except PilotExecutionBlocked as exc:
        preserve_failure(directory,owner,exc)
    print(json.dumps({'event':'fixed_job_complete','owner':owner,'status':read(directory/'status.json')['status']}),flush=True)


def preserve_failure(directory,owner,exc):
    # Resource-infeasible is a measured outcome; do not adjust the deployment.
    manifest=read(OUT/'new_measurement_manifest.json')
    failed=[r for r in manifest['runs'] if r['owner']==owner and not r.get('valid_timed_execution',False)]
    resource=any(r.get('status')=='RUNTIME_RESOURCE_INFEASIBLE' for r in failed)
    status='runtime_resource_infeasible' if resource else 'external_interruption'
    save_json(directory/'status.json',{'status':status,'owner':owner,'reason':str(exc),
        'failed_measurements':failed,'deployment_unchanged':True,'retry_policy':'existing frozen zero retries',
        'remaining_repetitions':'not executed after invalid execution; not silently replaced'})
    if not resource:raise exc


def random_job(job,bank):
    name,seed=job['workload'],job['random_seed'];owner=f'B4-{seed}/{name}'
    directory=OUT/f'b4_seed_{seed}'/name;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'status.json').exists():return
    protocol=read(C/name/'protocol.json');catalog,hardware,roles,cal=inputs()
    workload=Workload(protocol['workload']['arrival_rates_rps']);ev=evaluator(directory,name,bank,owner,6)
    complete=directory/'search_complete.json'
    try:
        if complete.exists():trace=read(complete)
        else:
            with forbid_enumeration() as guards, patch('deployment_search.search.rank_candidates',
                    side_effect=AssertionError('LP ranking forbidden in Random baseline')) as rank:
                result=IntegerGuidedDeploymentSearch(ev,predictor=None,runtime_objective=NoiseAwareRuntimeObjective(1.,.05),
                    selection_policy='random',transition_policy='runtime',random_seed=seed,
                    trace_callback=lambda r:save_json(directory/'search_trace.json',r.to_dict())).search(
                    workload,catalog,hardware,SearchConfig(**protocol['search_config']),roles=roles,
                    capacity_mode='empirical',memory_mode='empirical',calibration=cal)
                assert rank.call_count==0
            trace=result.to_dict();trace['enumeration_calls']=sum(g.call_count for g in guards)
            trace['LP_rank_calls']=0;trace['random_seed']=seed
            frozen=next(s for s in read(F/'baseline_deployments.json')['states'] if s['baseline']=='B4_SEED' and s['workload']==name and s['random_seed']==seed)
            assert all(trace['seed']['deployment'][m]==frozen['deployment'][m] for m in catalog)
            assert trace['runtime_evaluation_count']<=6 and trace['search_iterations']<=3 and trace['enumeration_calls']==0
            first=sorted([c for c in trace['iterations'][0]['candidates'] if c['selected_top_b']],key=lambda c:c['rank'])
            expected=sorted([s for s in read(F/'baseline_deployments.json')['states'] if s['baseline']=='B4_INITIAL_CANDIDATE'
                and s['workload']==name and s['random_seed']==seed],key=lambda s:s['origin']['random_selection_order'])
            assert [c['mutation']['deployment'] for c in first]==[{m:s['deployment'][m] for m in catalog} for s in expected]
            save_json(directory/'search_trace.json',trace);save_json(complete,trace)
        frozen_decision={'search_trace':ref(directory/'search_trace.json'),'seed':trace['seed']['deployment'],
            'final':trace['deployment'],'final_routing':trace['canonical']['routing'],'confirmation_does_not_influence_search':True}
        path=directory/'post_search_freeze.json'
        if path.exists():assert read(path)==frozen_decision
        else:save_json(path,frozen_decision)
        counts={'seed':trace['seed']['deployment'],'final':trace['deployment']}
        route={'seed':trace['iterations'][0]['canonical']['routing'],'final':trace['canonical']['routing']}
        identities={which:ev.identity(counts[which],workload,{tuple(k.split('::')):v for k,v in route[which].items()}) for which in counts}
        reps={'seed':[],'final':[]};used=set()
        for which,index in protocol['confirmation_order']:
            rep=bank.confirmation(directory,owner,name,identities[which],which,index,used)
            assert rep['raw_result_path'] not in used
            used.add(rep['raw_result_path']);reps[which].append(rep)
            save_json(directory/'confirmation_results.json',{'complete':False,'repetitions':reps,'influences_search':False})
        aggregates={which:aggregate(sorted(rr,key=lambda r:r['repetition']),'independent_confirmation',
                       directory/'confirmation'/f'{which}_aggregate.json',counts[which]) for which,rr in reps.items()}
        for which,value in aggregates.items():save_json(directory/'confirmation'/f'{which}_aggregate.json',value)
        save_json(directory/'confirmation_results.json',{'complete':True,'repetitions':reps,'aggregates':aggregates,
            'influences_search':False,'seed_and_final_donor_paths_disjoint':True})
        save_json(directory/'status.json',{'status':'complete','owner':owner,'random_seed':seed,
            'logical_states':trace['runtime_evaluation_count'],'same_frozen_budget_as_Proposed':True})
    except PilotExecutionBlocked as exc:preserve_failure(directory,owner,exc)
    print(json.dumps({'event':'random_job_complete','owner':owner,'status':read(directory/'status.json')['status']}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    for key,value in read(F/'baseline_definitions.json')['measurement_protocol']['execution_environment_overrides'].items():os.environ[key]=value
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=prepare()
        if args.prepare_only:return
        bank=EvidenceBank();bank.persist()
        try:
            for phase,jobs in (('A',manifest['phase_A']),('B',manifest['phase_B'])):
                for job in jobs:
                    verify_inputs()
                    save_json(OUT/'campaign_status.json',{'status':'running','phase':phase,'active_job':job,
                        'new_physical_attempts':len(bank.new_runs),'new_valid_timed_runs':sum(r.get('valid_timed_execution',False) for r in bank.new_runs)})
                    print(json.dumps({'event':'job_start','phase':phase,'job':job}),flush=True)
                    (fixed_job if phase=='A' else random_job)(job,bank)
                save_json(OUT/f'phase_{phase}_complete.json',{'complete':True,'jobs':jobs,'definitions_unchanged':True})
            verify_inputs()
            from .report import report
            report()
            save_json(OUT/'campaign_status.json',{'status':'complete','stopped_after_module':'8D-2',
                'new_valid_timed_runs':sum(r.get('valid_timed_execution',False) for r in bank.new_runs),
                'physical_attempts':len(bank.new_runs),'ablation':False,'dynamic':False,'second_hardware':False})
            save_json(OUT/'artifact_manifest.json',{'artifacts':{str(p.relative_to(OUT)):ref(p) for p in sorted(OUT.rglob('*'))
                if p.is_file() and p.name not in ('artifact_manifest.json','campaign.log','.campaign.lock','.runtime_cache.lock')}})
            print(json.dumps({'event':'module8d2_complete','new_timed_runs':len(bank.new_runs)}),flush=True)
        except BaseException as exc:
            save_json(OUT/'campaign_status.json',{'status':'blocked','reason':str(exc),'traceback':traceback.format_exc(),
                'preserved_attempts':len(bank.new_runs),'no_unplanned_retry_or_budget_growth':True})
            raise


if __name__=='__main__':main()
