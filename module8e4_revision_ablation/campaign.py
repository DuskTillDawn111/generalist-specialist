"""Reuse-first scheduled validation of only M10/M01; official endpoints are read-only."""
import os
import fcntl
import traceback
from .common import *
from .variants import make, config, flags_dict
from module8e3_heldout_validation.campaign import Actor, MeasurementBank, request_for_slot, round_barrier
from deployment_core import Workload
from deployment_search.runtime_cache import digest, save_json
from deployment_search.result import jsonable
from deployment_search.tests.fixtures import forbid_enumeration
from integer_replica_pilot.evaluator import ReplicaRuntimeEvaluator, runner, summarize, aggregate, PilotExecutionBlocked
from joint_deploy.measurement.schema import MeasurementConfig


class AblationActor(Actor):
    def checkpoint(self, result):
        for tr in result.iterations:
            tr['ablation_flags'] = flags_dict(self.method)
            tr['selection_policy'] = 'GLOBAL_DELTA_Z_TOP_2' if self.method == 'M10' else 'CLASS_DIVERSIFIED_TOP_2'
        super().checkpoint(result)

    def work(self):
        try:
            engine = make(self.method, self.evaluator, self.checkpoint)
            result = engine.search(self.workload,self.catalog,self.hardware,config(self.method),roles=self.roles,
                capacity_mode='empirical',memory_mode='empirical',calibration=self.cal)
            self.checkpoint(result)
            save_json(self.directory/'search_complete.json',result.to_dict())
        except BaseException as exc:
            self.error=exc
            save_json(self.directory/'actor_failure.json',{'error':repr(exc),'traceback':traceback.format_exc()})
        finally:
            with self.condition:self.done=True;self.condition.notify_all()


class ReuseBank(MeasurementBank):
    def __init__(self):
        self.cap=read(OUT/'reuse_audit.json')['worst_case_new_timed_runs']
        self.frontier_state_cap=read(OUT/'reuse_audit.json')['missing_physical_states_upper_bound']
        super().__init__(self.guarded_runner, directory=OUT)
        self.donors=old_donors()

    def guarded_runner(self, *args):
        verify_execution()
        assert len(self.runs)<=self.cap, 'Prerun new measurement upper bound exceeded'
        return runner(*args)

    def accounting(self):
        a=super().accounting()
        a['reused_E3_repetition_references']=sum(r['cache_hit'] and r['physical_raw_id'].startswith('e3_') for r in self.references)
        a['reused_E4_repetition_references']=sum(r['cache_hit'] and not r['physical_raw_id'].startswith('e3_') for r in self.references)
        a['M00_M11_new_timed_runs']=0
        a['frozen_caps']={'logical_new_method_search_evaluations':72,'new_timed_runs':self.cap,
            'new_search_physical_states':self.frontier_state_cap}
        a['within_budget']=a['logical_search_evaluations']<=72 and len(self.runs)<=self.cap
        return a

    def write(self):
        super().write()
        save_json(OUT/'new_measurement_manifest.json',{'new_runs':self.runs,'count':len(self.runs),
            'raw_sharing_preserves_original_samples':True,'M00_M11_remeasurement':False})


def prepare_campaign():
    assert read(OUT/'regression_equivalence.json')['status']=='PASS'
    assert read(OUT/'adapter_validation.json')['status']=='PASS'
    verify_inputs()
    from candidate_pool_audit.provenance import environment_snapshot
    from replica_scaling.telemetry import snapshot
    old=read(E3/'campaign_manifest.json');env=environment_snapshot();gpu=snapshot()
    assert env==old['environment'],'Environment drift'
    assert all(gpu[k]==old['GPU'][k] for k in ['uuid','name','driver','total_mib']), 'Hardware drift'
    assert not gpu['processes'] and gpu['used_mib']<=128, 'GPU not isolated'
    assets=read(E3/'hash_verification.json')['model_asset_checks']
    for r in assets:assert file_hash(r['path'])==r['expected'],r['path']
    native=read(ROOT/'results/integer_replica_search_real_pilot/before_hashes.json')['native_models_and_adapters']
    for name,r in native.items():assert file_hash(ROOT/name)==r['sha256'],name
    schedule=[]
    for block in read(F/'execution_order.json')['workloads']:
        slots=[]
        for slot in block['slots']:
            if slot['method']=='B2':continue
            slots.append(dict(slot,method={'M1':'M10','M2':'M01'}[slot['method']]))
        schedule.append(dict(block,slots=slots))
    immutable(OUT/'execution_order.json',{'workloads':schedule,'mapping':{'M1':'M10','M2':'M01'},'B2_omitted':True})
    protected={p.name:ref(p) for p in OUT.iterdir() if p.name in ['ablation_spec.json','ablation_spec.json.sha256',
        'feature_flag_validation.json','regression_equivalence.json','adapter_validation.json','reuse_audit.json','execution_order.json']}
    protected.update({'E3_'+k:ref(E3/k) for k in ['campaign_manifest.json','report_manifest.json','physical_measurements.json','m1_results.json','m2_results.json','runtime_context.json','heldout_mechanism_audit.csv']})
    protected.update({'frozen_'+k:r for k,r in old['frozen_shared_inputs'].items()})
    protected.update({'E3_source_'+k:r for k,r in old['execution_sources'].items()})
    for key,value in read(F/'method_comparison_freeze.json').items():
        if isinstance(value,dict) and 'path' in value and 'sha256' in value:protected['shared_source_'+key]=value
    immutable(OUT/'campaign_manifest.json',{'module':'8E-4','prepared_at_utc':utc(),'sources':{
        str(p.relative_to(ROOT)):ref(p) for p in Path(__file__).parent.glob('*.py')},'protected_inputs':protected,
        'environment':env,'GPU':gpu,'model_assets_verified':len(assets),'native_assets_verified':len(native),
        'M00_M11_new_runs_allowed':0,'new_timed_run_cap':read(OUT/'reuse_audit.json')['worst_case_new_timed_runs'],
        'ROUTING_REVISION_PENDING':True})


def run_workload(block,bank,catalog,hardware,roles,cal):
    name=block['workload'];w=next(w for w in read(F/'heldout_workloads.json')['workloads'] if w['id']==name)
    workload=Workload(w['lambda_rps']);bank.catalog=catalog;bank.workload=workload
    protocol=read(F/'method_comparison_freeze.json');directory=OUT/name
    actors={m:AblationActor(m,workload,catalog,hardware,roles,cal,OUT/(m.lower()+'_search_traces')/name) for m in NEW}
    evaluators={m:ReplicaRuntimeEvaluator(directory/m/'search_cache',catalog,CONTEXT,protocol['measurement_settings'],
        seeds=protocol['search_seeds'],max_unique=6,run=bank.raw_run) for m in NEW}
    plan=next(r for r in read(F/'planning_sanity_check.json')['workloads'] if r['workload']==name)
    for m,a in actors.items():
        a.thread.start();req=a.wait_pending();assert req.index==0
        assert req.deployment==plan['seed']['deployment'] and jsonable(req.routing)==plan['canonical']['routing']
    confirmations={m:{'seed':[],'final':[]} for m in NEW};barriers={};decisions=False
    try:
        for index,slot in enumerate(block['slots']):
            verify_execution()
            bank.active=dict(slot,workload=name,workload_sha256=block['workload_sha256'],logical_slot=index,
                protocol_sha256=file_hash(F/'method_comparison_freeze.json'),sharing_policy_sha256=file_hash(F/'measurement_sharing_policy.json'))
            log={'workload':name,'slot_index':index,'slot':slot,'started_at_utc':utc(),'status':'STARTED'}
            bank.schedule_log.append(log);bank.write()
            save_json(OUT/'campaign_status.json',{'status':'running','workload':name,'slot_index':index,'active_slot':slot,'accounting':bank.accounting()})
            m=slot['method'];actor=actors[m]
            if slot['phase']=='search':
                if slot['role']=='candidate' and slot['iteration'] not in barriers:
                    barriers[slot['iteration']]=round_barrier(actors,slot['iteration'])
                    immutable(directory/f'round_{slot["iteration"]}_barrier.json',{'commitments':barriers[slot['iteration']],
                        'at_utc':utc(),'before_any_round_runtime_release':True})
                req=request_for_slot(actor,slot)
                if req is None:
                    log.update(status='SKIPPED',reason='Frozen stopping or absent request; no slot reassignment',completed_at_utc=utc());bank.write();continue
                ident=evaluators[m].identity(req.deployment,workload,req.routing)
                logical={'method':m,'workload':name,'workload_sha256':block['workload_sha256'],'phase':'search','role':slot['role'],
                    'logical_slot':index,'method_state_index':slot['logical_state_index'],'budget_charge':1,'deployment':ident['deployment'],
                    'routing_hash':digest(ident['routing']),'identity_sha256':digest(ident),'state_identity':ident,
                    'round_barrier':barriers.get(slot.get('iteration')),'status':'selected'}
                bank.evaluations.append(logical);assert bank.accounting()['within_budget'];bank.write()
                runtime=evaluators[m].evaluate(req.deployment,workload,req.routing)
                logical.update(status='completed',raw_result_paths=runtime.raw_result_paths,repetitions=runtime.valid_repetition_count,stability=runtime.stability_state_v2)
                actor.return_result(req,runtime);actor.wait_pending()
            else:
                if not decisions:
                    for a in actors.values():assert a.wait_pending() is None and a.done and not a.error
                    immutable(directory/'post_search_freeze.json',{'at_utc':utc(),'confirmations_do_not_influence_search':True,
                        'traces':{m:ref(a.directory/'search_complete.json') for m,a in actors.items()}});decisions=True
                role=slot['role'];tr=actor.trace;x=tr['seed']['deployment'] if role=='seed' else tr['deployment']
                route=tr['iterations'][0]['canonical']['routing'] if role=='seed' else tr['canonical']['routing']
                routing={tuple(k.split('::')):v for k,v in route.items()}
                ident=evaluators[m].identity(x,workload,routing)
                cfg=MeasurementConfig(ident['deployment'],ident['workload'],ident['routing'],random_seed=slot['seed'],**protocol['measurement_settings']).as_dict()
                folder=directory/m/'confirmation';folder.mkdir(parents=True,exist_ok=True);prefix=folder/(role+'_rep_'+str(slot['repetition']))
                raw=Path(str(prefix)+'.json');logpath=Path(str(prefix)+'.log');configpath=Path(str(prefix)+'.config.json')
                assert not raw.exists(),'Do not rerun confirmation';save_json(configpath,cfg)
                outcome=bank.raw_run(cfg,raw,logpath,configpath,CONTEXT)
                rep=summarize(read(raw),workload,catalog,outcome,raw);rep.update(repetition=slot['repetition'],random_seed=slot['seed'],post_search_confirmation=True)
                assert rep['valid_runtime_repetition'],'Invalid confirmation; zero retries'
                save_json(Path(str(prefix)+'.summary.json'),rep);confirmations[m][role].append(rep)
            log.update(status='COMPLETED',completed_at_utc=utc());bank.write()
        for m,a in actors.items():
            aggs={}
            for role,reps in confirmations[m].items():
                assert len(reps)==2 and len({str(Path(r['raw_result_path']).resolve()) for r in reps})==2
                x=a.trace['seed']['deployment'] if role=='seed' else a.trace['deployment'];path=directory/m/'confirmation'/(role+'_aggregate.json')
                aggs[role]=aggregate(sorted(reps,key=lambda r:r['repetition']),'independent_confirmation',path,x);save_json(path,aggs[role])
            assert not {str(Path(r['raw_result_path']).resolve()) for r in confirmations[m]['seed']} & {str(Path(r['raw_result_path']).resolve()) for r in confirmations[m]['final']}
            save_json(directory/m/'result.json',{'method':m,'workload':name,'status':'complete','confirmations':aggs,
                'search_runtime':a.trace['runtime'],'deployment':a.trace['deployment'],'routing':a.trace['canonical']['routing'],
                'logical_search_evaluations':a.trace['runtime_evaluation_count'],'flags':flags_dict(m)})
        save_json(directory/'status.json',{'status':'complete'});print(json.dumps({'event':'workload_complete','workload':name,'accounting':bank.accounting()}),flush=True)
    except BaseException as exc:
        for a in actors.values():a.abort(exc)
        raise


def main():
    for k,v in read(F/'method_comparison_freeze.json')['execution_environment_overrides'].items():os.environ[k]=v
    with (OUT/'.campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/'campaign_status.json').exists(),'No restart or retry after campaign start'
        prepare_campaign();catalog,hardware,roles,cal=inputs();bank=ReuseBank();bank.write()
        try:
            with forbid_enumeration() as guards:
                for block in read(OUT/'execution_order.json')['workloads']:run_workload(block,bank,catalog,hardware,roles,cal)
                assert sum(g.call_count for g in guards)==0
            verify_execution();assert bank.accounting()['within_budget']
            save_json(OUT/'campaign_status.json',{'status':'complete','at_utc':utc(),'accounting':bank.accounting(),'enumeration_calls':0})
        except BaseException as exc:
            bank.write();save_json(OUT/'campaign_status.json',{'status':'incomplete','error':repr(exc),'traceback':traceback.format_exc(),'accounting':bank.accounting()})
            from .report import report
            report();raise
        from .report import report
        report()


if __name__=='__main__':main()
