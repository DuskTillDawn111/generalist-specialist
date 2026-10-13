import argparse,fcntl,traceback,os
from .common import *
from .prepare import prepare
from deployment_core import Workload
from deployment_search import IntegerGuidedDeploymentSearch,SearchConfig
from deployment_search.real_objective import NoiseAwareRuntimeObjective
from deployment_search.tests.fixtures import forbid_enumeration
from integer_replica_pilot.evaluator import ReplicaRuntimeEvaluator,PilotExecutionBlocked
from integer_replica_multiworkload.campaign import confirmation

def run_workload(name):
    verify_freeze();directory=OUT/name;protocol=read(directory/'protocol.json')
    catalog,hardware,roles,cal=inputs();workload=Workload(protocol['workload']['arrival_rates_rps'])
    evaluator=ReplicaRuntimeEvaluator(directory/'search_cache',catalog,directory/'runtime_context.json',
                  protocol['measurement_settings'],seeds=protocol['search_seeds'],max_unique=6)
    complete=directory/'search_complete.json'
    if complete.exists():trace=read(complete)
    else:
        print(json.dumps({'event':'workload_start','workload':name}),flush=True)
        with forbid_enumeration() as guards:
            result=IntegerGuidedDeploymentSearch(evaluator,predictor=None,runtime_objective=NoiseAwareRuntimeObjective(1.,.05),
                trace_callback=lambda r:save_json(directory/'search_trace.json',r.to_dict())).search(workload,catalog,hardware,
                SearchConfig(**protocol['search_config']),roles=roles,capacity_mode='empirical',memory_mode='empirical',calibration=cal)
        trace=result.to_dict();trace['enumeration_calls']=sum(g.call_count for g in guards)
        assert trace['runtime_evaluation_count']<=6 and trace['search_iterations']<=3 and trace['enumeration_calls']==0
        save_json(directory/'search_trace.json',trace);save_json(complete,trace)
    verify_freeze()
    confirmation(directory,trace,protocol,evaluator,catalog,workload)
    save_json(directory/'status.json',{'status':'complete','workload':name,'runtime_states':trace['runtime_evaluation_count'],
               'stopped_after_search_and_confirmation':True})
    print(json.dumps({'event':'workload_complete','workload':name,'deployment':trace['deployment'],
                     'runtime_states':trace['runtime_evaluation_count']}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    # Preserve the historical OCR library setup for replica names paddleocr#i.
    # The original Worker.start applies it only to the literal family name.
    cuda_lib='/root/miniconda3/envs/ocr_env/lib/python3.11/site-packages/nvidia/cu13/lib'
    current=os.environ.get('LD_LIBRARY_PATH','')
    os.environ['LD_LIBRARY_PATH']=cuda_lib+(':'+current if current and current!=cuda_lib else '')
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        freeze=prepare()
        if args.prepare_only:return
        save_json(OUT/'campaign_status.json',{'status':'running','execution_order':freeze['execution_order']})
        try:
            for name in freeze['execution_order']:
                if (OUT/name/'status.json').exists() and read(OUT/name/'status.json')['status']=='complete':continue
                save_json(OUT/'campaign_status.json',{'status':'running','active_workload':name,'execution_order':freeze['execution_order']})
                run_workload(name)
            verify_freeze()
            from .report import report
            report()
            save_json(OUT/'campaign_status.json',{'status':'complete','stopped_after_module':'8C','baseline':False,
                                               'ablation':False,'dynamic':False,'second_hardware':False})
            from .report import artifact_manifest
            artifact_manifest()
            print(json.dumps({'event':'module8c_complete'}),flush=True)
        except BaseException as exc:
            save_json(OUT/'campaign_status.json',{'status':'blocked','reason':str(exc),'type':type(exc).__name__,
                         'no_retry_or_budget_growth':True,'traceback':traceback.format_exc()})
            raise
if __name__=='__main__':main()
