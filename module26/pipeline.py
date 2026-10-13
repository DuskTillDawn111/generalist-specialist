"""Durable supervision and conditional authorized CPU stages after GPU ends."""
from .common import *
import subprocess

PYTHON=str(ROOT/'envs/data_env/bin/python')
def execute(module,log):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with (OUT/log).open('x') as stream:
        subprocess.run([PYTHON,'-u','-m',module],cwd=str(ROOT),env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)

def alive(pid):
    try:
        stat=Path('/proc')/str(pid)/'stat'
        return stat.exists() and stat.read_text().split(') ',1)[1][0]!='Z'
    except (FileNotFoundError,ProcessLookupError):return False

def record_terminal_interruption():
    """Never retry a second interruption; terminal evidence remains incomplete."""
    from replica_scaling.telemetry import snapshot
    amendment=read(OUT/'infrastructure_recovery_manifest.json');manifest=read(OUT/'frozen_replication_manifest.json')
    executed=[]
    for slot in amendment['remaining_full_slots']:
        path=OUT/'raw_runs'/(slot['run_id']+'.json')
        if path.exists():executed.append(slot['run_id']);continue
        heartbeat=read(OUT/'durable_heartbeat.json') if (OUT/'durable_heartbeat.json').exists() else {}
        if heartbeat.get('run_id')!=slot['run_id']:break
        journal=OUT/'request_journals'/(slot['run_id']+'.jsonl');rows=[]
        if journal.exists():
            for line in journal.open():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:break # Last torn line preserved, not fabricated.
        checkpoints=[r for r in rows if r['kind']=='CHECKPOINT'];gpu=snapshot()
        save('raw_runs/'+path.name,dict(slot=slot,status='FAILED_INFRASTRUCTURE_INTERRUPTION',valid_session=False,
            errors=['Durable controller disappeared; no second recovery. Incomplete session has no normal U'],phases={},
            worker_identities={},replica_to_family={f'{m}#{i}':m for m,n in slot['deployment'].items() for i in range(n)},
            baseline_GPU=read(OUT/'thermal_prelaunch'/(slot['run_id']+'.json'))['samples'][-1],
            journal=ref(journal) if journal.exists() else None,partial_request_evidence=rows,
            telemetry=dict(samples=[r['telemetry'] for r in checkpoints],errors=[]),
            gpu_release=dict(verified=not gpu['processes'],samples=[gpu]),shutdown={},
            full_U=None,interruption_cause_not_proven=True,source_manifest=ref(OUT/'frozen_replication_manifest.json')))
        executed.append(slot['run_id']);break
    save('replication_campaign.json',dict(at_utc=now(),executed=executed,
        stop=dict(reason='SECOND_INFRASTRUCTURE_INTERRUPTION_NO_RETRY'),planned_full_runs=6,
        initial_foreground_partial_attempt=ref(OUT/'replication_campaign_interrupted_attempt1.json'),
        remaining=[s['run_id'] for s in amendment['remaining_full_slots'] if s['run_id'] not in executed],no_outcome_retry=True))

def main():
    assert not (OUT/'pipeline_completed.json').exists()
    pid=read(OUT/'durable_controller_launch.json')['pid']
    while not (OUT/'replication_campaign.json').exists():
        if not alive(pid):
            time.sleep(2.)
            if not (OUT/'replication_campaign.json').exists():record_terminal_interruption()
            break
        time.sleep(10.)
    execute('module26.report','replication_report.log')
    result=read(OUT/'replication_result.json')
    if result['Stage_C_permitted']:execute('module26.autonomy','autonomous_optimizer.log')
    else:save('autonomous_optimizer_not_executed.json',dict(Stage_C_executed=False,reason=result['status'],
        no_conclusion_that_optimizer_cannot_discover_positive_configuration=True,new_pool_search=False))
    execute('module26.finalize','finalize.log');execute('module26.verify','final_verification.log')
    save('pipeline_completed.json',dict(at_utc=now(),status=read(OUT/'module26_gate.json')['status'],verification_passed=True,
        no_next_Module=True,new_pool_search=False,new_predictor_development=False))
    print('MODULE26_PIPELINE_COMPLETE',flush=True)

if __name__=='__main__':main()
