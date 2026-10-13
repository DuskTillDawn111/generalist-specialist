"""One documented infrastructure recovery; unchanged GPU inference and routes.

The initial foreground controller disappeared with no full raw outcome. Retain
that failure and use a detached controller with incremental evidence. A second
infrastructure interruption is terminal, never repeated for favorable results.
"""
from .common import *
from . import gpu as original

class DurableMonitor(original.ThermalMonitor):
    def __init__(self,runtime,slot):
        super().__init__(runtime,slot);self.persisted=set();self.last_checkpoint=0.;self.persisted_phases=set()
        self.journal=OUT/'request_journals'/(slot['run_id']+'.jsonl');self.journal.parent.mkdir(parents=True,exist_ok=True)
    def sample(self):
        value=super().sample();active=getattr(self,'active_phase',None)
        if active and time.monotonic()-self.last_checkpoint>=10.:
            self.last_checkpoint=time.monotonic();new=[]
            key=active['label']
            if key not in self.persisted_phases:
                new.append(dict(kind='PHASE_WINDOW',label=key,start_time_ns=active['planned_start_time_ns'],
                    planned_end_time_ns=active['planned_end_time_ns'],planned_duration_seconds=active['planned_duration_seconds']))
                self.persisted_phases.add(key)
            pending=[]
            for record in active['request_records']:
                if record['status'] in ['completed','error','incomplete']:
                    identifier=(key,record['request_id'])
                    if identifier not in self.persisted:
                        new.append(dict(kind='REQUEST_TERMINAL',phase=key,record=copy.deepcopy(record)));self.persisted.add(identifier)
                elif record.get('actual_submit_time_ns'):
                    pending.append(copy.deepcopy(record))
            new.append(dict(kind='CHECKPOINT',at_unix_ns=time.time_ns(),phase=key,telemetry=value,pending=pending,
                complete_raw_not_yet_available=True))
            with self.journal.open('a') as f:
                for row in new:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,default=json_native)+'\n')
                f.flush()
            save('durable_heartbeat.json',dict(at_utc=now(),controller_pid=os.getpid(),run_id=self.slot['run_id'],phase=key,
                elapsed_seconds=(time.perf_counter_ns()-active['planned_start_time_ns'])/1e9,journal=str(self.journal)))
        return value

def main():
    amendment=read(OUT/'infrastructure_recovery_manifest.json');manifest=read(OUT/'frozen_replication_manifest.json')
    assert not (OUT/'replication_campaign.json').exists(),'Completed or failed durable campaign cannot retry'
    verify(manifest['implementation']+[manifest['catalog'],manifest['utility_definition'],manifest['utility_reference'],manifest['plan']['sample_pool']])
    verify(amendment['implementation']);assert read(OUT/'positive_probe_audit.json')['passed']
    original.ThermalMonitor=DurableMonitor
    original_save=original.save
    def tagged_save(name,value):
        if name.startswith('raw_runs/'):
            value['infrastructure_recovery_manifest']=ref(OUT/'infrastructure_recovery_manifest.json')
        original_save(name,value)
    original.save=tagged_save
    slots=amendment['remaining_full_slots'];observations=[];executed=[];stop=None
    for slot in slots:
        print('REPLICATION_START',slot['run_id'],now(),flush=True)
        try:raw=original.run(slot,manifest)
        except Exception as exc:
            save('prelaunch_failures/'+slot['run_id']+'.json',dict(slot=slot,error=repr(exc),no_measurement=True))
            stop=dict(run_id=slot['run_id'],reason='PRELAUNCH_FAILED_NO_SECOND_REPAIR',error=repr(exc));break
        executed.append(slot['run_id']);path=OUT/'raw_runs'/(slot['run_id']+'.json');ob=observe(raw,path)
        observations.append(ob);save('replication_observations.json',observations)
        mem=original.memory_values(raw,manifest['safe_factor'])
        print('REPLICATION_END',slot['run_id'],'eligible',ob['eligible'],'unstable',ob['hard_unstable'],'U',ob['U'],'errors',raw['errors'],now(),flush=True)
        if not ob['eligible'] or not mem['passed'] or not raw['gpu_release']['verified']:
            stop=dict(run_id=slot['run_id'],reason='UNSTABLE_OR_ILLEGAL_OR_RUNTIME_FAILURE_NO_SECOND_REPAIR',
                eligible=ob['eligible'],hard_unstable=ob['hard_unstable'],memory=mem,errors=raw['errors']);break
        pair=[o for o in observations if o['repetition']==slot['repetition']]
        if len(pair)==2:
            a=next(o for o in pair if 'proposed' in o['run_id']);b=next(o for o in pair if 'control' in o['run_id'])
            print('PAIRED_DELTA',slot['repetition'],a['U']-b['U'],'retain regardless of sign',flush=True)
    save('replication_campaign.json',dict(at_utc=now(),executed=executed,stop=stop,planned_full_runs=6,
        initial_foreground_partial_attempt=ref(OUT/'replication_campaign_interrupted_attempt1.json'),
        total_GPU_startup_attempts_including_initial_interruption=len(executed)+1,
        no_outcome_retry=True,no_negative_outcome_early_stop=True,
        remaining=[s['run_id'] for s in slots if s['run_id'] not in executed],new_predictor=False,new_pool_search=False))
    print('STAGE_B_CAMPAIGN_COMPLETE',len(executed),'stop',stop,flush=True)

if __name__=='__main__':main()
