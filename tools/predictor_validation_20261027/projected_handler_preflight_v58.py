"""Run all original candidate cohorts on CPU; no seal or GPU collection.

Forecasts are stored as exact, fully verified gzip JSON without retaining
redundant multi-gigabyte uncompressed files. UNKNOWN is an observed outcome.
"""
from __future__ import annotations

import argparse
from multiprocessing.connection import wait as wait_pipes
import multiprocessing
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import tempfile
import time

from .implementation_source_closure_v27 import implementation_sources, strict_references
from .learned_availability_forecast_v20 import verify_sources
from .native_collect import receipt
from .projected_handler_forecast_v58 import ObservableContextForecast
from .prospective_flow_contract_v11 import gate_cell
from .unknown_preflight_archive_v1 import save_synced

SCHEMA='gsdeploy.projected_worker_handler_context_full_cohort_CPU_preflight.v58'
CANDIDATES=('shared_balanced','shared_skewed','shared_spill_low','shared_spill_high')


class ProgressScheduler:
    def __init__(self,scheduler,path,candidate):
        self.scheduler,self.path,self.candidate=scheduler,path,candidate
        self.calls,self.last=0,0.

    def decide(self,view):
        result=self.scheduler.decide(view);self.calls+=1
        if time.monotonic()-self.last>=20:
            with tempfile.NamedTemporaryFile('w',dir=self.path.parent,delete=False) as stream:
                json.dump(dict(candidate_id=self.candidate,clock_ns=view.clock_ns,
                    scheduler_calls=self.calls,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),stream)
                temporary=Path(stream.name)
            os.replace(temporary,self.path);self.last=time.monotonic()
        return result

    def provenance(self):return self.scheduler.provenance()


class ObservedForecast(ObservableContextForecast):
    def engine(self,*,max_events):
        engine=super().engine(max_events=max_events)
        engine._engine._engine.scheduler=ProgressScheduler(engine.scheduler,self.progress,self.candidate)
        return engine


class DigestWriter:
    def __init__(self,stream):self.stream,self.sha,self.size=stream,hashlib.sha256(),0
    def write(self,value):
        data=value.encode('utf-8');self.sha.update(data);self.size+=len(data);self.stream.write(data)


def save_forecast(path,value):
    with path.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0) as compressed:
            writer=DigestWriter(compressed)
            json.dump(value,writer,ensure_ascii=False,allow_nan=False,separators=(',',':'))
            writer.write('\n')
        raw.flush();os.fsync(raw.fileno())
    sha=hashlib.sha256();size=0
    with gzip.open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            sha.update(block);size+=len(block)
    if sha.hexdigest()!=writer.sha.hexdigest() or size!=writer.size:
        raise ValueError('Full compressed forecast roundtrip must equal original bytes')
    return dict(artifact=receipt(path),encoding='gzip UTF-8 JSON',uncompressed_sha256=sha.hexdigest(),
        uncompressed_bytes=size,complete_lossless_roundtrip_verified=True)


# Construct once before fork. Each CPU worker gets a private copy of every
# lifecycle/controller state. No model, event clock or result is shared back.
_PARALLEL_MODEL=None


def execute_candidate(cid,prepared,out,plan_ref,plan,sources):
    model=_PARALLEL_MODEL
    if model is None:raise RuntimeError('Parent-initialized private CPU forecast required')
    verify_sources(sources);cell_began=time.monotonic()
    model.progress=out/(cid+'-progress.json');model.candidate=cid
    public=json.loads((prepared/('public-'+cid+'.json')).read_text())['forecast_inputs']
    routes=json.loads((prepared/('routes-'+cid+'.json')).read_text())
    result=model.execute(plan_ref,cid,public,routes,model.bundle['startup']['training_model'],max_events=5_000_000)
    candidate=next(c for c in plan['candidates'] if c['id']==cid)
    cell=gate_cell(result,result['full_public_cohort'],candidate,plan['run_id'],plan['queue_sample_offsets_ns'],
        phase_start_ns=public['phase_start_ns'])
    save_synced(out/(cid+'-unsealed-cell.json'),cell)
    compressed=save_forecast(out/(cid+'-forecast.json.gz'),result)
    forecast=result.get('forecast') or result.get('unscored_forecast_diagnostic',{}).get('forecast')
    summary=dict(candidate_id=cid,status=result['status'],reason=result.get('reason'),
        result=compressed,metrics=result.get('metrics'),cohort_size=len(result['full_public_cohort']),
        starts=None if forecast is None else len(forecast['starts']),
        receipts=None if forecast is None else len(forecast['finishes']),
        elapsed_seconds=time.monotonic()-cell_began,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        independent_process_id=os.getpid())
    verify_sources(sources);save_synced(out/(cid+'-progress-report.json'),summary)
    return summary


def candidate_process(connection,*args):
    try:
        connection.send(dict(ok=True,summary=execute_candidate(*args)))
    except BaseException as exc:
        connection.send(dict(ok=False,candidate_id=args[0],failure=dict(type=type(exc).__name__,reason=str(exc))))
    finally:
        connection.close()


def run(configuration,prepared,output):
    global _PARALLEL_MODEL
    prepared=Path(prepared).absolute();out=Path(output).absolute()
    if out.exists():raise FileExistsError(out)
    config_ref=receipt(configuration);plan_ref=receipt(prepared/'plan.json')
    plan=json.loads(Path(plan_ref['path']).read_text())
    if set(c['id'] for c in plan['candidates'])!=set(CANDIDATES):
        raise ValueError('All four original routing candidates required')
    sources=strict_references([config_ref,plan_ref,implementation_sources(__file__),
        *[receipt(prepared/(stem+'-'+cid+'.json')) for cid in CANDIDATES for stem in ('public','routes')]])
    out.mkdir(parents=True,exist_ok=False);save_synced(out/'intent.json',dict(schema=SCHEMA,sources=sources,
        candidates=list(CANDIDATES),GPU_started=False,seal_authorized=False,
        independent_forked_CPU_candidate_processes=True))
    began=time.monotonic();report=dict(schema=SCHEMA,status='RUNNING',cells=[],GPU_started=False,
        validation_labels_read=False,collection_eligible=False,protocol_sealed=False,
        independent_forked_CPU_candidate_processes=True,parallelism=4)
    try:
        verify_sources(sources)
        _PARALLEL_MODEL=ObservedForecast(json.loads(Path(config_ref['path']).read_text()),allow_unvalidated_training_hypothesis=True)
        sources=strict_references([sources,_PARALLEL_MODEL._lifecycle_sources]);save_synced(out/'source-before.json',sources)
        report['initialization_seconds']=time.monotonic()-began
        # CPU-only fork happens after immutable provider compilation. Every
        # candidate starts from the same unused numerical state in a separate
        # address space; no candidate carries the preceding one's history.
        context=multiprocessing.get_context('fork');workers={};pending={};failures=[]
        try:
            for cid in CANDIDATES:
                receive,send=context.Pipe(duplex=False)
                process=context.Process(target=candidate_process,args=(send,cid,prepared,out,plan_ref,plan,sources))
                process.start();send.close();workers[cid]=process;pending[receive]=cid
            # One explicit Process per candidate, not a reusable process pool:
            # even a very fast UNKNOWN cannot leave state for another route.
            while pending:
                ready=wait_pipes(list(pending),timeout=10.)
                for connection in ready:
                    cid=pending.pop(connection)
                    try:message=connection.recv()
                    except EOFError:message=dict(ok=False,candidate_id=cid,failure=dict(type='ChildExit',reason='CPU candidate exited without result'))
                    connection.close()
                    if message['ok']:
                        summary=message['summary'];report['cells'].append(summary)
                        print(json.dumps(dict(event='CANDIDATE_FINISHED',**summary)),flush=True)
                    else:failures.append(message)
                for connection,cid in list(pending.items()):
                    if not workers[cid].is_alive() and not connection.poll():
                        failures.append(dict(candidate_id=cid,failure=dict(type='ChildExit',reason=str(workers[cid].exitcode))))
                        pending.pop(connection);connection.close()
        finally:
            for process in workers.values():process.join()
            for connection in pending:connection.close()
        if failures:
            report['candidate_failures']=failures
            raise RuntimeError('CPU candidate processes failed: '+json.dumps(failures))
        report['cells'].sort(key=lambda c:CANDIDATES.index(c['candidate_id']))
        report.update(status='FOUR_FULL_COHORT_CPU_PREFLIGHTS_EXECUTED',
            all_original_six_term_metrics_available=all(c['metrics'] is not None for c in report['cells']),
            physical_prediction_accuracy_validated=False)
    except BaseException as exc:
        report.update(status='FAILED',failure=dict(type=type(exc).__name__,reason=str(exc)))
        raise
    finally:
        _PARALLEL_MODEL=None
        try:
            verify_sources(sources);report['source_bytes_unchanged']=True
        except Exception as exc:
            report['source_bytes_unchanged']=False;report['source_failure']=str(exc)
            report['status']='FAILED'
        report['elapsed_seconds']=time.monotonic()-began
        report['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save_synced(out/'source-after.json',[receipt(r['path']) for r in sources])
        save_synced(out/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configuration',required=True);p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();r=run(args.configuration,args.prepared,args.output)
    print(json.dumps(dict(status=r['status'],elapsed_seconds=r['elapsed_seconds'])),flush=True)
