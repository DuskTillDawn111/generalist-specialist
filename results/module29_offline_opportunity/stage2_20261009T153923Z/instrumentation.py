"""Sidecar observation. Frozen router/native worker files are never modified."""
import ast
import collections
import copy
import hashlib
import inspect
import json
import queue
import textwrap
import threading
import time
from pathlib import Path

from joint_deploy.measurement.deployment_runtime import DeploymentRuntime
import joint_deploy.measurement.deployment_runtime as native_runtime

OUT = Path(__file__).resolve().parent

def save(name, value):
    target = OUT / name
    assert target.parent == OUT, 'Outputs must stay in stage2 directory'
    tmp = target.with_suffix(target.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(target)

def ref(path):
    p = Path(path)
    return dict(path=str(p.resolve()), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def router_state(router):
    if router is None:
        return None
    return dict(mode=router.mode, tokens=router.tokens, last_time=router.last_time,
                next_g_time=router.next_g_time, fallback_until=router.fallback_until,
                service=[dict(worker=w, task=t, ewma_ms=v)
                         for (w,t),v in sorted(router.service.items())],
                predictions=dict(router.predictions),
                prediction_errors=copy.deepcopy(router.prediction_errors))

def restore_router(cls, parameters, state):
    r = cls(parameters)
    for k in ['mode', 'tokens', 'last_time', 'next_g_time', 'fallback_until']:
        setattr(r,k,state[k])
    r.service={(x['worker'],x['task']):x['ewma_ms'] for x in state['service']}
    r.predictions=dict(state['predictions'])
    r.prediction_errors=copy.deepcopy(state['prediction_errors'])
    return r

class Journal:
    """Serialization and disk writes are outside runtime's observation lock."""
    def __init__(self, filename):
        self.path=OUT/filename
        assert not self.path.exists()
        self.queue=queue.SimpleQueue(); self.seq=0; self.lock=threading.Lock()
        self.error=None; self.written=0
        self.thread=threading.Thread(target=self._write,daemon=True)
        self.thread.start()
    def emit(self, kind, **values):
        with self.lock:
            self.seq+=1
            self.queue.put(dict(seq=self.seq,kind=kind,**values))
    def _write(self):
        try:
            with self.path.open('x') as stream:
                last=time.monotonic()
                while True:
                    row=self.queue.get()
                    if row is None: break
                    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    self.written+=1
                    if time.monotonic()-last>1:
                        stream.flush();last=time.monotonic()
                stream.flush()
        except Exception as exc:
            self.error=repr(exc)
    def close(self):
        self.queue.put(None);self.thread.join(timeout=30)
        if self.thread.is_alive():raise RuntimeError('Journal writer did not finish')
        if self.error:raise RuntimeError(self.error)

def instrumented_collector():
    """Keep native collector AST verbatim; append one observer in its locked body."""
    original=ast.parse(textwrap.dedent(inspect.getsource(DeploymentRuntime._collect)))
    patched=copy.deepcopy(original)
    locked=[n for n in ast.walk(patched) if isinstance(n,ast.With)
            and any(ast.unparse(x.context_expr)=='self.lock' for x in n.items)]
    assert len(locked)==1
    observer=ast.parse('self._log_commit(record, kind, received_ns)').body[0]
    locked[0].body.append(observer)
    stripped=copy.deepcopy(patched)
    stripped_locked=[n for n in ast.walk(stripped) if isinstance(n,ast.With)
                     and any(ast.unparse(x.context_expr)=='self.lock' for x in n.items)]
    stripped_locked[0].body.pop()
    assert ast.dump(stripped)==ast.dump(original), 'Native collector logic changed'
    ast.fix_missing_locations(patched)
    namespace=dict(vars(native_runtime))
    exec(compile(patched,str(OUT/'instrumentation.py')+'::native_collector','exec'),namespace)
    return namespace['_collect']

class LoggedRuntime(DeploymentRuntime):
    _collect=instrumented_collector()
    def __init__(self, config, registry, journal):
        super().__init__(config,registry)
        self.journal=journal;self.visible={};self.router=None
        self.latest_received_ns=0;self.phase='startup';self.window_origin_ns=None
    def _log_commit(self,record,kind,received_ns):
        # Called after native RESULT/ERROR processing while native RLock is held.
        commit_ns=time.perf_counter_ns()
        record['collector_commit_time_ns']=commit_ns
        self.journal.emit('COLLECTOR_COMMIT',time_ns=commit_ns,
                          reader_received_time_ns=received_ns,phase=record['phase'],
                          request_id=record['request_id'],worker=record['model'],task=record['task'],
                          status=record['status'],response_kind=kind,
                          worker_latency_ms=record.get('worker_latency_ms'),
                          worker_queue_wait_ms=record.get('worker_queue_wait_ms'))
    def enqueue(self, record, sample):
        with self.lock:
            super().enqueue(record,sample)
            self.visible[record['request_id']]=record
            self.journal.emit('ENQUEUE',time_ns=record['enqueued_time_ns'],phase=record['phase'],
                              request_id=record['request_id'],task=record['task'],
                              worker=record['model'],status=record['status'])
    def observation(self,now_ns,harvest):
        # Caller holds lock. Native handler START/END never accessed here.
        before=router_state(self.router);observed=[]
        if harvest:
            for rid,r in list(self.visible.items()):
                if r['status']=='completed' and r['completion_time_ns']<=now_ns:
                    assert r['collector_commit_time_ns']<=now_ns
                    self.latest_received_ns=max(self.latest_received_ns,r['completion_time_ns'])
                    if self.router:self.router.observe_completion(r,now_ns/1e9)
                    observed.append(dict(request_id=rid,model=r['model'],task=r['task'],
                                         completion_time_ns=r['completion_time_ns'],
                                         collector_commit_time_ns=r['collector_commit_time_ns'],
                                         worker_latency_ms=r['worker_latency_ms']))
                    del self.visible[rid]
        pending=collections.defaultdict(list)
        for r in self.visible.values():
            if r['status'] in ['queued','submitted']:
                pending[r['model']].append({k:r.get(k) for k in
                    ['request_id','task','model','enqueued_time_ns','actual_submit_time_ns']})
        for rows in pending.values():rows.sort(key=lambda r:r['enqueued_time_ns'])
        visible=[dict(request_id=r['request_id'],worker=r['model'],task=r['task'],status=r['status'],
                      reader_received_time_ns=r.get('completion_time_ns'),
                      collector_commit_time_ns=r.get('collector_commit_time_ns'))
                 for r in self.visible.values()]
        return dict(pending=dict(pending),harvested_completions=observed,
                    router_before_harvest=before,router_before_decide=router_state(self.router),
                    visible_records=visible,latest_observed_completion_time_ns=self.latest_received_ns)

def guards(parameters, state, estimate, now, admitted, capacity_ok):
    """Diagnostic PASS/FAIL evaluations; actual choice always original decide()."""
    p=parameters;e=estimate
    gap=e['S_system_cost_proxy_ms']-e['G_system_cost_proxy_ms']
    tokens=min(1.,state['tokens']+max(0.,now-state['last_time'])*p['G_token_rate_rps']) if state['last_time'] is not None else state['tokens']
    mode=state['mode']
    if mode and gap<p['exit_margin_ms']:mode=False
    elif not mode and gap>p['entry_margin_ms']:mode=True
    values=dict(admitted=admitted,capacity_ok=capacity_ok,
                prediction_fallback=now>=state['fallback_until'],
                G_target_inflight=e['G_target_inflight']<p['G_max_target_inflight'],
                G_empty=e['G_pending']==0,cooldown=now>=state['next_g_time'],
                token_backpressure=tokens>=1.,hysteresis_mode_after_update=mode,
                exit_cost_gap=gap>p['exit_margin_ms'],entry_cost_gap=gap>p['entry_margin_ms'])
    return {k:'PASS' if v else 'FAIL' for k,v in values.items()}
