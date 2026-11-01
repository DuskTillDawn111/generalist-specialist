"""Source-rebound execution boundary; prepare/audit are CPU only.

The collector body below is the frozen v3 collect body with a new audit and
RAW_SCHEMA. Shared dispatch/checkpoint/cleanup helpers are reused unchanged.
No old module global is patched. The new raw is deliberately rejected by old
v3 review/intake; an explicit new review adapter gates every following attempt.
"""
from __future__ import annotations
import argparse
import copy
import ctypes
import fcntl
import json
import os
from pathlib import Path
import struct
import sys
import time

from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.runner import record
from . import twoFIFO_owned_collect_v3 as original
from . import twoFIFO_source_rebound_prepare_v1 as prep
from . import twoFIFO_training_semantic_preflight_v2 as strict
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .stage_response_features_v1 import receipt, checked
from .twoFIFO_training_manifest_v1 import require, digest, OWNERS, MEASUREMENT_NS, DRAIN_NS
from .profile_owned_collect_v3 import StartupBoardMonitor, memory_ownership, MONITORING_SCOPE
from .twoFIFO_owned_collect_v3 import (ActivityLedger, atomic_save, initial_rows,
    runtime_config, wait_until, dispatch_case, audit_activity, finish_collection,
    ACTIVITY_FILE, ACTIVITY_SCHEMA, TIME_DOMAIN)

ROOT = prep.ROOT
SCHEMA = 'gsdeploy.twoFIFO_source_rebound_execution.v1'
RAW_SCHEMA = 'gsdeploy.twoFIFO_source_rebound_training_capture.v1'
CONT_SCHEMA = SCHEMA + '.continuation'
CAMPAIGN = ROOT/'artifacts/predictor-twoFIFO-source-rebound-v1-20261028-cpu09/campaign.json'
CAMPAIGN_SHA = '735f6d5095319fb83eae0738a88ae5cb9dcea372b01aa3c4493fe6d752ed7f9e'
PREP_SHA = '8dde55bc378dafba7a5bf02de244b9eef700fe88a957672903d9937ea43ddea2'
supervisor = original.supervisor


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _verify(ref):
    require(type(ref) is dict and set(ref) == {'path', 'sha256'}
        and receipt(ref['path']) == ref, 'Immutable source/evidence receipt changed: '+str(ref))


def audit_campaign(path=CAMPAIGN):
    """Independent closure, workload, context and actual-input verification."""
    path = Path(path).resolve()
    if path == CAMPAIGN.resolve():
        require(receipt(path)['sha256'] == CAMPAIGN_SHA, 'Exact independently reviewed cpu02 source-rebound campaign required')
    else:
        require(path.name == 'campaign.json', 'Canonical refreshed source-rebound campaign required')
    campaign = _read(path)
    require(campaign.get('schema') == prep.SCHEMA and campaign.get('status') ==
        'CPU_PREPARED_SOURCE_REBOUND_NOT_COLLECTED' and campaign.get('attempt_count') == 12
        and campaign.get('no_retries') is True and campaign.get('GPU_started') is False
        and campaign.get('numerical_fit_run') is False and campaign.get('validation_labels_read') is False
        and campaign.get('execute_entrypoint') is None, 'Exact CPU-only preparation required')
    require(campaign.get('source_rebound_id') and len(campaign['source_rebound_id']) == 32
        and campaign['preparation_source']['path'] == str(Path(prep.__file__).resolve())
        and campaign['preparation_source']['sha256'] == PREP_SHA,
        'Exact source-bound preparation implementation required')
    _verify(campaign['preparation_source']); _verify(campaign['old_campaign'])
    old_path, old_campaign, old_plans = prep._old_campaign(campaign['old_campaign']['path'])
    require(len(campaign['attempts']) == 12, 'Exactly twelve rebound attempts required')
    pair = [receipt(p) for p in prep.PAIR]
    require(campaign['current_sources'] == pair and len(campaign['frozen_sources']) == 2,
        'Exact current pair binding required')
    for live, frozen in zip(pair, campaign['frozen_sources']):
        _verify(frozen)
        require(frozen['path'] == str(path.parent/'source-freeze'/Path(live['path']).relative_to(ROOT))
            and live['sha256'] == frozen['sha256'], 'Canonical rebound pair freeze required')
    _verify(campaign['context']); context = checked(campaign['context'])
    require(campaign['context']['path'] == str(path.parent/'context.json'), 'Canonical rebound context required')
    old_context_ref = old_plans[0][1]['context']; old_context = checked(old_context_ref)
    expected_context = prep._replace_context(old_context, old_context_ref['path'],
        path.parent/'source-freeze', campaign['frozen_sources'], campaign['source_rebound_id'])
    require(context == expected_context, 'Rebound context changed beyond the two exact source bindings')
    execution_sources = context['execution_sources']
    require(len({x['original_path'] for x in execution_sources}) == len(execution_sources)
        and context['sources'] == [x['frozen'] for x in execution_sources], 'Complete unique context source closure required')
    for item in execution_sources:
        _verify(item['frozen'])
        require(receipt(item['original_path'])['sha256'] == item['frozen']['sha256'],
            'Context executable differs from frozen source')
    # Frozen designs are authenticated, then their public cases are rebuilt
    # without calling strict.load_frozen(), whose old current-source gate is
    # intentionally incompatible with this rebound campaign.
    base = checked(dict(path=str(strict.BASE), sha256=strict.BASE_SHA))
    entry = checked(dict(path=str(strict.ENTRY), sha256=strict.ENTRY_SHA))
    pool = strict.source_catalog(context)
    inputs = strict.expected_inputs(context, pool)
    require(base['inputs'] == entry['inputs'] == inputs, 'Actual source inputs differ from frozen declarations')
    strict.compare_blocks(base['capture_blocks'], strict.expected_base_blocks(inputs))
    from . import twoFIFO_OCR_entry_manifest_v2 as ocr
    strict.compare_blocks(entry['capture_blocks'], ocr.build_cases(dict(inputs=inputs,
        capture_blocks=strict.expected_base_blocks(inputs))))
    require(base['runtime_policy'] == entry['runtime_policy'] == dict(admission=None, batching=False,
        weight_reuse=False, enqueue_all_without_waiting_for_RESULT_or_READY=True), 'Original natural FIFO policy required')
    impl = implementation_sources(original.__file__)
    impl_paths = {r['path'] for r in impl}
    attempts = []; old_ids = {p['execution_id'] for _, p in old_plans}
    for ref, (old_ref, old_plan) in zip(campaign['attempts'], old_plans):
        _verify(ref); plan = checked(ref)
        require(ref['path'] == str(path.parent/old_plan['attempt_id']/'plan.json'), 'Canonical attempt plan path required')
        expected = copy.deepcopy(old_plan)
        for key in ('execution_id', 'context', 'source_freeze', 'sources', 'source_rebound',
                    'execution_context_sha256', 'GPU_started'):
            expected[key] = plan.get(key)
        require(plan == expected, 'Rebound changed original immutable workload semantics')
        require(plan['schema'] == original.SCHEMA+'.attempt' and plan['split'] == 'training'
            and plan['no_retries'] is True and plan['GPU_started'] is False, 'Exact training/no-retry plan required')
        require(plan['context'] == campaign['context'] and plan['base_manifest'] == receipt(strict.BASE)
            and plan['OCR_entry_manifest'] == receipt(strict.ENTRY), 'Exact design and rebound context binding required')
        require(plan['runtime_config'] == runtime_config().as_dict() and plan['runtime_policy'] == base['runtime_policy']
            and plan['resident_owners'] == list(OWNERS) and plan['measurement_ns'] == MEASUREMENT_NS
            and plan['drain_ns'] == DRAIN_NS and plan['supervisor_budget'] == supervisor.BUDGET
            and plan['preflight_free_bytes_required'] == supervisor.PREFLIGHT, 'Original runtime/windows/budget required')
        require(plan['collector_timing_policy'] == old_plan['collector_timing_policy']
            and plan['priming'] == original.priming_catalog(context), 'Original timing and eighteen priming requests required')
        wanted = next(b for b in original.shard_cases(base, entry) if b['capture_id'] == plan['capture_id'])['shards'][plan['shard_index']-1]
        require(plan['cases'] == wanted and len(wanted) <= 13 and plan['shards_in_block'] == 4,
            'Exact original balanced public shard required')
        _verify(plan['prepared_pool'])
        rows = [json.loads(line) for line in Path(plan['prepared_pool']['path']).read_text().splitlines()]
        require(len(rows) == 16 and {r['pool_id']: r for r in rows} == pool,
            'Exact complete actual training/priming pool required')
        expected_sources = copy.deepcopy(old_plan['sources'])
        by_path = {r['path']: r for r in pair}
        for item in expected_sources:
            if item['path'] in by_path: item.update(by_path[item['path']])
        require(plan['sources'] == expected_sources, 'Exact old closure plus only rebound pair required')
        for item in plan['sources']: _verify(item)
        require(len(plan['source_freeze']) == len(impl) and
            {r['original_path'] for r in plan['source_freeze']} == impl_paths, 'Full unique frozen collector entry closure required')
        old_freeze = {r['original_path']: r['frozen'] for r in old_plan['source_freeze']}
        for item in plan['source_freeze']:
            original_path = item['original_path']; frozen = item['frozen']; _verify(frozen)
            require(frozen['path'] == str(path.parent/'source-freeze'/Path(original_path).relative_to(ROOT))
                and frozen['sha256'] == receipt(original_path)['sha256'], 'Canonical matching frozen executable required')
            require(original_path in by_path or frozen['sha256'] == old_freeze[original_path]['sha256'],
                'Non-pair frozen source changed from original campaign')
        require(plan['sources'] == strict_references(plan['sources']), 'Unique canonical direct source closure required')
        require(plan['source_rebound'] == dict(schema=prep.SCHEMA+'.plan_binding', old_plan=old_ref,
            old_campaign=campaign['old_campaign'], current_sources=pair, frozen_sources=campaign['frozen_sources'],
            context=campaign['context'], validation_labels_read=False, GPU_started=False,
            source_rebound_id=campaign['source_rebound_id']), 'Exact source rebound provenance required')
        require(plan['execution_context_sha256'] == digest({k:v for k,v in plan.items() if k != 'execution_context_sha256'}),
            'Execution context drift')
        require(type(plan['execution_id']) is str and len(plan['execution_id']) == 32 and plan['execution_id'] not in old_ids,
            'Fresh execution identity required')
        attempts.append(plan)
    require(len({p['execution_id'] for p in attempts}) == 12 and len({p['attempt_id'] for p in attempts}) == 12,
        'Twelve unique execution and attempt identities required')
    return campaign


def audit_plan(path):
    campaign_path = Path(path).resolve().parent.parent/'campaign.json'
    campaign = audit_campaign(campaign_path)
    require(receipt(path) in campaign['attempts'], 'Only exact rebound campaign plans may execute')
    return _read(path)


class SourceChangeGuard:
    """Kernel event queue detects source change-and-restore as well as drift.

    Watch only authenticated file names. No polling/hash work runs in measured
    intervals. Queue overflow, replacement, chmod, writes or deletion fail closed.
    """
    MASK = 0x2 | 0x4 | 0x8 | 0x40 | 0x80 | 0x100 | 0x200 | 0x400 | 0x800
    def __init__(self, refs):
        self.refs = strict_references(refs); self.fd = -1; self.names = {}; self.paths = {}; self.inode_wds = set()
        self.baseline_stats = {r['path']: self._stat(r['path']) for r in self.refs}
        # Existing immutable evidence may legitimately have nlink>1. Its
        # exact baseline nlink is authenticated and any runtime change fails.
        self.baseline_dirs = {}
        for ref in self.refs:
            q = Path(ref['path']).resolve()
            while q.parent != q:
                self.baseline_dirs[str(q.parent)] = self._dir_stat(q.parent)
                q = q.parent
        lib = ctypes.CDLL(None, use_errno=True)
        lib.inotify_init1.argtypes = [ctypes.c_int]; lib.inotify_init1.restype = ctypes.c_int
        lib.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        lib.inotify_add_watch.restype = ctypes.c_int
        self.fd = lib.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0: raise OSError(ctypes.get_errno(), 'Cannot establish source change guard')
        try:
            groups = {}
            for ref in self.refs:
                p = Path(ref['path']).resolve()
                # A parent-of-parent rename can redirect an otherwise watched
                # pathname without mutating its original inode. Watch every
                # path component through the filesystem root, so swap/restore
                # of an ancestor is evidence even when final SHA is identical.
                while p.parent != p:
                    groups.setdefault(p.parent, set()).add(p.name)
                    p = p.parent
            for parent, names in groups.items():
                wd = lib.inotify_add_watch(self.fd, os.fsencode(parent), self.MASK)
                if wd < 0: raise OSError(ctypes.get_errno(), 'Cannot watch source directory '+str(parent))
                self.names[wd] = names; self.paths[wd] = str(parent)
            # Watch each canonical inode itself. A pre-existing or transient
            # hardlink alias can write through the inode without a directory
            # event on the canonical parent; inode events fail closed.
            for ref in self.refs:
                wd = lib.inotify_add_watch(self.fd, os.fsencode(ref['path']), self.MASK)
                if wd < 0: raise OSError(ctypes.get_errno(), 'Cannot watch source inode '+ref['path'])
                self.inode_wds.add(wd)
            self.check()
        except BaseException:
            self.close(); raise
    @staticmethod
    def _stat(path):
        try: st = os.stat(path)
        except OSError as exc: raise ValueError('Authenticated source path unavailable: '+str(path)) from exc
        return (st.st_dev, st.st_ino, st.st_mode, st.st_nlink, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

    @staticmethod
    def _dir_stat(path):
        try: st = os.stat(path)
        except OSError as exc: raise ValueError('Authenticated source ancestor unavailable: '+str(path)) from exc
        return (st.st_dev, st.st_ino, st.st_mode)

    def check(self):
        for ref in self.refs:
            require(self._stat(ref['path']) == self.baseline_stats[ref['path']], 'Authenticated source metadata changed during run: '+ref['path'])
        for path, baseline in self.baseline_dirs.items():
            require(self._dir_stat(path) == baseline, 'Authenticated source ancestor changed during run: '+path)
        while True:
            try: data = os.read(self.fd, 1024*1024)
            except BlockingIOError: break
            if not data: break
            pos = 0
            while pos < len(data):
                wd, mask, cookie, size = struct.unpack_from('iIII', data, pos); pos += 16
                name = data[pos:pos+size].split(b'\0', 1)[0].decode(errors='replace'); pos += size
                require(not (mask & (0x4000 | 0x8000)), 'Source watch overflow/loss; run rejected')
                if wd in self.inode_wds:
                    raise ValueError('Authenticated source inode changed during run')
                require(wd in self.names, 'Unknown source watch event; run rejected')
                require(name not in self.names[wd] and not (mask & (0x400 | 0x800)),
                    'Source/evidence changed during run: '+self.paths[wd]+'/'+name)
        for ref in self.refs: _verify(ref)
    def close(self):
        if self.fd >= 0: os.close(self.fd); self.fd = -1
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def prepare(campaign_path, output):
    campaign = audit_campaign(campaign_path); output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    attempts = [dict(plan=ref, attempt_id=checked(ref)['attempt_id'], execution_id=checked(ref)['execution_id'],
        collection=str(Path(ref['path']).parent/'collection'),
        claim=str(Path(ref['path']).parent/'source-rebound-attempt-claim.json')) for ref in campaign['attempts']]
    require(all(not Path(a['collection']).exists() and not Path(a['claim']).exists() for a in attempts),
        'No existing/failed/retried rebound attempt may enter a new execution manifest')
    manifest = dict(schema=CONT_SCHEMA, status='CPU_PREPARED_EXPLICIT_EXECUTE_REQUIRED', campaign=receipt(campaign_path),
        attempts=attempts, sources=strict_references([implementation_sources(__file__), evidence_sources(campaign)]),
        no_retries=True, GPU_started=False, numerical_fit_run=False, validation_labels_read=False,
        supervisor_budget=supervisor.BUDGET, preflight_free_bytes_required=supervisor.PREFLIGHT,
        raw_schema=RAW_SCHEMA, downstream_adapter_status='EXPLICIT_SOURCE_REBOUND_NATIVE_REVIEW_V1', automatic_fit=False)
    _write(output/'manifest.json', manifest); audit(output/'manifest.json'); return receipt(output/'manifest.json')


def audit(path):
    manifest = _read(path); campaign = audit_campaign(manifest['campaign']['path'])
    _verify(manifest['campaign'])
    require(manifest.get('schema') == CONT_SCHEMA and manifest.get('status') == 'CPU_PREPARED_EXPLICIT_EXECUTE_REQUIRED'
        and manifest.get('no_retries') is True and manifest.get('GPU_started') is False
        and manifest.get('numerical_fit_run') is False and manifest.get('validation_labels_read') is False
        and manifest.get('automatic_fit') is False and manifest.get('raw_schema') == RAW_SCHEMA
        and manifest.get('downstream_adapter_status') == 'EXPLICIT_SOURCE_REBOUND_NATIVE_REVIEW_V1'
        and manifest.get('supervisor_budget') == supervisor.BUDGET
        and manifest.get('preflight_free_bytes_required') == supervisor.PREFLIGHT, 'Exact source-rebound execution contract required')
    expected = [dict(plan=ref, attempt_id=checked(ref)['attempt_id'], execution_id=checked(ref)['execution_id'],
        collection=str(Path(ref['path']).parent/'collection'),
        claim=str(Path(ref['path']).parent/'source-rebound-attempt-claim.json')) for ref in campaign['attempts']]
    require(manifest['attempts'] == expected and manifest['sources'] == strict_references([implementation_sources(__file__), evidence_sources(campaign)]),
        'Exact order, canonical paths and complete executor closure required')
    for ref in manifest['sources']: _verify(ref)
    return manifest


def collect(plan_path,output,*,lease_fd,runtime_factory=DeploymentRuntime,monitor_factory=StartupBoardMonitor,
            clock_ns=time.perf_counter_ns,wait_until_ns=None,ownership_check=memory_ownership):
    supervisor.validate_lease(lease_fd);plan=audit_plan(plan_path);output=Path(output)
    require(not (output/'raw.json').exists(),'No overwrite/retry of a collection attempt')
    context=strict.checked(plan['context']);pool={r['pool_id']:r for r in (json.loads(l) for l in Path(plan['prepared_pool']['path']).read_text().splitlines())}
    registry={}
    for alias in OWNERS:
        family=alias.split('#')[0];binding=copy.deepcopy(context['worker_bindings'][family])
        binding['command'][1]=str(Path(__file__).with_name('semantic_progress_worker_v1.py'))
        binding['env'].update(context['threads']);binding['env']['PYTHONUNBUFFERED']='1';registry[alias]=binding
    runtime=runtime_factory(runtime_config(),registry);monitor=None
    executed=dict(attempt_id=plan['attempt_id'],execution_id=plan['execution_id'],cases=plan['cases'])
    atomic_save(output/'executed-cases.json',executed)
    injected=(runtime_factory is not DeploymentRuntime or monitor_factory is not StartupBoardMonitor
        or clock_ns is not time.perf_counter_ns or wait_until_ns is not None or ownership_check is not memory_ownership)
    raw=dict(schema=RAW_SCHEMA,collector_schema=RAW_SCHEMA,kind='CPU_TEST_ONLY' if injected else 'measured',
        backend='CPU_TEST_ONLY' if injected else 'native',split='training',
        status='starting',execution_id=plan['execution_id'],execution_context_sha256=plan['execution_context_sha256'],
        capture_id=plan['capture_id'],attempt_id=plan['attempt_id'],plan=receipt(plan_path),context=plan['context'],
        prepared_pool=plan['prepared_pool'],executed_cases=receipt(output/'executed-cases.json'),
        request_records=initial_rows(plan,pool),priming_request_records=[],
        public_case_calendars=copy.deepcopy(plan['cases']),priming_public_calendar=copy.deepcopy(plan['priming']),
        cases=[dict(case_id=c['case_id'],status='NOT_STARTED',offered=c['offered'],public_arrivals_sha256=c['public_arrivals_sha256'],request_ids=[r['request_id'] for r in c['arrivals']]) for c in plan['cases']],
        case_histories=[],case_preparations=[],measurement_windows=[],board_samples=[],telemetry_errors=[],errors=[],runtime_errors=[],lifecycle_boundaries=[],
        no_mutating_checkpoint_during_case=True,
        checkpoint_spans_file=ACTIVITY_FILE,checkpoint_spans_schema=ACTIVITY_SCHEMA,
        checkpoint_spans_execution_id=plan['execution_id'],checkpoint_time_domain=TIME_DOMAIN,
        checkpoint_context=dict(kind='before_startup',case_id=None),
        monitoring_scope=MONITORING_SCOPE,shutdown_memory_sampled=False,
        owned_peak_scope='sampled_live_declared_launcher_descendants_across_PGID_startup_through_final_drain_before_worker_close_not_instantaneous',
        performance_scope='unprofiled_original_sender_stdin_FIFO_with_semantic_phase_markers',
        source_bindings=plan['sources'],source_freeze=plan['source_freeze'],
        U=None,main_queue_provider_eligible=False,native_CUDA_API_numeric_authentication='UNQUALIFIED',
        training_feature_extraction_eligible=False,all_offered_registered_before_startup=True,
        source_rebound_executor_sources=implementation_sources(__file__))
    activity=ActivityLedger(output,plan['execution_id'],clock_ns)
    def checkpoint():
        context=raw['checkpoint_context'];now=clock_ns()
        if any(w['start_ns']<=now<=w['fixed_drain_deadline_ns'] for w in raw['measurement_windows']):
            # Interrupted/failed runs still retain their evidence after original
            # cleanup, but can never receive a successful timing-policy claim.
            raw['no_mutating_checkpoint_during_case']=False
        kwargs=dict(case_id=context['case_id'],phase=getattr(monitor,'phase','before_startup'),stage=context['kind'])
        with activity.span('checkpoint',**kwargs):
            with activity.span('checkpoint_locked_snapshot',**kwargs):
                with runtime.lock:
                    if monitor is not None:
                        with monitor._sample_lock:
                            raw['board_samples']=copy.deepcopy(monitor.samples);raw['telemetry_errors']=copy.deepcopy(monitor.errors)
                    raw['runtime_errors']=copy.deepcopy(runtime.errors)
                    snapshot=copy.deepcopy(raw)
            with activity.span('checkpoint_serialization_and_atomic_save',**kwargs):
                atomic_save(output/'raw.json',snapshot)
        activity.persist()
    def wait(target):
        if wait_until_ns is not None:return wait_until_ns(target)
        return wait_until(target,clock_ns,monitor)
    try:
        checkpoint();monitor=monitor_factory(runtime);monitor.sample();monitor.check();monitor.start()
        raw['lifecycle_boundaries'].append(dict(boundary='startup_begin',time_ns=clock_ns()));runtime.start()
        raw['ready_ownership_evidence']={}
        for alias in OWNERS:
            pid=runtime.ready[alias].get('pid');require(type(pid) is int,'Owned live READY pid required')
            evidence=ownership_check({pid:0.},runtime,monitor.collector_pid,monitor.process_group,ownership=monitor.ownership)
            require(evidence['owned_aliases']==[alias] and not evidence['unresolved_nvml_pids'] and not evidence['foreign_pids'],'Uncertified READY owner')
            raw['ready_ownership_evidence'][alias]=evidence['ownership_evidence']
        with runtime.lock:raw['ready']=copy.deepcopy(runtime.ready)
        raw['lifecycle_boundaries'].append(dict(boundary='startup_ready',time_ns=clock_ns()));monitor.set_phase('priming')
        raw['lifecycle_boundaries'].append(dict(boundary='priming_begin',time_ns=clock_ns()))
        for index,item in enumerate(plan['priming']):
            row=record('priming',index,item['task'],item['owner'],pool[item['pool_id']],clock_ns())
            row.update(request_id=plan['attempt_id']+':priming:'+str(index),profile_job_id='priming',
                split='training',validation_view='training')
            with runtime.lock:raw['priming_request_records'].append(row)
            runtime.enqueue(row,pool[item['pool_id']]['sample']);runtime.drain([row],300)
            require(row['status']=='completed','Original priming failed')
        raw['lifecycle_boundaries'].append(dict(boundary='priming_end',time_ns=clock_ns()))
        raw['checkpoint_context']=dict(kind='postpriming',case_id=None);checkpoint()
        for case in plan['cases']:
            result=dispatch_case(plan,case,runtime,pool,raw,clock_ns=clock_ns,wait_until_ns=wait,
                checkpoint=checkpoint,monitor=monitor,activity=activity)
            if result['status']!='completed':raise RuntimeError('Incomplete case; subsequent cases retained NOT_STARTED')
        audit_activity(raw,activity)
        require(raw['no_mutating_checkpoint_during_case'],'Checkpoint crossed a protected case interval')
        raw['status']='completed'
        raw['observation_end_ns']=clock_ns()
        raw['lifecycle_boundaries'].append(dict(boundary='measurement_and_drain_end',time_ns=raw['observation_end_ns']))
    except BaseException as exc:
        raw['status']='failed';raw['errors'].append(type(exc).__name__+': '+str(exc))
    finally:
        # Shared reviewed cleanup stops monitoring and takes its final sample
        # while workers are live, then closes workers. Final save is atomic.
        finish_collection(runtime,monitor,raw,checkpoint,clock_ns)
    return raw



def evidence_sources(campaign):
    plans = [checked(ref) for ref in campaign['attempts']]
    context = checked(campaign['context'])
    media = []
    for row in strict.source_catalog(context).values():
        if row['task_type'] != 'text': media.append(receipt(row['sample']['input']))
    return strict_references([campaign, plans, context, media, receipt(strict.BASE), receipt(strict.ENTRY)])


def execute(plan_path, output, *, manifest_path):
    manifest_ref = receipt(manifest_path); manifest = audit(manifest_path)
    plan_ref = receipt(plan_path); plan = audit_plan(plan_path)
    item = next(a for a in manifest['attempts'] if a['plan'] == plan_ref)
    require(Path(output).resolve() == Path(item['collection']), 'Canonical new-schema collection path required')
    require(not Path(output).exists(), 'Attempt already started; no retry of success/failure/unfinished collection')
    require(not Path(item['claim']).exists(), 'Attempt already claimed; no retry even if launch failed')
    supervisor.public_environment_gate()
    # Persist the no-retry claim before invoking the outer supervisor. A
    # prelaunch crash remains a visible HOLD, never a replacement attempt.
    _write(item['claim'], dict(schema=SCHEMA+'.claim', manifest=manifest_ref, plan=plan_ref,
        execution_id=plan['execution_id'], attempt_id=plan['attempt_id'], no_retries=True))
    command = [sys.executable, '-m', 'tools.predictor_validation_20261027.twoFIFO_source_rebound_execute_v1',
        '_child', '--manifest', str(Path(manifest_path).resolve()), '--plan', str(Path(plan_path).resolve()),
        '--output', str(Path(output).resolve()), '--execute']
    with SourceChangeGuard([manifest_ref, manifest['sources']]) as guard:
        def gate():
            guard.check(); _verify(manifest_ref); audit(manifest_path); audit_plan(plan_path)
            supervisor.public_environment_gate(); guard.check()
        return supervisor.supervise(command, output,
            environment={'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'},
            before_launch=gate, after_run=gate, metadata=dict(schema=SCHEMA+'.execution_binding',
                manifest=manifest_ref, plan=plan_ref, execution_id=plan['execution_id'], attempt_id=plan['attempt_id'],
                raw_schema=RAW_SCHEMA, downstream_adapter_required=True, automatic_fit=False))


def completed_execution(execution, manifest_ref, item):
    require(execution.get('profile_execution') == dict(schema=SCHEMA+'.execution_binding', manifest=manifest_ref,
        plan=item['plan'], execution_id=item['execution_id'], attempt_id=item['attempt_id'],
        raw_schema=RAW_SCHEMA, downstream_adapter_required=True, automatic_fit=False)
        and execution.get('budget') == supervisor.BUDGET
        and execution.get('status') == 'CHILD_EXITED_REQUIRES_PROFILE_REVIEW'
        and execution.get('child_reaped') is True and execution.get('live_group_pids') == []
        and execution.get('experiment_lock_held_through_group_exit') is True
        and execution.get('all_owned_descendants_exited_and_reaped') is True
        and execution.get('final_source_audit') == 'PASS'
        and execution.get('watchdog', {}).get('status') == 'EXITED'
        and execution['watchdog'].get('return_code') == 0,
        'Failed/incomplete/old/foreign execution receipt rejected; no retry')


def run(manifest_path, output, *, execute_enabled=False):
    """Review and rederive macros before every next attempt; never fit."""
    from . import twoFIFO_source_rebound_review_v1 as native_review
    from . import twoFIFO_source_rebound_macros_v1 as macros
    from .twoFIFO_observed_resources_v2 import observed as observed_resources
    manifest_ref = receipt(manifest_path); manifest = audit(manifest_path)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    lock = Path(manifest['campaign']['path']).parent/'source-rebound-continuation.lock'
    with lock.open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = dict(schema=CONT_SCHEMA+'.run', manifest=manifest_ref, no_retries=True,
            execute_invocations=0, numerical_fit_run=False, automatic_fit=False, validation_labels_read=False,
            completed_attempts=[])
        try:
            for item in manifest['attempts']:
                _verify(manifest_ref); audit(manifest_path)
                result['attempt_id'] = item['attempt_id']
                collection = Path(item['collection']); final = collection/'execution.json'
                if collection.exists() and not final.is_file():
                    result['status'] = 'WAITING_EXISTING_ATTEMPT_FINAL_RECEIPT'; break
                if not collection.exists() and Path(item['claim']).exists():
                    result['status'] = 'HOLD_CLAIMED_ATTEMPT_NO_RETRY'; break
                if not collection.exists() and not execute_enabled:
                    result['status'] = 'READY_EXPLICIT_EXECUTE_REQUIRED'; break
                if not collection.exists():
                    result['execute_invocations'] += 1
                    execute(item['plan']['path'], collection, manifest_path=manifest_path)
                require(final.is_file(), 'Executor returned without final receipt; no retry')
                execution_ref = receipt(final); completed_execution(checked(execution_ref), manifest_ref, item)
                raw_ref = receipt(collection/'raw.json'); raw = checked(raw_ref)
                require(raw.get('schema') == RAW_SCHEMA and raw.get('collector_schema') == RAW_SCHEMA
                    and raw.get('plan') == item['plan'] and raw.get('execution_id') == item['execution_id'],
                    'Exact new-schema raw binding required; old v3 capture is not rebound evidence')
                check_dir = output/item['attempt_id']; check_dir.mkdir()
                review_ref = native_review.review(raw_ref['path'], check_dir/'native-review', final,
                    manifest_path=manifest_path)
                reviewed = checked(review_ref)
                require(reviewed.get('status') == 'PASS_COMPLETE_SOURCE_REBOUND_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION'
                    and reviewed.get('schema') == native_review.SCHEMA
                    and reviewed.get('source_rebound_manifest') == manifest_ref
                    and reviewed.get('training_feature_extraction_eligible') is True and reviewed.get('issues') == [],
                    'New native source-rebound review NOT_ELIGIBLE; stop all following attempts')
                facts = observed_resources(raw)
                _write(check_dir/'observed-resources.json', facts)
                require(facts['startup']['energy_joules'] is not None and all(
                    r['measurement_energy_joules'] is not None for r in facts['cases']),
                    'Original power-gap gate: startup and every measurement energy must be observable')
                macro_doc = macros.extract(review_ref['path'])
                require(macro_doc['source_rebound_manifest'] == manifest_ref,
                    'Macro source rebound binding drift')
                _write(check_dir/'macros-v1.json', macro_doc)
                _verify(manifest_ref); audit(manifest_path); _verify(execution_ref); _verify(raw_ref)
                result['completed_attempts'].append(dict(attempt_id=item['attempt_id'], execution=execution_ref,
                    training_review=review_ref, macros=receipt(check_dir/'macros-v1.json')))
            else:
                result.update(status='COMPLETE_ALL_TWELVE_REBOUND_NATIVE_ATTEMPTS_REVIEWED_NO_FIT',
                    next_gate='NEW_SOURCE_BOUND_INTAKE_AND_COORDINATOR_REQUIRED', fit_eligible=False)
        except BaseException as exc:
            result.update(status='STOPPED_NO_RETRY', error=type(exc).__name__+': '+str(exc))
        _write(output/'report.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='op', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--campaign', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('audit'); p.add_argument('--manifest', required=True)
    p = sub.add_parser('run'); p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
    p.add_argument('--execute', action='store_true')
    for op in ('execute', '_child'):
        p = sub.add_parser(op); p.add_argument('--manifest', required=True); p.add_argument('--plan', required=True)
        p.add_argument('--output', required=True); p.add_argument('--execute', required=True, action='store_true')
    a = parser.parse_args()
    if a.op == 'prepare': result = prepare(a.campaign, a.output)
    elif a.op == 'audit': result = audit(a.manifest)
    elif a.op == 'run': result = run(a.manifest, a.output, execute_enabled=a.execute)
    elif a.op == 'execute': result = execute(a.plan, a.output, manifest_path=a.manifest)
    else:
        manifest = audit(a.manifest); plan = audit_plan(a.plan)
        require(any(i['plan'] == receipt(a.plan) and i['collection'] == str(Path(a.output).resolve())
            for i in manifest['attempts']), 'Child canonical manifest/plan/output binding required')
        claim = _read(Path(a.plan).parent/'source-rebound-attempt-claim.json')
        require(claim == dict(schema=SCHEMA+'.claim', manifest=receipt(a.manifest), plan=receipt(a.plan),
            execution_id=plan['execution_id'], attempt_id=plan['attempt_id'], no_retries=True), 'Exact exclusive attempt claim required')
        result = collect(a.plan, a.output, lease_fd=int(os.environ.get(supervisor.LEASE_FD, '-1')))
    print(json.dumps(result))
    return 1 if result.get('status') in ('failed', 'FAILED_NO_RETRY', 'STOPPED_NO_RETRY') else 0


if __name__ == '__main__': raise SystemExit(main())
