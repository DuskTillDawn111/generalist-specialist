"""Serial native development, frozen selection, and independent paired validation.

All outputs are append-only stage directories. Source/resource failures stop
the workflow and preserve evidence; no automatic retries or concurrent GPU use.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.queue_refined_20261023.protocol import receipt,digest,validate_packet,write
from tools.queue_refined_20261023.analysis import analyze

HERE=Path(__file__).resolve().parent
EXPECTED_GPU='GPU-8a262fe0-598c-46b7-c953-41d9da84a47e'


def publish(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def gpu_idle():
    uuid=subprocess.run(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True,timeout=5).stdout.strip()
    if uuid!=EXPECTED_GPU:raise RuntimeError('Different physical GPU')
    active=subprocess.run(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True,timeout=5).stdout.strip()
    if active:raise RuntimeError('GPU occupied; serial boundary failed')
    return dict(uuid=uuid,active_compute_pids=[],unix_ns=time.time_ns())


def resources_valid(raw):
    if raw.get('preflight',{}).get('gpu',{}).get('uuid')!=EXPECTED_GPU:raise RuntimeError('Raw GPU identity mismatch')
    memory=raw.get('memory_evidence',{});energy=raw.get('energy_evidence',{})
    if not memory.get('measured') or not memory.get('resource_valid'):raise RuntimeError('Memory evidence invalid')
    if not energy.get('coverage_valid'):raise RuntimeError('Energy coverage invalid')
    if raw['status'] not in {'completed','completed_with_failures'}:raise RuntimeError('Native run failed: '+str(raw.get('errors')))


def select_validation_jobs(packet,report,seeds):
    """Freeze the best and strongest distinct alternative before validation."""
    ranked=[r for r in report['ranking'] if r['complete_eligible_all_declared_seeds']]
    if len(ranked)<2:raise ValueError('Fewer than two fully eligible development candidates; selection unresolved')
    def mixed(row):return row['candidate']['strategy'] in {'queue_quality','queue_latency'}
    alternative=next((row for row in ranked[1:] if mixed(row)!=mixed(ranked[0])),None)
    if alternative is None:raise ValueError('No eligible deployment/routing control from the other routing class')
    selected=[ranked[0],alternative];templates={digest(dict(deployment=j['policy_contract']['deployment'],strategy=j['strategy'],
        policy_contract_sha256=digest(j['policy_contract']))):j for j in packet['jobs']}
    ranks={row['candidate_id']:i+1 for i,row in enumerate(report['ranking'])}
    jobs=[]
    for index,seed in enumerate(seeds):
        order=selected if index%2==0 else list(reversed(selected))
        for row in order:
            job=copy.deepcopy(templates[row['candidate_id']]);job['seed']=seed
            job['label']=job['label'].rsplit('-seed',1)[0]+f'-seed{seed}'
            job['phase']='independent_validation' if seed in (81,82) else 'final_confirmation'
            claim='INDEPENDENT_FIXED_SELECTION_NATIVE_VALIDATION' if seed in (81,82) else 'FINAL_FIXED_SELECTION_NATIVE_CONFIRMATION'
            job['identity'].update(seed=seed,measurement_claim=claim,phase=job['phase'],
                development_rank=ranks[row['candidate_id']],
                selection_role='winner' if row==selected[0] else 'strongest_other_class_control',
                selection_report_sha256=digest(report))
            job['id']=digest(job['identity'])
            jobs.append(job)
    validation=copy.deepcopy(packet);validation.update(jobs=jobs,seeds=list(seeds),
        status='PREDECLARED_SELECTION_FROZEN_BEFORE_VALIDATION',GPU_started=False,
        selection=dict(candidates=selected,rule='Best mean original six-term U and strongest alternative from the other routing class (queue mixed versus static/specialist-only); require every declared development seed eligible.',
            validation_seeds=list(seeds),order='Alternating best/alternative and alternative/best by seed',
            no_reselection_on_validation=True,development_report_digest=digest(report)))
    return validation


def execute(packet_path,output,*,wait_progress=None,validation_seeds=(81,82,71,72,73),declaration=None):
    packet_path=Path(packet_path).resolve();packet=json.loads(packet_path.read_text())
    if not validate_packet(packet)['valid']:raise ValueError('Invalid immutable development packet')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    bindings=[receipt(packet_path)]+[receipt(HERE/name) for name in ['workflow.py','analysis.py','search.py']]
    if declaration is not None:
        declared=json.loads(Path(declaration).read_text())
        declared_seeds=declared['independent_validation']['seeds']+declared['final_confirmation']['seeds']
        if declared_seeds!=list(validation_seeds):raise ValueError('Validation seeds differ from predeclaration')
        if declared['development']['jobs']!=len(packet['jobs']):raise ValueError('Development job count differs from predeclaration')
        bindings.append(receipt(declaration))
    state=dict(schema='gsdeploy.queue_aware_workflow.v1',status='RUNNING',phase='WAIT_EXISTING_GPU_BATCH' if wait_progress else 'DEVELOPMENT',
        PID=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),bindings=bindings,stages=[],
        validation_seeds=list(validation_seeds),no_retries=True,no_reselection_on_validation=True)
    progress=output/'progress.json';publish(progress,state);child=None
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    def verify():
        if any(receipt(r['path'])!=r for r in bindings):raise ValueError('Workflow source changed during execution')
    def pulse():
        state['heartbeat_unix_ns']=time.time_ns();publish(progress,state)
    def stage(path,label):
        nonlocal child
        registered=json.loads(path.read_text());target=output/label;target.mkdir(exist_ok=False)
        status=dict(label=label,packet=receipt(path),jobs=[],status='RUNNING');state['stages'].append(status);state['phase']=label;pulse()
        raws=[]
        for job in registered['jobs']:
            verify()
            if receipt(path)!=status['packet']:raise ValueError('Frozen stage packet changed on disk')
            if not validate_packet(registered)['valid']:raise ValueError('Native packet binding changed')
            boundary=gpu_idle();jobout=target/job['label'];log=target/(job['label']+'.log')
            command=[sys.executable,'-u',str(HERE/'native_runner.py'),'--packet',str(path),'--job',job['id'],
                '--output',str(jobout),'--execute']
            state['current']=dict(label=job['label'],stage=label,output=str(jobout),gpu_boundary=boundary,started_unix_ns=time.time_ns())
            with log.open('x') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state['current']['pid']=child.pid;pulse()
                while child.poll() is None:pulse();time.sleep(5)
            rawpath=jobout/'run.json'
            if not rawpath.exists():raise RuntimeError('Missing terminal raw; inspect '+str(log))
            raw=json.loads(rawpath.read_text());utility=raw.get('evaluation',{}).get('utility',{})
            row=dict(job_id=job['id'],label=job['label'],raw=receipt(rawpath),exit_code=child.returncode,
                status=raw['status'],eligible=utility.get('eligible'),U=utility.get('U'),errors=raw.get('errors'))
            status['jobs'].append(row);raws.append(rawpath);pulse()
            if raw['experiment_job_id']!=job['id']:raise RuntimeError('Wrong terminal job identity')
            resources_valid(raw);gpu_idle()
        state.pop('current',None);status['status']='COLLECTED';pulse()
        report=analyze(raws,registered['seeds'],target/'analysis')
        from tools.queue_refined_20261023.search import measured_table_search
        table=json.loads((target/'analysis/measured-search-table.json').read_text())
        dfs=measured_table_search(table,'plain_dfs');bounded=measured_table_search(table,'bounded')
        if dfs['winner_id']!=bounded['winner_id'] or dfs['objective']!=bounded['objective']:
            raise RuntimeError('Measured candidate table optimizer/DFS mismatch')
        winner=report['winner']
        if ((winner is None)!=(dfs['winner_id'] is None)
            or (winner is not None and (dfs['winner_id']!=winner['candidate_id']
                or not math.isclose(dfs['objective'],winner['mean_U'],rel_tol=0,abs_tol=1e-12)))):
            raise RuntimeError('Native analysis ranking and measured-table DFS mismatch')
        write(target/'analysis/measured-dfs-comparison.json',dict(plain_dfs=dfs,bounded=bounded,
            same_winner_and_value=True,scope='Saved measured table lookups; not a GPU runtime speedup benchmark'))
        status.update(status='COMPLETE_AUDITED',report=receipt(target/'analysis/report.json'));pulse()
        return report
    def signal_handler(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,signal_handler)
    try:
        if wait_progress:
            while True:
                verify();previous=json.loads(Path(wait_progress).read_text());state['waiting_for']=dict(path=str(wait_progress),status=previous['status']);pulse()
                if previous['status'].startswith('COMPLETE'):break
                if previous['status']!='RUNNING':raise RuntimeError('Previous GPU batch stopped: '+previous['status'])
                time.sleep(5)
            state.pop('waiting_for',None)
        development=stage(packet_path,'development')
        frozen=select_validation_jobs(packet,development,list(validation_seeds))
        validation_path=output/'frozen-validation-packet.json';write(validation_path,frozen)
        state['frozen_selection']=receipt(validation_path);pulse()
        stage_packets=[]
        for label,seeds in [('independent-validation',[81,82]),('final-confirmation',[71,72,73])]:
            subset=copy.deepcopy(frozen);subset['seeds']=seeds
            subset['jobs']=[j for j in frozen['jobs'] if j['seed'] in seeds]
            path=output/(label+'-packet.json');write(path,subset);stage_packets.append((label,path))
        # Both phase packets are frozen before either reserved phase begins.
        state['reserved_phase_packets']=[receipt(p) for _,p in stage_packets];pulse()
        phase_reports={}
        for label,path in stage_packets:
            if receipt(path) not in state['reserved_phase_packets']:raise ValueError('Reserved phase packet changed')
            phase_reports[label]=stage(path,label)
        raw_paths=[Path(j['raw']['path']) for s in state['stages'][1:] for j in s['jobs']]
        validation=analyze(raw_paths,list(validation_seeds),output/'all-independent-pairs-analysis')
        selection=frozen['selection'];result=dict(status='COMPLETE_NATIVE_COMPARISON_AND_FIXED_SELECTION_VALIDATION',
            development_winner=development['winner'],validation_ranking=phase_reports['independent-validation']['ranking'],
            confirmation_ranking=phase_reports['final-confirmation']['ranking'],combined_ranking=validation['ranking'],
            selection=selection,validation_comparisons=phase_reports['independent-validation']['comparisons'],
            confirmation_comparisons=phase_reports['final-confirmation']['comparisons'],combined_comparisons=validation['comparisons'],
            all_validation_outcomes_eligible=validation['full_declared_table_complete'],
            no_reselection_on_validation=True,physical_full_pool_optimality=False,
            shared_gpu_interference='Measured natively; effect not assumed from CPU replay')
        write(output/'result.json',result);state.update(status='COMPLETE',phase='COMPLETE',result=receipt(output/'result.json'))
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        state.update(status='STOPPED_FOR_DIAGNOSIS_NO_RETRY',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        return 1
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        state['updated_utc']=datetime.now(timezone.utc).isoformat();publish(progress,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--output',type=Path);parser.add_argument('--wait-progress',type=Path);parser.add_argument('--protocol',type=Path);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:print(json.dumps(validate_packet(json.loads(args.packet.read_text()))))
    else:
        if args.output is None:parser.error('--output required for execution')
        raise SystemExit(execute(args.packet,args.output,wait_progress=args.wait_progress,declaration=args.protocol))
