"""Enable synthesis retirement on the retained Baichuan configuration and verify.

This is a before/after equivalence replay, never new independent QA evidence.
Frozen source trees and previous results are only read.
"""
import argparse
import copy
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/baichuan-no-synthesis-20261021'
QA=ROOT/'artifacts/qa-development-revision-20261020/tydiqa-new-native-confirmation80-v2'
SOURCE=QA/'source'
PRIOR=ROOT/'artifacts/confirmation-release-stage-20261020'
PYTHON='/root/autodl-tmp/gsdeploy-replacement-assets/envs/baichuan-native/bin/python'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def prepare():
    OUT.mkdir(exist_ok=False)
    release=read(QA/'release.json')
    for path,digest in release['sources'].items():assert sha(path)==digest,path
    bindings={str(p):sha(p) for p in SOURCE.glob('gsdeploy/**/*.py')}
    jobs=[]
    for task in ['text','image','ocr','asr']:
        if task=='text':
            old=QA/'qa80-generalist_1'
            job=read(old/'job.json');spec=read(old/'worker.json')
            baseline=old/'results'
        else:
            old=PRIOR/'retirement-packets/calibration'/f'baichuan-retired-synthesis-{task}-calibration'
            # Use the original unretired baseline for exact output comparison.
            job=read(old/'job.json')
            spec=read(job['spec'])
            baseline=ROOT/'artifacts/supervision-20261019T131816Z'/f'baichuan-combined-{task}-calibration-results'/f'baichuan-combined-{task}-calibration'
        folder=OUT/task
        folder.mkdir()
        before=copy.deepcopy(spec['execution'])
        before.setdefault('memory_optimizations',{}).pop('retire_audio_output',None)
        # Keep the current cache policy identical to the retained unretired run.
        spec['execution']['cache_release_threshold_mib']=512
        spec['execution'].setdefault('memory_optimizations',{})['retire_audio_output']=True
        spec['candidate']['precision']='Synthesis-only retirement; '+sha(Path(job['spec']))
        spec['id']='baichuan-no-synthesis-'+task
        spec['resources']['request_output']=str(folder/'native-requests')
        save(folder/'worker.json',spec)
        job.update(id=spec['id'],spec=str(folder/'worker.json'),python=PYTHON,
            phase='quality',rounds=1,replicas=1,minimum_seconds=0,
            minimum_requests=len(job['samples']),exact_cohort=True)
        # Diagnostic parent protocols remain diagnostic; never promote them to formal evidence.
        if read(spec['resources']['protocol']).get('diagnostic_only'):job['phase']='diagnostic'
        save(folder/'job.json',job)
        baseline_records=baseline/(task+'-round0.jsonl')
        assert baseline_records.is_file(),baseline_records
        jobs.append(dict(task=task,count=len(job['samples']),folder=str(folder),
            baseline=str(baseline),baseline_raw_sha256=sha(baseline_records),
            spec_sha256=sha(folder/'worker.json'),job_sha256=sha(folder/'job.json')))
    save(OUT/'release.json',dict(scope='SYNTHESIS_RETIREMENT_OUTPUT_EQUIVALENCE_NOT_NEW_CONFIRMATION',
        frozen_utc=datetime.now(timezone.utc).isoformat(),source_root=str(SOURCE),
        source_hashes=bindings,jobs=jobs,script_sha256=sha(__file__),
        removed=['audio_head','audio_tokenizer.audio_decoder','audio_tokenizer.audio_flow_matching_decoder'],
        retained='language, vision, complete audio input encoder and bridge',pool_admitted=False))
    print('Frozen',sum(j['count'] for j in jobs),'four-task equivalence requests',flush=True)

def collect(task):
    release=read(OUT/'release.json')
    for path,digest in release['source_hashes'].items():assert sha(path)==digest,path
    entry=next(e for e in release['jobs'] if e['task']==task)
    folder=Path(entry['folder'])
    assert sha(folder/'worker.json')==entry['spec_sha256']
    assert sha(folder/'job.json')==entry['job_sha256']
    sys.path.insert(0,str(SOURCE))
    sys.path.append('/root/autodl-tmp/gsdeploy-assets/envs/profile-overlay')
    from gsdeploy.profiling.compact_experiments import profile_job
    result=profile_job(folder/'job.json',folder/'results')
    assert result['status']=='COMPLETE',result.get('error')

def run():
    release=read(OUT/'release.json')
    state=dict(status='RUNNING_EQUIVALENCE',pid=os.getpid(),completed=[],pool_admitted=False)
    def update():
        state['utc']=datetime.now(timezone.utc).isoformat();save(OUT/'state.json',state)
    update()
    try:
        with (ROOT/'artifacts/.gpu-experiment.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for entry in release['jobs']:
                task=entry['task'];state['current']=task;update()
                env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',
                    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONPATH=str(SOURCE))
                env.pop('DS_ACCELERATOR',None)
                with (OUT/(task+'.log')).open('x') as log:
                    subprocess.run([PYTHON,'-u',str(Path(__file__).resolve()),'collect','--task',task],
                        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                folder=Path(entry['folder'])
                new=[json.loads(l) for l in (folder/f'results/{task}-round0.jsonl').open()]
                baseline=Path(entry['baseline'])
                oldpath=baseline/(task+'-round0.jsonl')
                assert sha(oldpath)==entry['baseline_raw_sha256']
                old={r['sample_id']:r for r in [json.loads(l) for l in oldpath.open()]}
                assert len(new)==len(old)==entry['count'] and {r['sample_id'] for r in new}==set(old)
                fields=['raw_output','prediction','success','finish_reason','generated_tokens','terminal_token_id','eos_token_ids']
                changes=[dict(id=r['sample_id'],fields=[k for k in fields if r.get(k)!=old[r['sample_id']].get(k)])
                    for r in new if any(r.get(k)!=old[r['sample_id']].get(k) for k in fields)]
                profile=read(folder/'results/profile.json');prior=read(baseline/'profile.json')
                audits=[r['memory_optimizations']['retired_audio_synthesis'] for r in new]
                assert all(a['retained_parameter_objects_unchanged'] for a in audits)
                comparison=dict(task=task,count=len(new),valid=sum(r['success'] for r in new),
                    changes=changes,unchanged=not changes,
                    old_resident_GiB=prior['resident_after_warmup_mib']/1024,
                    new_resident_GiB=profile['resident_after_warmup_mib']/1024,
                    old_peak_board_GiB=prior['peak_board_used_mib']/1024,
                    new_peak_board_GiB=profile['peak_board_used_mib']/1024,
                    retirement_audit=audits[0],raw_sha256=sha(folder/f'results/{task}-round0.jsonl'))
                save(folder/'comparison.json',comparison)
                state['completed'].append(task);update()
                assert not changes and comparison['valid']==entry['count'],'Output equivalence failed; preserve results'
        state.update(status='COMPLETE_FOUR_TASK_EQUIVALENCE',current=None);update()
    except Exception as exc:
        state.update(status='FAILED_PRESERVED',error=str(exc),traceback=traceback.format_exc());update();raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','collect']);parser.add_argument('--task')
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='run':run()
    else:collect(args.task)
