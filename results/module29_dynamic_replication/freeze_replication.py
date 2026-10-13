from common_replication import *
import subprocess, platform
from module21.trajectory import arrivals, choose
from joint_deploy.measurement.load_generator import SamplePool
from replica_scaling.telemetry import snapshot

def main():
    assert not (OUT/'frozen_replication_manifest.json').exists()
    save('historical_metadata_before.json',metadata())
    dev=ROOT/'results/module28r_dynamic_followup'
    original=json.loads((dev/'frozen_candidate_manifest.json').read_text())
    candidate=json.loads((dev/'frozen_positive_candidate.json').read_text())
    inventory=[]
    # Read the complete development output, including durable journals and raw runs.
    for p in sorted(dev.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:continue
        inventory.append(ref(p))
        if p.suffix=='.json':json.loads(p.read_text())
        elif p.suffix=='.jsonl':
            for line in p.read_text().splitlines():json.loads(line)
        elif p.suffix=='.csv':
            csv.field_size_limit(128*1024*1024)
            list(csv.DictReader(p.open()))
    preserved=original['implementation']+original['reused_implementation']+[original[k] for k in ['catalog','utility_definition','utility_reference','quality_policy']]
    for r in preserved:assert ref(r['path'])['sha256']==r['sha256'],r['path']
    assert ref(OUT/'dynamic_router.py')['sha256']==candidate['router_source']['sha256']
    gpu=snapshot()
    assert not gpu['processes'] and gpu['used_mib']<=128
    for k in ['uuid','name','driver','total_mib']:assert gpu[k]==original['GPU'][k],k
    rates=candidate['candidate']['lambda_rps']
    assert rates['lid']==10.5 and all(q==.2 for t,q in rates.items() if t!='lid')
    # Consecutive new identifiers fixed once; no seed selection from GPU outcomes.
    seeds=[291001,291002,291003]
    used=set();seed_sources=[]
    def scan(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in ['seed','random_seed'] and isinstance(v,int):used.add(v)
                if k in ['new_seeds','seeds'] and isinstance(v,list):used.update(x for x in v if isinstance(x,int))
                scan(v)
        elif isinstance(value,list):
            for v in value:scan(v)
    for root in [ROOT/'results/module27_positive_split_discovery',OLD,dev]:
        for p in root.rglob('*.json'):
            if '__pycache__' in p.parts:continue
            scan(json.loads(p.read_text()));seed_sources.append(ref(p))
    assert not set(seeds)&used
    f=copy.deepcopy(original)
    f.update(module='29',at_unix_ns=time.time_ns(),GPU=gpu,rates=rates,
        catalog=ref(OUT/'selected_pool_catalog.json'),development_candidate=ref(dev/'frozen_positive_candidate.json'),
        maximum_total_attempts=9,maximum_formal_gpu_seconds=21600,independent_slots=[],
        maximum_validation_attempts=9,maximum_exploration_attempts=0,exploration_slots=[],
        failure_policy='ZERO GPU retries/replacements. Stop on unsafe/technical failure, invalid cohort/admission/memory, unclean release, system or worker unstable, or failed thermal pairing. Negative utility alone does not stop the nine registered runs.',
        optimizer_modified=False,next_module_automatic=False)
    f['plan']['GPU']=gpu
    f['safety']['campaign_policy']=f['failure_policy']
    f['selection']=dict(delta_over=.01,min_task_count=72,thermal_match_c=5.,all_three_legal_stable=True,both_dynamic_paths_nonzero=True)
    f['replication_protocol']=dict(seeds=seeds,groups=3,measurement_seconds=2400.,warmup_seconds=30.,drain_timeout_seconds=60.,
        original_thresholds=DEFAULT_THRESHOLDS,paired_t_df=2,paired_t95_lower_over_zero=True,
        all_six_deltas_over=.01,all_nine_legal_stable=True,all_dynamic_both_paths=True,
        negative_outcomes_never_repeated=True,development_excluded_from_statistics=True,
        gate_rule='Invalid/incomplete/unsafe => REPLICATION_INVALID. Any delta<=.01 or missing G/S => DYNAMIC_NOT_REPRODUCED. All six>.01 with either t95 CI crossing zero => DYNAMIC_POSITIVE_INCONCLUSIVE; otherwise DYNAMIC_POSITIVE_REPLICATED. System mean/P95 consistency reported separately; no claim of uniform latency improvement without evidence.')
    cover=[];orders=[['s_only','fixed10','dynamic'],['dynamic','s_only','fixed10'],['fixed10','dynamic','s_only']]
    for group,seed in enumerate(seeds,1):
        traces={}
        for label,seconds in [('warmup',30.),('measurement',2400.)]:
            pool=SamplePool(f['plan']['sample_pool']['path'],list(rates),seed)
            events=arrivals(rates,seconds,seed,label)
            assert events==arrivals(rates,seconds,seed,label)
            for e in events:e['item']=pool.next(e['task'])
            counts=collections.Counter(e['task'] for e in events)
            assert set(counts)==set(rates)
            assert all(0<e['offset_seconds']<seconds for e in events)
            path=f'trajectories/group{group}_{label}.json'
            save(path,dict(seed=seed,label=label,rates=rates,duration=seconds,arrival_profile='steady',events=events))
            traces[label]=ref(OUT/path);cover.append(dict(group=group,seed=seed,label=label,counts=dict(counts),trace=traces[label]))
        for arm in orders[group-1]:
            slot=copy.deepcopy(candidate['candidate']);fraction=.1 if arm=='fixed10' else .05 if arm=='dynamic' else 0.
            rr=copy.deepcopy(slot['routing']);s=f['parameters']['S_worker'].rsplit('#',1)[0]
            rr['lid']={s:rates['lid']*(1-fraction)}
            if fraction:rr['lid']['qwen']=rates['lid']*fraction
            i=len(f['independent_slots'])+1
            slot.update(run_id=f'G{group}_{arm}',stage='D',scenario='LOW',arrival_profile='steady',point_id='LOW',
                seed=seed,repetition=group,group=group,measurement_seconds=2400.,routing=rr,
                arm=arm,G_fraction=fraction,trajectories=traces,run_order=i)
            f['independent_slots'].append(slot)
    # Record all local model files, including weight bytes, not just cache identifiers.
    weight_roots=[ROOT/'models/qwen2.5-omni-7b',ROOT/'models/module13b/openai--whisper-medium',
        ROOT/'models/candidate_pool_v2/roberta_sst2',ROOT/'models/candidate_pool_v2/convnext_tiny',ROOT/'models/module14b']
    weights=[]
    for root in weight_roots:
        assert root.is_dir(),root
        for p in sorted(root.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:weights.append(ref(p))
    save('model_files_sha256.json',weights)
    envs=[]
    for executable in sorted({v['python'] for v in f['plan']['registry'].values()}):
        version=subprocess.run([executable,'-B','-c','import sys,platform; print(sys.version); print(platform.platform())'],capture_output=True,text=True,check=True)
        packages=subprocess.run([executable,'-B','-m','pip','freeze','--all'],capture_output=True,text=True,check=True)
        envs.append(dict(python=executable,version=version.stdout,packages=packages.stdout,stderr=packages.stderr))
    save('software_environment.json',dict(platform=platform.platform(),uname=os.uname(),environments=envs,GPU=gpu))
    source_paths=[ROOT/'performance_model/local_calibration/stability_v2.py',ROOT/'module26/common.py',ROOT/'candidate_pool_audit/new_worker.py']
    source_paths += [p for p in (ROOT/'joint_deploy').rglob('*.py') if '__pycache__' not in p.parts]
    source_paths += [p for p in (ROOT/'benchmark/scripts').glob('*.py')]
    f['reused_implementation']=list({r['path']:r for r in preserved+[ref(p) for p in source_paths]}.values())
    f['implementation']=[ref(p) for p in sorted(OUT.glob('*.py'))]
    f['model_files']=ref(OUT/'model_files_sha256.json');f['environment']=ref(OUT/'software_environment.json')
    save('stage_a_source_inventory.json',dict(development_files=inventory,seed_audit_sources=seed_sources,
        prior_reserved_and_used_seeds=sorted(used),new_seeds=seeds,never_prior_used_or_reserved=True))
    f['development_inventory']=ref(OUT/'stage_a_source_inventory.json')
    save('generator_coverage.json',cover)
    save('frozen_replication_manifest.json',f)
    csvsave('gpu_run_manifest.csv',[dict(**slot,status='FROZEN_BEFORE_GPU',formal_seconds=2400.,warmup_seconds=30.,
        drain_seconds_max=60.,technical_retries=0,manifest=ref(OUT/'frozen_replication_manifest.json')) for slot in f['independent_slots']])
    print('FROZEN',[(s['run_id'],s['seed']) for s in f['independent_slots']],flush=True)

if __name__=='__main__':main()
