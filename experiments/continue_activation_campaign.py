"""Sequential native campaign continuation with durable receipts and no GPU overlap."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def save(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True,type=Path)
    args=p.parse_args();base=args.base.resolve();reference=base/'references-v2'
    status_path=base/'continuation-v1.json'
    if status_path.exists():
        status=json.loads(status_path.read_text())
        if status['status']!='waiting_for_reference_episodes' or status['completed']:raise FileExistsError(status_path)
        status['restart_while_waiting_unix']=time.time()
    else:status=dict(status='waiting_for_reference_episodes',started_unix=time.time(),completed=[],errors=[])
    save(status_path,status)
    wait_deadline=time.monotonic()+7200
    while not (reference/'episode-2/run.json').exists():
        if time.monotonic()>wait_deadline:raise TimeoutError('Reference episode completion wait budget exhausted')
        time.sleep(2)
    # The reference writer saves after clean runtime shutdown. Subsequent
    # collectors also take the same exclusive GPU lock before loading workers.
    for index in (1,2):
        raw=json.loads((reference/f'episode-{index}/run.json').read_text())
        if raw['status']!='completed' or raw.get('errors'):
            status.update(status='reference_capture_failed');save(status_path,status);return 1
    from gsdeploy.config import SystemConfig
    from gsdeploy.reference_freeze import freeze_references
    from gsdeploy.evaluation import evaluate_run
    from gsdeploy import normalization as norm
    try:
        config=SystemConfig.load(ROOT/'configs/system_reference_v2_20261103.json')
        manifest=json.loads((reference/'measurements-partial.json').read_text())
        manifest['episode_runs']=['episode-1/run.json','episode-2/run.json']
        save(reference/'measurements-complete.json',manifest)
        frozen=freeze_references(config,manifest,reference)
        frozen_path=reference/'normalization-frozen.json'
        if frozen_path.exists():raise FileExistsError(frozen_path)
        save(frozen_path,frozen)
        data=config.as_dict();data['catalog'][norm.KEY]=frozen
        data['latency_reference_ms']={t:r['deadline_ms'] for t,r in frozen['tasks'].items()}
        measured_config=SystemConfig(**data)
        save(reference/'system-frozen.json',measured_config.as_dict())
        evaluations={str(index):evaluate_run(measured_config,json.loads((reference/f'episode-{index}/run.json').read_text()))
                     for index in (1,2)}
        save(reference/'evaluations-frozen.json',evaluations)
        status['reference_freeze']='passed'
    except Exception as exc:
        # Capacity and input-isolated predictor measurements have no dependency
        # on utility denominators. Preserve this failure and continue those.
        status['reference_freeze']='failed';status['errors'].append('reference_freeze: '+repr(exc))
    save(status_path,status)
    training=base/'training-v2';runs=[]
    for suffix in ('','-round2'):
        for task in ('ocr','image'):
            for n in (1,2):runs.append(training/f'g2-image-ocr{n}'/f'capacity-{task}{suffix}')
    for name in ('solo-round1','solo-round2','overlap-round1','overlap-round2'):
        runs.append(training/'g2-image-ocr1'/name)
    for directory in runs:
        status.update(status='collecting',current=str(directory));save(status_path,status)
        with (directory/'collector.log').open('x') as log:
            result=subprocess.run([sys.executable,str(ROOT/'experiments/run_activation_capture.py'),
                                   '--output',str(directory)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        if result.returncode!=0:
            status.update(status='capture_failed');status['errors'].append(str(directory));save(status_path,status);return 1
        raw=json.loads((directory/'raw.json').read_text())
        if raw['status']!='completed':
            status.update(status='capture_failed');status['errors'].append(str(directory));save(status_path,status);return 1
        status['completed'].append(str(directory));save(status_path,status)
    status.update(status='initial_capture_plan_completed',current=None,finished_unix=time.time());save(status_path,status)
    return 0


if __name__=='__main__':raise SystemExit(main())
