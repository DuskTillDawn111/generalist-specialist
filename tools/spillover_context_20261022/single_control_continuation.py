"""CPU preparation; finite serial continuation only after the six-run audit.

Never modifies the predecessor supervisor, packets or runtime. Preparation
reads plans/source only. Explicit execute waits for COMPLETE before reading
predecessor terminal queues and audit, then invokes the original serial runner.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

SPEC=importlib.util.spec_from_file_location('_frozen_continuation',Path(__file__).with_name('staged_continuation.py'))
base=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(base)
ROOT=base.ROOT;BASE=base.BASE;TOOLS=base.TOOLS


def predecessor_status(progress):
    status=progress.get('status')
    if status=='COMPLETE':return True
    if status in {'STARTING','WAITING_FINE_TERMINALS'} or isinstance(status,str) and status.startswith('RUNNING_'):return False
    raise RuntimeError('Predecessor stopped or unrecognized: '+str(status))


def predecessor_audit(progress,report,expected_ids,report_path):
    if not predecessor_status(progress):raise RuntimeError('Predecessor not COMPLETE')
    if progress.get('comparison')!=base.receipt(report_path):raise ValueError('Predecessor comparison receipt changed')
    if not progress.get('stages') or any(s.get('exit_code')!=0 for s in progress['stages']):
        raise RuntimeError('Predecessor stages incomplete/failed')
    base.audit_gate(report,expected_ids)
    # U, eligibility, ranking and measured benefit never condition execution.


def prepare(output,predecessor_path,single_packet):
    output=Path(output).resolve();predecessor_path=Path(predecessor_path).resolve();single_packet=Path(single_packet).resolve()
    if output.exists():raise FileExistsError('Never overwrite prepared continuation')
    prior=json.loads(predecessor_path.read_text());base.freeze_valid(prior)
    packet=json.loads(single_packet.read_text())
    if len(packet['jobs'])!=2 or [j['seed'] for j in packet['jobs']]!=[107,108]:raise ValueError('Expected single-control paired seeds')
    expected=dict(deployment={'text':1,'generalist_2':1,'ocr':1},routing={'text':{'text':.925,'generalist_2':.925},
        'image':{'generalist_2':.01},'asr':{'generalist_2':.01},'ocr':{'ocr':.01}})
    if any(j['plan']!=expected for j in packet['jobs']):raise ValueError('Single-control declared plan changed')
    checked=subprocess.run([sys.executable,str(TOOLS/'run_fine_development.py'),'--packet',str(single_packet)],
        cwd=ROOT,text=True,capture_output=True,check=True,timeout=60)
    validation=json.loads(checked.stdout)
    if validation.get('valid') is not True:raise ValueError('Packet CPU validation failed')
    files={Path(p['path']) for p in prior['frozen_files']}
    files.update([Path(__file__).resolve(),Path(base.__file__).resolve(),predecessor_path,single_packet,
        ROOT/'tests/test_single_control_continuation.py'])
    for bound in packet['bindings']:
        if base.receipt(bound['path'])!=bound:raise ValueError('Packet binding changed')
        files.add(Path(bound['path']))
    execute=output/'execution';single=execute/'single-controls-gpu';comparison=execute/'all-eight-comparison';plots=execute/'figures'
    command=lambda tool,*args:[prior['python']['path'],'-u',str(TOOLS/tool),*map(str,args)]
    fine_queue=prior['fine_queue'];strong_queue=str(Path(prior['outputs']['strong'])/'queue-progress.json')
    stages=[dict(name='single_gpu',command=command('run_fine_development.py','--packet',single_packet,'--output',single,'--execute')),
        dict(name='all_eight_compare',command=command('compare_development.py','--registration',prior['fine_packet']['path'],fine_queue,
            '--registration',prior['strong_packet']['path'],strong_queue,'--registration',single_packet,single/'queue-progress.json','--output',comparison)),
        dict(name='plot',command=command('plot_development.py','--comparison',comparison/'report.json','--output',plots))]
    request=dict(schema='gsdeploy.single_control_continuation.request.v1',status='PREPARED_NOT_EXECUTED',
        predecessor=base.receipt(predecessor_path),predecessor_progress=str(Path(prior['execution'])/'progress.json'),
        single_packet=base.receipt(single_packet),python=prior['python'],runner=base.receipt(TOOLS/'run_fine_development.py'),
        frozen_files=[base.receipt(p) for p in sorted(files)],root=str(ROOT),execution=str(execute),
        wait_seconds=5,maximum_seconds=7200,packet_validation=validation,
        outputs=dict(single=str(single),comparison=str(comparison),plots=str(plots)),stages=stages,
        policy=dict(no_retries=True,no_live_raw=True,no_utility_dependent_scheduling=True,
            preparation_reads_predecessor_results=False,predecessor='COMPLETE plus exact six terminal resource and shutdown audits',
            GPU='Original runner checks declared UUID and idle boundary before/after each serial job',
            failure='STOP_WITH_EVIDENCE',deadline='7200s from explicit execute start including wait and all stages'))
    base.freeze_valid(request);output.mkdir(parents=True,exist_ok=False)
    base.publish(output/'request.json',request)
    snapshots=output/'source-snapshot';snapshots.mkdir()
    for path in sorted(files):
        if path.suffix=='.py' and path.is_relative_to(ROOT):
            target=snapshots/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    (output/'README.zh.md').write_text('''# 单专用对照有限续跑

仅准备，未启动GPU。等待已冻结staged-continuation-v1 COMPLETE，核验六条已登记终态及资源/回收审计；随后使用原runner串行跑text1+G2+OCR的107/108两条，不改原packet、supervisor或v5源码。完成后对三个packet共八条做原compare与plot。收益和eligible不构成调度条件。任一失败STOP无重试，总期限7200秒。准备过程不读取前序进度、队列或结果。

请求SHA绑定源码、所有packet依赖和解释器；显式execute才进入等待/执行，目录存在即拒绝覆盖。所有终态依赖核验后才启动下一阶段。原runner的GPU idle边界不能防止外部启动器并发；延续root独占串行调度。正式运行前以CPU测试和root review为准。
''')
    return base.receipt(output/'request.json')


class Supervisor(base.Supervisor):
    def wait_predecessor(self):
        prior=json.loads(Path(self.request['predecessor']['path']).read_text());base.freeze_valid(prior)
        self.state['status']='WAITING_PREDECESSOR_COMPLETE'
        while True:
            self.checkpoint();path=Path(self.request['predecessor_progress'])
            if path.exists():
                content=path.read_bytes();progress=json.loads(content)
                if predecessor_status(progress):
                    comparison_path=Path(prior['outputs']['comparison'])/'report.json';report=json.loads(comparison_path.read_text())
                    packets=[json.loads(Path(prior[k]['path']).read_text()) for k in ['fine_packet','strong_packet']]
                    ids=[j['id'] for p in packets for j in p['jobs']]
                    if len(ids)!=6 or len(set(ids))!=6:raise ValueError('Expected exact distinct six predecessor declarations')
                    predecessor_audit(progress,report,ids,comparison_path)
                    for packet,packet_ref,queue_path,count in zip(packets,[prior['fine_packet'],prior['strong_packet']],
                            [Path(prior['fine_queue']),Path(prior['outputs']['strong'])/'queue-progress.json'],[4,2]):
                        if not base.queue_gate(json.loads(queue_path.read_text()),packet,packet_ref,count,self.request['runner']):
                            raise RuntimeError('Predecessor queue not terminal after COMPLETE')
                    base.freeze_valid(self.request);base.freeze_valid(prior)
                    (self.output/'predecessor-complete-snapshot.json').write_bytes(content)
                    self.state['predecessor_audit']=base.receipt(comparison_path);self.checkpoint();return packets
            time.sleep(self.request['wait_seconds'])

    def work(self):
        packets=self.wait_predecessor();self.run_stage(self.request['stages'][0])
        single=json.loads(Path(self.request['single_packet']['path']).read_text())
        queue_path=Path(self.request['outputs']['single'])/'queue-progress.json'
        if not base.queue_gate(json.loads(queue_path.read_text()),single,self.request['single_packet'],2,self.request['runner']):
            raise RuntimeError('Single-control queue not terminal after exit')
        self.run_stage(self.request['stages'][1])
        comparison=Path(self.request['outputs']['comparison'])/'report.json'
        base.audit_gate(json.loads(comparison.read_text()),[j['id'] for p in [*packets,single] for j in p['jobs']])
        self.run_stage(self.request['stages'][2]);self.state.update(status='COMPLETE',comparison=base.receipt(comparison))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--output',type=Path,default=BASE/'single-control-continuation-v1')
    prep.add_argument('--predecessor',type=Path,default=BASE/'staged-continuation-v1/request.json')
    prep.add_argument('--single-packet',type=Path,default=BASE/'text185-single-specialist-controls-v1/experiment-packet.json')
    run=sub.add_parser('execute');run.add_argument('--request',type=Path,required=True);run.add_argument('--request-sha256',required=True)
    args=parser.parse_args()
    if args.mode=='prepare':print(json.dumps(prepare(args.output,args.predecessor,args.single_packet)));return 0
    if base.receipt(args.request)['sha256']!=args.request_sha256:raise ValueError('Request SHA mismatch')
    return Supervisor(json.loads(args.request.read_text())).execute()


if __name__=='__main__':raise SystemExit(main())
