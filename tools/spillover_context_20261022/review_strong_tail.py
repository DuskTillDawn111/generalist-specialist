"""Readable tail interpretation, static figure and unqueued minimal next choice."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.diagnosis/'manifest.json').read_text())
    for name,expected in manifest['files'].items():assert sha(args.diagnosis/name)==expected
    r=json.loads((args.diagnosis/'report.json').read_text());pair=next(p for p in r['pairs'] if p['seed']==108)
    proposal=dict(schema='gsdeploy.strong_tail_minimum_followup_choice.v1',status='CPU_PROPOSAL_PENDING_DECISION_NOT_QUEUED',
        priority=1,purpose='Check reproducibility and temporal/order sensitivity of observed strong108 tail before fitting a resource/background causal explanation.',
        use_pending_evidence_first='Wait for already-running single-specialist107 terminal authorization and audit; do not read its live raw. It is a useful control but does not independently resolve the seed108 time-block anomaly.',
        minimum_new_conditions_if_gpu_followup_is_chosen=[
            dict(order=1,label='repeat_strong108_same_registered_trace',source_job_id='3424b9bdbca8ba76994e181f35e7ef5ecc647fe7bfb679a0755586075172f55b',
                deployment={'text':2,'image':1,'ocr':1,'asr':1},seed=108,horizon_seconds=600,warmup_seconds=30,drain_seconds=300),
            dict(order=2,label='repeat_fine108_nosplit_same_registered_trace',source_job_id='274df30b865ff132c2f1f61d65e9ec0cd4d51adb654b22acd299d3a03946831f',
                deployment={'text':2,'generalist_2':1,'ocr':1},seed=108,horizon_seconds=600,warmup_seconds=30,drain_seconds=300)],
        new_execution_blocks=2,new_deployment_or_route_identities=0,
        order_reason='Reverse original fine108 then strong108 order in a new matched block. This is minimal replication, not enough for a general causal or statistical superiority claim.',
        fixed_inputs=['Exact same registered108 arrivals/input sequence and dedicated text aliases',
            'Native execution/protocol/checkpoints and original warmup/drain',
            'Same measurement logging and failure/resource validity rules'],
        preregistered_diagnostics=['Native-service aliasFIFO alongside measured P95',
            'Predetermined30s/60s scheduled blocks and same alias/input conditional service',
            'Whether60–90s extra native work and long queue episode repeat or move',
            'Memory/energy validity and existing telemetry; retain failed/resource-invalid outcomes'],
        decision_afterward='If anomaly does not repeat or moves, keep temporal uncertainty and do not fit a deterministic resident/background multiplier. If deployment-specific slowdown repeats, design one fixed-residency activity contrast next.',
        old_G2_activity_pair='Remains PENDING_DECISION_NOT_QUEUED; not directly indicated by the current absence of direct background-handler coincidence.',
        no_GPU_started=True,no_model_fitting=True,no_predictor_change=True,
        limitation='Repeated108 is development replication with viewed inputs/outcomes, not a new independent validation seed or an optimality proof.')
    save(out/'minimum-calibration-choice.json',proposal)
    text='''# 强专108长尾：集中时间块减速被队列放大

同一108到达/input/专用alias下，strong text P95为12361.55ms，fine-nosplit为5461.72ms。全部1127条text的request/protocol/generation、生成token与raw_output逐条相同；不是输入或输出长度变化。

使用实际native service的独立alias FIFO得到12346.94/5454.48ms；加入实际handler和host提交时间后为12350.17/5456.48ms。因此差距主要已包含在服务时序和队列，host提交不是主因。worker时间戳为Unix ns，已用start_unix_ns归一化，host monotonic另算；逐条queue/handler时钟等式已核验。

strong text平均service794.04ms，相对fine752.23ms仅增加5.56%，总work多47.13s。时间并不均匀：60–90s的69条计划text请求多耗22.57s，210–240s另多20.18s；fine自身240–270s反而较慢，strong在该块少耗14.34s。strong text#0出现计划69.25–159.83s共92条连续queue>1s的长episode，平均等6.88s、最高12.27s。

事后仅把strong60–90s块换成fine同条请求native服务，其他真实服务固定不变，P95从12346.94降至5574.78ms；换60–120s降至5485.23ms。替换窗口是在看到结果后分析的，结果不可相加，也不是实际GPU反事实。把时间波动平滑为本run同alias/input服务均值，strong P95只剩4209.31ms（fine3548.93ms）；说明长尾不能用全程平均或一个静态驻留倍率解释。

最慢前7条text都不与任何背景handler区间重叠。strong108仅53条text与背景handler重叠，其相对fine同条服务比均值1.094；其余1074条为1.079，差异小且有长度选择偏差。背景handler区间包含CPU/launch/sync，并非GPU kernel活动；缺少直接重叠不能排除异步、allocator、设备状态或其他持续效应。目前没有证据把60–90s减速直接归因为某个背景任务。

107对照没有相同异常：strong/fine native FIFO P95为3535.33/3435.99ms，替换单个30s/60s块最多仅降约1.62ms；全部1156条text控制和输出也逐条相同。因此一次strong108显著差距不足以支持稳定的部署服务倍率或通用背景干扰定律。

最小下一步是先使用已在运行、待终态授权的single107控制。若之后仍需GPU校准，建议只追加两个重复执行块：同108原strong，然后同108原fine-nosplit，反转原顺序，保持原trace/input/alias/协议。无需新增配置或路由。它检验长尾是否重现/移时，不做因果或最优性证明。旧G2固定驻留活动pair维持待决定；当前没有理由立刻将它排队。完整机器提案见minimum-calibration-choice.json。

未重拟合或更改冻结预测；只读4个已审计终态raw，未读single107 live，未启动GPU。
'''
    (out/'README.zh.md').write_text(text)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
        colors=['#2563eb','#dc2626']
        for job,color in zip(pair['jobs'],colors):
            blocks=job['scheduled30s_blocks'];x=[b['begin_seconds']+15 for b in blocks];label=job['condition']
            axes[0].plot(x,[b['service']['mean'] for b in blocks],'-o',color=color,label=label,markersize=3)
            axes[1].plot(x,[b['observed_e2e']['p95']/1000 for b in blocks],'-o',color=color,label=label,markersize=3)
        strong=pair['jobs'][1];axes[2].bar([b['begin_seconds']+15 for b in strong['scheduled30s_blocks']],
            [b['excess_over_exact_fine_total_ms']/1000 for b in strong['scheduled30s_blocks']],width=24,color='#7c3aed')
        axes[0].set_ylabel('Mean native service (ms)');axes[1].set_ylabel('Block P95 e2e (s)');axes[2].set_ylabel('Strong minus fine\nservice work (s)');axes[2].set_xlabel('Scheduled measurement time (s)')
        axes[0].legend(frameon=False);axes[2].axhline(0,color='#374151',linewidth=.8)
        for ax in axes:ax.grid(alpha=.2);ax.axvspan(60,90,color='#f59e0b',alpha=.12)
        fig.suptitle('Seed108: same text arrivals, inputs and aliases; post-hoc diagnostics')
        fig.tight_layout();fig.savefig(out/'strong-fine108-time-blocks.png',dpi=170);plt.close(fig)
    except ImportError:pass
    save(out/'manifest.json',dict(script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        inputs={str((args.diagnosis/'manifest.json').resolve()):sha(args.diagnosis/'manifest.json')},
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()},queued=False,GPU_started=False))
    print(json.dumps(dict(output=str(out),new_conditions=0,proposed_repeat_blocks=2,queued=False)))

if __name__=='__main__':main()
