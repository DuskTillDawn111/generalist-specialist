"""Seal a readable interpretation and an unqueued matched activity design."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'artifacts/static-spillover-study-20261022'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    report=json.loads((args.diagnosis/'report.json').read_text());manifest=json.loads((args.diagnosis/'manifest.json').read_text())
    for name,expected in manifest['files'].items():assert sha(args.diagnosis/name)==expected
    schedule_path=STUDY/'local-context-seed108-forecast-v1/text185_g2share_1of32_seed108-requests.json'
    original=json.loads(schedule_path.read_text())
    fields=['index','arrival_seconds','task','pool_id','alias','family','n']
    full=[{k:r[k] for k in fields} for r in original]
    core=[r for r in full if not (r['family']=='generalist_2' and r['task']=='text')]
    overlay=[r for r in full if r['family']=='generalist_2' and r['task']=='text']
    assert len(full)==1144 and len(core)==1109 and len(overlay)==35
    assert len({r['pool_id'] for r in overlay})==2
    save(out/'fixed-specialist-and-background-input-schedule.json',core)
    save(out/'activity-on-G2-text-overlay-input-schedule.json',overlay)
    design=dict(schema='gsdeploy.fixed_residency_activity_pair.proposal.v1',status='CPU_ONLY_UNQUEUED_DIAGNOSTIC_PROPOSAL',
        deployment={'text':2,'generalist_2':1,'ocr':1},same_resident_workers_and_checkpoint_contract=True,
        measurement_seconds=600,drain_seconds=300,warmup_seconds=30,
        identical_warmup_policy='Use same frozen largest-three-per-task sample list in both conditions, including G2 text warmup; do not vary checkpoint loading or warmup by routed activity.',
        conditions=[dict(id='A_G2_text_idle_resident',schedule='fixed-specialist-and-background-input-schedule.json',
            text_specialist_rps=1.7921875,G2_text_probe_rps=0.,other_nominal_rates={'image':.01,'asr':.01,'ocr':.01}),
            dict(id='B_G2_text_active_same_residency',schedule='A exact fixed schedule plus activity-on-G2-text-overlay-input-schedule.json',
                text_specialist_rps=1.7921875,G2_text_probe_rps=.0578125,other_nominal_rates={'image':.01,'asr':.01,'ocr':.01})],
        exact_held_constant=['Every dedicated-text arrival, input, alias, order and nominal route rate',
            'All image/ASR/OCR arrivals, inputs and aliases','text replica count2, G2 replica count1, OCR replica count1',
            'Hardware, native protocol, execution/checkpoints and warmup/drain'],
        single_changed_factor='35 predetermined G2-text probe requests added to an otherwise identical input/alias schedule; G2 remains resident in both.',
        purpose='Estimate activity-induced change in dedicated-text conditional service under fixed resident state and fixed specialist input/alias queues; distinguish from route-induced alias remapping.',
        execution_blocks=dict(minimum_two_conditions=2,recommended_order=['A','B','B','A'],
            independent_execution_blocks=4,why='Counterbalance temporal order; one A/B pair cannot separate drift.',queued=False),
        diagnostics=['Dedicated text service per same alias/input with run-level contrasts',
            'Per-alias arithmetic work and per-time-block service/queue changes',
            'G2 probe service and measured activity, with failed/resource-invalid runs retained',
            'Memory and energy evidence for validity, without total-utility ranking between differing workload schedules'],
        limited_G2_probe_support=dict(inputs=2,requests=35,rule='Deliberately reproduces current split exposure; cannot estimate full32-input G2 task population or seven-pool predictor.'),
        objective_scope='Explicit calibration subprotocol; A and B have different total offered load and cannot be compared as alternative deployments at the original target workload.',
        next_action='Review already running strong-specialist controls before queueing; no GPU work requested or initiated by this proposal.')
    save(out/'matched-activity-pair-proposal.json',design)
    proposed_cpu_features=dict(schema='gsdeploy.alias_planned_context.next_cpu_diagnostic.v1',status='PROPOSED_NOT_FITTED_NOT_ACTIVATED',
        features=['Exact planned per-alias baseline workload sum/horizon','Same-family partner baseline workload sum/horizon',
            'Per-alias planned input-frequency entropy','Per-alias baseline difficulty distribution (mean/quantiles/CV)',
            'Per-alias planned task entropy','Prespecified time-window baseline work'],
        source='Only frozen input/trace/alias route and original per-input n-state baselines; no actual service, generated tokens, realized overlap or measured activity as features.',
        available_two_seed_diagnostic='planned_trace_baseline_context already recorded for every text alias in diagnosis report; outcome labels remain development only.',
        caveat='Existing frozen candidate already uses per-input b and exact alias FIFO, so alias input effects were partially present. Missing item is an explicit alias/partner-dependent service context, not all awareness of input mix.',
        next_stage='CPU-only grouped diagnostic on existing development runs, then predeclare a new candidate if justified; preserve failed forecast and avoid immediate dynamic overlap or GPU expansion.')
    save(out/'next-cpu-feature-diagnostic-proposal.json',proposed_cpu_features)
    text='''# 107/108 反转：失败保留与最小活动配对

冻结预测在108的L/P方向均失败：预测split−nosplit为+0.02780/+0.11365，实测为−0.16441/−1.30789；nosplit P低估61.8%，split P低估46.8%。未重拟合、未改旧预测、未激活。

实际native服务按原alias FIFO回放，107 ΔP=+0.18730（实测+0.18892），108 ΔP=−1.30352（实测−1.30789）。host提交/handler替换只改变很小残差；108专用text实际服务替换冻结预测后即给出ΔP=−1.47439，而仅换G2-text或背景仍为正。主要不确定性在条件服务与专用队列，不能归因成通用更快。

两seed中分流都会重映射专用输入：nosplit两个alias各固定16/32输入；split各31/32，G2只接2/32输入。107专用总work879.7→866.4s，最忙alias472.3→443.2s；108总work847.8→845.5s几乎不变，最忙alias476.3→429.7s。108 no-split两个alias work差104.8s，split仅13.9s；队列尾明显减小。因此总体算术service几乎不变也能通过别名均衡改变P95。

G2实际单请求更慢：107分流36条平均3.276s，相同输入原专用1.277s；108分流35条平均3.920s，相同输入原专用0.851s。分流增加通用慢请求，专用别名均衡/减负能否覆盖该代价随seed变化。107共同alias/input细胞中split/no-split服务比中位1.043，108为0.991；实际专用服务的条件差异与时间变化足以让107符号翻回。固定共同输入均值的事后替换在两seed均给出收益，但它未观测跨方案真实反事实服务，不能作因果证明。

计划已知基线也能看到别名输入不平衡：107 no-split基线work435.6/522.9s，split470.7/439.3s；108 no-split539.9/394.5s，split441.6/458.2s。现模型的family-self/other上下文不含alias/partner输入分布，但逐输入b与精确aliasFIFO已经部分表达了它，不能说模型完全没看alias。下一步可先CPU诊断每alias计划基线work、输入熵/难度分布、同族partner负载，保持实际service仅为开发结果标签。

最小固定驻留活动pair已写成机器提案和纯输入schedule：固定text2+G2n1+OCRn1以及同一组专用/背景到达、input、alias；A仅令G2-text probe idle，B叠加当前35条G2-text probe。它隔离活动与原路由改变别名输入的影响。建议A/B/B/A四执行块平衡时序，仅为局部校准，不把不同总负载的两条件当目标部署效用比较。当前不排GPU，先审阅已在运行的强专控制。

两seed是开发诊断，不能据108收益宣称算法已会主动找最优路由，不能宣称混合普遍胜出或最优性已证。
'''
    (out/'README.zh.md').write_text(text)
    save(out/'manifest.json',dict(script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        inputs={str((args.diagnosis/'manifest.json').resolve()):sha(args.diagnosis/'manifest.json'),str(schedule_path):sha(schedule_path)},
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()},gpu_started=False,queued=False))
    print(json.dumps(dict(output=str(out),conditions=2,queued=False,G2_probe_inputs=2)))

if __name__=='__main__':main()
