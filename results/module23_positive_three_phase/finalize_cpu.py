"""Independent CPU invariant checks and stopped-campaign report; no GPU use."""
from pathlib import Path
import csv
import hashlib
import json
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from module20.optimizer import predict_scenarios

def read(name):return json.loads((OUT/name).read_text())
def write(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    result=read('cpu_optimizer_recheck.json');protocol=read('cpu_protocol.json');d=read('six_model_catalog.json')
    gate=read('module23_gate.json');states=read('all_deployment_states.json')
    S='openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1';G='phi4mm'
    assert not result['Stage_A_passed'] and not gate['GPU_started'] and gate['formal_GPU_runs']==0
    assert len(states)==864 and sum(not s['reject_reasons'] for s in states)==43
    assert len(list(csv.DictReader((OUT/'gpu_paired_runs.csv').open())))==0
    assert len(list(csv.DictReader((OUT/'worker_stability.csv').open())))==0
    required=['cpu_optimizer_recheck.json','colocation_preflight.json','frozen_gpu_manifest.json',
              'gpu_paired_runs.csv','system_utility_comparison.csv','utility_components.csv',
              'per_task_metrics.csv','worker_stability.csv','module23_gate.json']
    assert all((OUT/p).is_file() for p in required)
    assert read('frozen_gpu_manifest.json')['frozen'] is False
    original_hashes=[]
    for r in result['frozen_inputs']:
        assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
        original_hashes.append(r['path'])
    checks=[];decomposition={}
    for stage,record in result['stages'].items():
        assert len([f for f in record['candidates'] if '_D' in f['candidate_id']])==43
        ranked=[f for f in record['candidates'] if (f['prediction'] or {}).get('eligible')]
        assert record['winner']['prediction']['U']==max(f['prediction']['U'] for f in ranked)
        for arm in ['winner','baseline']:
            f=record[arm];p=f['prediction'];x=f['deployment'];rr=f['routing'];rates=record['rates']
            assert all(m in x and t in d['models'][m]['supported_tasks'] and q>=0
                       for t,row in rr.items() for m,q in row.items())
            assert all(abs(sum(rr[t].values())-q)<1e-8 for t,q in rates.items())
            assert all(str(n) in d['memory_table_mib'][m] and str(n) in d['capacity_factors'][m] for m,n in x.items())
            recreated=predict_scenarios(d,x,rr,rates,protocol['budget_mib'],protocol['latency_references_ms'],protocol['weights'])
            assert recreated['eligible']==p['eligible'] and abs(recreated['U']-p['U'])<1e-12
            assert abs(p['U']-sum(p[k+'_contribution'] for k in ['Q','T','L','P','M']))<1e-12
            assert p['safe_memory_mib']<=protocol['budget_mib']+1e-8
            checks.append(dict(stage=stage,arm=arm,recomputed_U=recreated['U'],passed=True))
        if stage=='LOW':assert S in record['baseline']['deployment']
        if stage=='MID':assert S not in record['baseline']['deployment']
        if stage=='HIGH':
            assert record['winner']['deployment']==record['baseline']['deployment']
            assert record['baseline']['routing']['asr']=={S:record['rates']['asr']}
            assert all(record['winner']['routing'][t]==record['baseline']['routing'][t] for t in record['rates'] if t!='asr')
        decomposition[stage]={k:record['winner']['prediction'][k+'_contribution']-record['baseline']['prediction'][k+'_contribution'] for k in ['T','L','P','M']}
        assert abs(sum(decomposition[stage].values())-record['predicted_delta_U'])<1e-12
    integrity=read('historical_integrity.json');assert integrity['metadata_unchanged'] and integrity['scientific_input_hashes_unchanged']
    tests=(OUT/'original_optimizer_tests.log').read_text();assert 'Ran 7 tests' in tests and 'OK' in tests
    mem=read('colocation_preflight.json')['predicted_six_model_x1_safe_memory_mib']
    summary=['# Module 23：Positive-Utility Three-Phase Full-System Validation',
        '', '**最终 Gate：THREE_PHASE_CPU_NOT_REPRODUCED。Stage A 未通过，按用户停止规则未启动 Stage B 或 C。正式 GPU 运行 0/18。不能宣称正效用三阶段成立。**',
        '', '## 冻结输入与真实优化器',
        '', '模型池固定为 Phi-4-multimodal、Whisper-large-v3、RoBERTa-base-SST2、ConvNeXt-tiny、PARSeq、MMS-LID-256，三个阶段一致。Text/Image/OCR/LID/VQA 为 4/2/5/0.3/0.2 QPS，ASR 为 0.02/0.50/1.55 QPS，预算 B3=32019.2544 MiB。',
        '', '`U = -0.25*T_loss -0.30*L_norm -0.30*P_norm -0.15*M_norm`。原 Module20A 权重、六任务归一化参考、质量准入和 safe factor 均未改动。Phi 的 Text/Image/OCR 分别 0.905/0.92/0.93，未达到冻结门槛 0.925/1.0/0.96，因此不是合法路径；Phi 的 ASR/VQA 和其余五个 Specialist 对应路径全部准入。质量证据及排除项完整保存在 CPU JSON。',
        '', '重新调用项目现有 `module20.optimizer.Generator`（fraction=None、原 cap 0.8/0.95、固定主权重）和 `predict_scenarios`。另外枚举全部 864 个零或实测副本组合；43 个满足预算及六任务覆盖的部署全部参与同一真实效用评估器的路由优化。每个部署允许完整容量合法的连续 ASR 路由，包括纯 G、纯 S、混合；其余任务在此冻结准入池内各只有一个合法模型路径，不是人为固定背景路由。',
        '', '连续求解使用两组预先固定 DE 种子、端点和区间采样、局部标量精化，没有给胜出候选添加 ASR S 驻留、15% 分流或三阶段结构约束。所有合法部署已竞争；连续路由为数值搜索，未提供全局最优证明。原 MILP 仅是启发式候选生成器，完整枚举改善了 MID 的选择。',
        '', '**CPU 排名使用原 Module20 的周期 FIFO surrogate，未复用或冒充旧 Poisson 模拟输出。请求设计仍为独立 Poisson；周期 surrogate 对 Poisson 队列、样本变异和共置干扰的预测局限没有消除，CPU 数值不是真实 GPU 效用。**',
        '', '## Stage A 独立结果',
        '', '|阶段|自主 ASR G 占比|Whisper 驻留|预测 Selected U|预测 Baseline U|预测 ΔU|结构 / 预测增益 gate|',
        '|---|---:|---|---:|---:|---:|---|']
    for stage,r in result['stages'].items():
        f=r['winner'];b=r['baseline'];c=r['checks']
        summary.append(f"|{stage}|{f['ASR_G_fraction']:.8%}|{'是' if S in f['deployment'] else '否'}|{f['prediction']['U']:+.9f}|{b['prediction']['U']:+.9f}|{r['predicted_delta_U']:+.9f}|{'通过' if c['expected_structure'] else '失败'} / {'通过' if c['predicted_gain_gt_001'] else '失败'}|")
    summary += ['', '现有 Generator 的两次自由 MILP（cap=0.8/0.95）都给出 LOW/MID 不部署 Whisper、HIGH 全走 Whisper。因此，原启发式候选集本身也未复现三阶段；表中结果是扩大到全部合法部署并使用原真实效用评估器重新排名后的自主选择，未将新增连续搜索冒充原有候选生成算法。',
                '', 'HIGH 的 Whisper 画像服务时间约 494.9ms，而周期 1.55 QPS 的间隔约 645.2ms；该独立 ASR worker 的周期 surrogate 没有产生 Poisson 到达的排队尾部。Phi 画像 ASR 服务较慢，原评估器因此没有选择非零 G 分流。这个原因来自原预测模型，不能据此断言实际 Poisson GPU 分流无收益。MID 的微小非零 G share 也是有限周期画像的数值选择，不能当作实际非零执行证据。',
                '', 'LOW 比较器为同负载下增配 Whisper 的最优候选；MID 比较器为不部署 Whisper 的最优候选；HIGH 比较器保持胜出部署和所有非 ASR 路由完全相同，只将 ASR 全部送往 Whisper。比较器筛选发生在无约束胜出候选选定之后，不参与强制选择。',
                '', '|阶段|Selected safe MiB|最大预测 rho|该阶段 CPU 求解墙钟秒|', '|---|---:|---:|---:|']
    for stage,r in result['stages'].items():
        p=r['winner']['prediction'];summary.append(f"|{stage}|{p['safe_memory_mib']:.3f}|{p['max_rho']:.6f}|{r['wall_seconds']:.3f}|")
    summary += ['', f"Stage A 总墙钟时间 {result['wall_seconds']:.3f} 秒。所有胜出候选通过质量、预算、合法副本数、流量守恒和名义容量检查；稳定性仅是画像预测，未获 GPU 认证。",
        '', '## 效用分项：仅 CPU 预测差值', '', '|阶段|吞吐|平均 E2E|P95 E2E|显存|合计|', '|---|---:|---:|---:|---:|---:|']
    for stage,c in decomposition.items():
        summary.append('|'+stage+'|'+'|'.join(f'{c[k]:+.9f}' for k in ['T','L','P','M'])+f"|{sum(c.values()):+.9f}|")
    summary += ['', '正号表示 Selected 的收益，负号表示损失。以上来自固定 offered QPS 权重的全系统预测，不能当作 GPU 归因；执行服务时间与含共享 FIFO 排队的 E2E 始终区别。',
        '', '## Stage B/C 与统计证据',
        '', f'六模型各一副本的冻结画像 safe memory 合计 {mem:.4f} MiB，低于 B3 的余量为 {protocol["budget_mib"]-mem:.4f} MiB。这只说明画像预算可行，**未实测六模型共置**。未加载模型、未执行短时探针、未生成正式到达轨迹或冻结 GPU 运行 manifest。',
        '', f"冻结 ASR standalone 服务时间：Phi={d['models'][G]['service_time_ms']['asr']:.6f} ms，Whisper-large-v3={d['models'][S]['service_time_ms']['asr']:.6f} ms。此次 GPU 分布、共置偏差和预测/GPU 比率全部未知，不能填写 0 或拿历史不同配置替代。",
        '', '没有正式 GPU 观测，三阶段实际 ΔU、波动、置信区间和统计功效均为 NOT_TESTED。LOW 的 120s 窗口仅有 2.4 个预期 ASR 请求，不足以支撑尾部评价；10000s 仅将预期计数提高到 200、上 5% 约 10 个，并非功效保证。Stage A 失败后没有启动实验，未确定或激活测量时长和种子，不存在追显著、临时延长、重挑种子。',
        '', 'HIGH S-only 的实际稳定性未知，未将其预测 rho 当作实测稳定性，也未使用截尾延迟形成效用优势或容量优势结论。',
        '', '## 七个问题的最终回答',
        '', '1. **真实优化器是否自主形成三阶段？否。** 以自由 MILP 和完整部署/路由竞争的结果为依据，最终 Stage A gate 失败，详见上表。',
        '2. **六模型是否能在 B3 内真实共置？未验证。** 冻结画像合计低于预算，实际共置和 safe memory 没有新 GPU 证据。',
        '3. **三阶段分别有真实正效用？LOW/MID/HIGH 均未测。** 预测的正收益不能替代 GPU 配对收益。',
        '4. **HIGH G/S 是否优于同驻留 S-only？未获得真实证据；CPU 未满足 HIGH 正效用 gate。**',
        '5. **哪些分量产生收益/损失？** CPU 预测分解如上，真实 GPU 分解未知。',
        '6. **预测与 GPU 服务时间偏差多大？未知。** 没有本次共置探针或正式 GPU 分布。',
        '7. **是否能宣称正效用三阶段成立？不能。** Module23 按 CPU 门槛失败停止。',
        '', '## 输出完整性与复现',
        '', '要求的 10 个输出全部保留。`colocation_preflight.json` 标记 NOT_RUN_STAGE_A_FAILED；`frozen_gpu_manifest.json` 明确 frozen=false、slots=[]。`gpu_paired_runs.csv`、`worker_stability.csv` 只有表头；其余 CSV 的 CPU 行明确标注 CPU_PREDICTION_ONLY，实际 GPU 字段留空。没有把未运行标作成功或失败 GPU 观测。',
        '', f"原优化器 7 个测试通过，6 个 Selected/Baseline 预测独立重算一致。{integrity['files_checked']} 个既有文件的大小/mtime 不变，所有实际使用的科学输入 SHA256 不变；未声称重新哈希全部历史字节。首次 CPU 程序因不可行候选的 None 处理报错，修复后使用完全相同科学输入、路由域和种子重算；首次日志保留在 `cpu_recheck_attempt1.log`。无 GPU 重试。",
        '', '复现命令（仅写本输出目录的 CPU 核验与报告）：`CUDA_VISIBLE_DEVICES=\'\' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 envs/data_env/bin/python results/module23_positive_three_phase/finalize_cpu.py`。完整 CPU 搜索脚本拒绝覆盖已完成实验，原始搜索与失败日志均保留。',
        '', 'Module 9–22 历史实验未修改；未进入 Module24。完成后停止。']
    (OUT/'module23_summary.md').write_text('\n'.join(summary)+'\n')
    write('final_validation.json',dict(passed=True,required_artifacts_present=True,independent_recomputed_checks=checks,
        original_optimizer_tests=7,scientific_hashes_verified=original_hashes,historical_integrity=integrity,
        formal_GPU_runs=0,CPU_prediction_never_claimed_as_GPU_evidence=True,summary_present=True))
    print('FINAL_VALIDATION_PASSED')

if __name__=='__main__':main()
