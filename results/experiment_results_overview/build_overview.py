"""Summarize saved experiments without changing their frozen artifacts."""
from pathlib import Path
import csv
import hashlib
import json
import math
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCES = set()


def read(rel):
    path = ROOT / rel
    SOURCES.add(path)
    return json.loads(path.read_text())


def read_csv(rel):
    path = ROOT / rel
    SOURCES.add(path)
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def link(rel, label):
    path = ROOT / rel
    assert path.is_file(), path
    SOURCES.add(path)
    return f'[{label}]({path})'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
        ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def export(name, rows):
    assert rows
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(row, metric):
    value = row['replicate_statistics'][metric]
    assert row['n'] == len(value['values']) == 2
    assert math.isclose(sum(value['values']) / 2, value['mean'], abs_tol=1e-9)
    return f"{value['mean']:.3f} ± {value['sample_std']:.3f}"


def main():
    models = {key: {r['workload']: r for r in read(
        f'results/module8e3_heldout_validation/{key}_results.json')['workloads']}
        for key in ('m1', 'm2', 'b2')}
    summary = read('results/module8e3_heldout_validation/v2_vs_v1_summary.json')
    gap = read('results/module8e3_heldout_validation/b2_gap_analysis.json')
    ablation = read('results/module8e4_revision_ablation/ablation_claim_audit.json')
    effects = read('results/module8e4_revision_ablation/factorial_effects.json')
    current = []
    for record in summary['per_workload']:
        workload = record['workload']
        a, b, c = (models[m][workload] for m in ('m1', 'm2', 'b2'))
        delta = b['weighted_mean_E2E_ms'] - a['weighted_mean_E2E_ms']
        assert math.isclose(delta, record['M2_vs_M1']['mean_difference_ms'], abs_tol=1e-9)
        row = dict(workload=workload, intensity=record['intensity'],
            confirmation_n=2, M2_vs_M1=record['M2_vs_M1']['outcome'],
            M1_vs_B2=record['M1_vs_B2']['outcome'], M2_vs_B2=record['M2_vs_B2']['outcome'],
            mean_gain_percent=(a['weighted_mean_E2E_ms']-b['weighted_mean_E2E_ms']) / a['weighted_mean_E2E_ms'] * 100,
            p95_gain_percent=(a['weighted_task_P95_ms']-b['weighted_task_P95_ms'])/a['weighted_task_P95_ms']*100)
        for key, result in zip(('v1', 'v2', 'B2'), (a, b, c)):
            assert result['stability'] == 'steady' and result['completion_ratio'] == 1
            stats(result, 'weighted_mean_e2e')
            stats(result, 'weighted_p95_e2e')
            row.update({f'{key}_mean_ms': result['weighted_mean_E2E_ms'],
                f'{key}_mean_sample_std_ms': result['replicate_statistics']['weighted_mean_e2e']['sample_std'],
                f'{key}_weighted_task_p95_ms': result['weighted_task_P95_ms'],
                f'{key}_p95_sample_std_ms': result['replicate_statistics']['weighted_p95_e2e']['sample_std'],
                f'{key}_planning_memory_mib': result['planning_memory_mib'],
                f'{key}_peak_vram_mean_mib': result['peak_VRAM_mib']['mean'],
                f'{key}_peak_vram_max_mib': max(result['peak_VRAM_mib']['values']),
                f'{key}_deployment': json.dumps(result['deployment_vector'], sort_keys=True),
                f'{key}_N_G': result['N_G'], f'{key}_N_S': result['N_S']})
        current.append(row)
    actual_counts = Counter(r['M2_vs_M1'] for r in current)
    assert all(actual_counts[k] == v for k, v in summary['counts'].items())
    assert len(current) == 6
    for name, source_key in (('v1', 'G1'), ('v2', 'G2')):
        calculated = sum(max((r[f'{name}_mean_ms']-r['B2_mean_ms']) / r['B2_mean_ms'],0) for r in current)/6
        assert math.isclose(calculated,gap[source_key],abs_tol=1e-12)
    e3_cost = read('results/module8e3_heldout_validation/run_accounting.json')
    e4_cost = read('results/module8e4_revision_ablation/run_accounting.json')
    assert e3_cost['new_timed_runs'] == 181 and e3_cost['confirmation_runs'] == 40
    assert e4_cost['new_timed_runs'] == 6 and e4_cost['confirmation_runs'] == 0
    export('heldout_v1_v2_B2.csv', current)
    ablation_rows = read_csv('results/module8e4_revision_ablation/ablation_comparison.csv')
    assert len(ablation_rows) == 30
    export('ablation_comparisons.csv', ablation_rows)
    for comparison, counts in ablation['comparison_counts'].items():
        actual = Counter(r['outcome'] for r in ablation_rows if r['comparison'] == comparison)
        assert all(actual[k] == v for k, v in counts.items())
    old = read('results/module8c_final_real_search/module8c_summary.json')
    dev = old['workloads']
    export('v1_development_search_and_confirmation.csv', [dict(
        workload=r['workload'], search_seed_mean_ms=r['seed_mean_e2e_ms'],
        search_final_mean_ms=r['final_mean_e2e_ms'], search_mean_gain_percent=r['mean_improvement_percent'],
        confirmation_seed_mean_ms=r['confirmation_seed_mean_ms'],
        confirmation_final_mean_ms=r['confirmation_final_mean_ms'],
        confirmation_mean_gain_percent=r['confirmation_mean_improvement_percent'],
        confirmation_objective_improved=r['confirmation_objective_direction_maintained'],
        search_states=r['runtime_states'], planning_memory_mib=r['final_memory_mib'],
        final_deployment=json.dumps(r['final'], sort_keys=True)) for r in dev])
    baselines = read_csv('results/module8d_real_baselines/static_baseline_comparison.csv')
    export('v1_development_fixed_baselines.csv', baselines)
    baseline_means = {}
    for key in ('b0', 'b1', 'b2', 'b3'):
        data = read(f'results/module8d_real_baselines/{key}_results.json')
        items = data['results']
        assert isinstance(items, list)
        baseline_means[key] = {r['workload']: r for r in items}

    inventory = [
        ('Stage 1.5 / MILP', '正确性与可扩展性', 'CPU/oracle', '19 场景一致；20/30-model 合成求解未调用枚举', 'results/deployment_seed/summary.md'),
        ('Stage 2A', '测量链路', 'GPU smoke', '2 个 smoke、150 条请求记录；只证明链路', 'results/stage2a/summary.md'),
        ('Stage 2B', '路由端点比较', '历史 GPU', '固定部署下 specialist-first 大幅降低 E2E；非当前质量约束主结果', 'results/stage2b/routing_characterization/routing_characterization_findings.md'),
        ('2 Foundation', '预测数据基础', '历史数据汇总', '初始 18 runs / 148 task-model rows', 'results/performance_model_foundation/dataset_summary.md'),
        ('2B', '特征定向 profiling', '历史 GPU', '33 runs / 11 configurations；保留不稳定与 incomplete', 'results/performance_model_profiling/qwen/profiling_summary.md'),
        ('2C', '主动采样', '历史 GPU+覆盖诊断', '新增18 runs；总69 runs / 463 rows；覆盖距离下降42.8%', 'results/performance_model_active_sampling/summary.md'),
        ('2D + DQA', '跨部署预测泛化', '纠错后离线评估', '整体 PARTIAL；strict C-GBT MAE56.964ms，extended465.814ms', 'results/data_quality_audit/audit_summary.md'),
        ('2E', '少样本 bias 校准', '纠错后离线评估', 'FAIL；K=1 strict MAE57.763→71.197ms', 'results/performance_model_local_calibration_corrected/local_calibration_summary.md'),
        ('2F', '候选排序验证', '纠错后离线评估', '跨部署可比组不足；不能批准线上 scorer', 'results/performance_model_ranking_corrected/ranking_summary.md'),
        ('2G', '可比组 profiling', '纠错后历史 GPU', '45 次计划执行；合法 strict 1组、extended 2组，仍不足', 'results/performance_model_comparable_groups_corrected/module2g_summary.md'),
        ('3B', '闭环搜索 pilot', '历史 GPU', 'recoverable→steady；12997.344→84.679ms，不能直接当稳定区收益', 'results/deployment_search_real_pilot/module3b_summary.md'),
        ('3C', '等预算多负载搜索', '历史 GPU', '3 组从 Qwen seed 改善76.40%–80.14%；旧模型池', 'results/deployment_search_multiworkload/module3c_summary.md'),
        ('4B', '固定部署路由搜索', '历史 GPU pilot', 'NEUTRAL；59.216ms保持不变，2个候选回退', 'results/routing_search_real_pilot/module4b_summary.md'),
        ('5B', 'warm 动态转移', '历史 GPU pilot', '18→11次状态评估；节省38.89%；旧算法、phase式执行', 'results/dynamic_adaptation_real_pilot/module5b_summary.md'),
        ('7A', '模型池审计', '资源/能力审计', '扩展通用与专用模型候选；pending能力独立标记', 'results/candidate_pool_v2/candidate_pool_audit.md'),
        ('7A-2', '服务时间补全', '隔离 GPU profiling', '新增14条路径、42轮；完成25/25', 'results/candidate_pool_v2_service_demand/summary.md'),
        ('7C-1', '多副本容量', 'GPU scaling', '54 runs；g2 Qwen1.216 / Whisper1.238 / ResNet1.978', 'results/replica_scaling_real/module7c1_summary.md'),
        ('7C-2', '经验容量校准', 'CPU planning', '148 tests；Qwen副本up排名1→8；无linear GPU对照', 'results/replica_capacity_calibration/calibration_summary.md'),
        ('7C-3', '整数副本搜索', 'GPU pilot', '最终Qwen+Whisper；均值260.237→105.740ms', 'results/integer_replica_search_real_pilot/module7c3_summary.md'),
        ('7C-4', '整数搜索多负载', 'GPU+原W2复用', '3组收敛同一Qwen+Whisper；确认收益53.59%–63.48%', 'results/integer_replica_search_multiworkload/module7c4_summary.md'),
        ('8A', '质量与负载审计', '质量 GPU+离线审计', '发现旧负载接近；Phi/MiniCPM VQA验证；正式阈值待8B冻结', 'results/quality_workload_audit/module8a_summary.md'),
        ('8B', '质量准入与pool v2.1', '6轮新profiling+规划', '27/27服务路径；21准入边；VQA三个Generalist可用', 'results/module8b_quality_pool_finalization/module8b_summary.md'),
        ('8C', 'v1六负载搜索', '开发集 GPU', '3组确认收益，3组无收益；4种最终部署', 'results/module8c_final_real_search/module8c_summary.md'),
        ('8D-1', '基线冻结', 'CPU协议冻结', 'Seed/Qwen/greedy/static/random比较合同', 'results/module8d_baseline_freeze/module8d1_summary.md'),
        ('8D-2', 'v1基线比较', '开发集 GPU+精确复用', '对greedy 0胜1平5负；对Qwen 6胜', 'results/module8d_real_baselines/module8d2_summary.md'),
        ('8D-3', 'v1失败诊断', '9轮新固定状态GPU+离线轨迹', '多因素：邻域、排序/路由和两步接受障碍', 'results/module8d3_failure_diagnosis/module8d3_summary.md'),
        ('8E-1', 'v2修改与回放', '离线开发验证', '新增SWAP与分名额；6组回放都有cache miss，不作final结果', 'results/module8e1_method_revision/module8e1_summary.md'),
        ('8E-2', '留出协议冻结', 'CPU预注册', '3个新mixture×2强度；相同搜索预算', 'results/module8e2_heldout_freeze/module8e2_summary.md'),
        ('8E-3', 'v2留出验证', '正式留出 GPU', 'v2/v1=2胜4平0负；B2正均值gap减少58.07%', 'results/module8e3_heldout_validation/module8e3_summary.md'),
        ('8E-4', '两因素消融', '验证后消融+精确复用', '单项均0胜6平；联合2胜4平；支持本预算协同', 'results/module8e4_revision_ablation/module8e4_summary.md'),
    ]
    export('experiment_inventory.csv', [dict(stage=s, question=q, evidence_type=t,
        result=v, source=str(ROOT/p)) for s,q,t,v,p in inventory])

    text = [
        '**现有实验结果整理 — 截至 2026-10-05（北京时间）**',
        '覆盖 results 中的研究主线：测量、性能建模、部署搜索、路由/动态 pilot、模型池、质量、多副本、v1 基线、v2 留出验证及消融。旧 benchmark/formal 和 placement 原始实验保留为数据来源，不跨协议合并。本文整理已有结果，没有新增 GPU 实验、改变算法或修改冻结产物。',
        '**1．当前最重要的结论**',
        '- v2 在预冻结留出协议上相对 v1：2 WIN / 4 TIE / 0 LOSS。两个 WIN 都来自 mix_02 的 Phi→Qwen 替换。\n- v2 相对 Specialist-greedy + Minimal Generalist（B2）：0 WIN / 3 TIE / 3 LOSS；v1 为0/1/5。v2 缩小差距，尚未整体超越 B2。\n- 两项修改单独使用均为0/6/0；联合使用才产生2个 WIN。证据支持本集合、本预算下的机制协同。\n- 收益伴随资源变化：两个 WIN 的规划显存都增加6206 MiB。双副本候选没有进入这次主验证/消融的真机评估。\n- 当前静态算法证据已通过审计；路由修改、新v2动态、多硬件和资源预算扫描仍缺正式结果。',
        '**2．测量口径与公平性**',
        '当前实验使用同一 NVIDIA RTX 4090，冻结容量49140 MiB；主搜索10s warmup、210s measurement、三次重复；每方法每负载最多6次逻辑状态评估（含seed）、Top-B=2、最多3轮、patience=1、predictor=None。最终主比较使用两次独立确认测量，确认结果不改变搜索决策。',
        '均值是任务到达率加权E2E；P95是任务P95的到达率加权平均。sample std 描述重复离散，不是置信区间。WIN/TIE/LOSS按冻结NoiseAwareRuntimeObjective判定：稳定性优先，steady均值1ms容差，延迟获胜需平均延迟相对改善至少5%，再按既有P95/显存次序比较。TIE不等于延迟数值完全相同。',
        '缓存精确复用仍占逻辑搜索名额，复用raw不增加独立样本。留出集是3个新mixture各2个强度，沿用原任务数据与单硬件；不能当作6个独立任务分布。消融是验证后集合，未作为第二次独立留出验证。',
        '**3．v2正式留出主结果：独立确认 n=2**',
        table(['负载','v1均值 ± std (ms)','v2均值 ± std (ms)','B2均值 ± std (ms)','v2均值下降','v2/v1','v2/B2'],
            [[r['workload'].replace('heldout_',''), stats(models['m1'][r['workload']], 'weighted_mean_e2e'),
              stats(models['m2'][r['workload']], 'weighted_mean_e2e'), stats(models['b2'][r['workload']], 'weighted_mean_e2e'),
              f"{r['mean_gain_percent']:.2f}%",r['M2_vs_M1'],r['M2_vs_B2']] for r in current]),
        table(['负载','v1加权任务P95 (ms)','v2加权任务P95 (ms)','B2加权任务P95 (ms)','v2 P95下降'],
            [[r['workload'].replace('heldout_',''), f"{r['v1_weighted_task_p95_ms']:.3f}",f"{r['v2_weighted_task_p95_ms']:.3f}",
              f"{r['B2_weighted_task_p95_ms']:.3f}",f"{r['p95_gain_percent']:.2f}%"] for r in current]),
        'mix_02_moderate的v1两次确认均值分别为65.249和37.029ms，sample std=19.955ms；33.56%的聚合改善必须连同该波动报告。前次算法解释中的36.979→33.826ms是搜索阶段n=3，而此处51.139→33.975ms是独立确认n=2，两者不能混用。所有本组正式最终状态steady、completion=1。',
        f"B2正均值差距定义为各负载max((L方法−LB2)/LB2,0)后六负载等权平均：v1={gap['G1']*100:.5f}%，v2={gap['G2']*100:.5f}%，相对减少{(gap['G1']-gap['G2'])/gap['G1']*100:.5f}%。该值是相对B2差距的减少，不是v2平均延迟下降58%。",
        link('results/module8e3_heldout_validation/module8e3_summary.md','原留出报告')+'；'+link('results/module8e3_heldout_validation/heldout_method_comparison.csv','原逐负载比较'),
        '**4．v2消融：谁产生了收益**',
        'R1=定向SWAP/替换闭合；R2=按候选类别分配Top-B。M00=v1，M10=仅R1，M01=仅R2，M11=完整v2。',
        table(['比较','WIN','TIE','LOSS'], [[k,v['WIN'],v['TIE'],v['LOSS']] for k,v in ablation['comparison_counts'].items()]),
        'mix_02 moderate/high中，M10已生成Phi→Qwen，但全局Δz排名第6，未获实测名额；M01没有生成该SWAP；M11将其放在第2并实测接受。机制链条覆盖“生成→排名遗漏→分类入选→实测→接受”。这支持两项修改对已观察收益的共同作用，不证明普遍必要性。',
        '8E-3中替换闭合候选12次进入Top-B，10次实际评估、2次接受、8次回退，另2次因剩余预算未评估。8E-4复用原v1/v2端点数据，只新增6次测量、0次新确认；复用记录不是新的独立样本。',
        link('results/module8e4_revision_ablation/module8e4_summary.md','原消融报告')+'；'+link('results/module8e4_revision_ablation/mix02_case_study.md','机制案例'),
        '**5．v1开发集结果及强基线**',
        '以下六组用于v1评估与问题诊断，后来也参与v2开发回放，不能重新包装成v2留出结果。搜索与独立确认分列：',
        table(['负载','搜索seed→final (ms)','搜索均值下降','确认seed→final (ms)','确认均值下降','确认收益'],
            [[r['workload'],f"{r['seed_mean_e2e_ms']:.3f}→{r['final_mean_e2e_ms']:.3f}",f"{r['mean_improvement_percent']:.2f}%",
              f"{r['confirmation_seed_mean_ms']:.3f}→{r['confirmation_final_mean_ms']:.3f}",f"{r['confirmation_mean_improvement_percent']:.2f}%",
              'YES' if r['confirmation_objective_direction_maintained'] else 'NO'] for r in dev]),
        'v1相对冻结基线：',
        table(['基线','定义','v1 WIN/TIE/LOSS','可支持的结论'],[
            ['B0','MILP seed-only','3/3/0','三个负载搜索有效'],
            ['B1','Qwen-centric','6/0/0','在这六组优于Qwen中心方案'],
            ['B2','Specialist-greedy + Minimal Generalist','0/1/5','不能声称优于强贪心'],
            ['B3','固定Static Hybrid','3/3/0','有限支持按负载部署'],
            ['B4','Random-neighbor，3个随机seed','2/15/1（18个比较）','引导搜索优势有限；重叠状态不是独立重复']]),
        'B2按每任务选择质量准入路径中最快的specialist，没有specialist时使用最快generalist；沿用确定性greedy路由。以下是历史开发集search-protocol n=3均值，仅用于理解原基线结果：',
        table(['负载','v1 (ms)','B0','B1','B2','B3'],[[r['workload'],f"{r['final_mean_e2e_ms']:.3f}"]+
            [f"{baseline_means[m][r['workload']]['metrics']['mean_e2e_ms']:.3f}" for m in ('b0','b1','b2','b3')] for r in dev]),
        'reasoning/VQA-heavy的Random-2207达到Qwen+S4，搜索均值86.406ms，v1为95.491ms；确认85.843/94.771ms也保留这一方向。不能据此声称LP排序普遍优于随机。',
        link('results/module8c_final_real_search/module8c_summary.md','原v1搜索报告')+'；'+link('results/module8d_real_baselines/module8d2_summary.md','原基线报告'),
        '**6．为什么要改v1：诊断证据**',
        '8D-3确认B2不是因为显存或容量不可行而被排除。部分从seed到B2的旧生成图最短路径为5步，超过最多3轮；新增中间态必须先被接受，形成两步替换障碍。reasoning移除Phi是已测有益候选，但旧排序排第4。',
        '固定reasoning部署，只改路由，均值95.491→88.855ms，约下降6.95%；Phi仍驻留，VQA全交Qwen。该固定状态对照证明路由差异确有影响；v2仍未改路由。69个转移事件/30个去重转移的LP–mean Spearman为0.584/0.470，不能把共享重复事件当独立样本。',
        link('results/module8d3_failure_diagnosis/module8d3_summary.md','原诊断报告'),
        '**7．支撑实验与历史分支**',
        '质量与候选池：27条verified+measured服务路径、21条准入边；六条排除边为Qwen/OCR、Phi/text/image/OCR、MiniCPM/text、PaddleOCR-VL/OCR。VQA/LID使用SEMANTIC_VALIDITY_ONLY，数值SLO仍UNSPECIFIED。Phi/MiniCPM的VQA准确率点估计0.965/0.960，Qwen0.900；能力与质量是逐路径判断。',
        table(['任务','质量政策'],[['text','accuracy ≥0.925'],['image','accuracy ≥1.0'],['ocr','exact match ≥0.96'],['asr','WER ≤0.042181'],['VQA/LID','语义有效准入；准确率报告，不设数值SLO']]),
        '多副本：7C-1完成54个真实执行；高负载paired throughput增益为Qwen1.216±0.094、Whisper1.238±0.034、ResNet1.978±0.046。Qwen/Whisper不能假设线性2倍；高压运行包含真实congestion/unstable，容量改善不等于稳态保证。',
        table(['模型','经验g(2)','单副本显存MiB','双副本显存MiB'],[['Qwen','1.215925','17978','35670'],['Whisper-small','1.237807','1616','3240'],['ResNet50','1.978475','604','1208']]),
        '旧整数搜索7C-4：三组均收敛Qwen+Whisper，独立确认均值改善60.46%、63.48%、53.59%；当时Qwen由唯一verified VQA路径强制驻留。它支持offload收益，但不支持三组不同G/S分配，也不证明复制模型更优。当前8E-3的x=2候选generated/Top-B/evaluated/accepted=32/0/0/0；8E-4四方法累计56/0/0/0，两个统计范围重叠，不能相加为独立样本。',
        '性能预测分支采用纠错后结论：2D strict的C-GBT MAE56.964ms、extended465.814ms，整体gate PARTIAL；2E bias-only calibration FAIL；2F/2G可比候选组不足。2D strict文件自身PASS不能替代整体PARTIAL。这解释了当前主算法predictor=None。',
        '早期4B路由搜索pilot为NEUTRAL，均值59.216ms保持不变；早期5B动态warm-start从18降至11次状态评估，节省38.89%，三个transition独立确认warm均值比cold低约21.35%–23.99%。5B使用旧模型池、旧算法和分phase执行，没有持续流量零停机验证，不能当当前v2动态结果。',
        '**8．实验台账与可用程度**',
        table(['阶段','问题','证据类型','结果','来源'],[[s,q,t,v,link(p,'报告')] for s,q,t,v,p in inventory]),
        '**9．主线新增测量计数**',
        table(['阶段','实际新增timed runs','备注'],[['8C','96','72 search + 24 confirmation；另2次测量前失败'],['8D-2','101','既有数据精确复用不重复计'],['8D-3','9','固定状态路由/部署交叉诊断'],['8E-3','181','141 search + 40 confirmation'],['8E-4','6','只测消融新状态，0新确认'],['上述五阶段合计','393','测量前失败另2次；physical attempts=395']]),
        '393只覆盖上表五阶段，并非项目全生命周期run总数；早期profiling、质量审计、scaling和动态pilot具有数据复用关系，未与此台账盲目求和。',
        '**10．可用于汇报的结论与尚缺的证据**',
        table(['主张','当前证据'],[
            ['混合部署能改善Qwen-centric服务','支持：当前v1开发集6胜及旧offload结果'],
            ['v2改善受限预算部署搜索','支持：预冻结留出2胜4平0负'],
            ['SWAP与分类名额共同贡献收益','支持：验证后消融在mix_02的两个强度协同'],
            ['v2超过强specialist贪心','尚不支持：0胜3平3负，只缩小差距'],
            ['实测反馈纠正规划代理的误判','支持：8C有12个正Δz但无实际改善的候选'],
            ['主算法通过多副本选择获得收益','尚无这次主结果证据：x=2未实测/接受'],
            ['当前v2具有动态适应收益','未验证；只有旧5B pilot'],
            ['跨硬件、收紧显存或更高压力仍有效','未验证；同平台当前冻结范围'],
            ['数值VQA/LID SLO已满足','不能主张；只冻结语义准入'],
            ['统计显著、全局最优或普遍优越','现有设计不支持']]),
        '汇报时建议以“质量准入→v1基线暴露问题→诊断→v2留出→消融”的证据顺序组织。资源/负载扫描、路由修改、v2动态协议和第二硬件实验是后续待补项；本次未执行这些实验。',
        '**附：可复用文件**',
        f'- [留出完整数值CSV]({OUT}/heldout_v1_v2_B2.csv)\n- [30项消融比较CSV]({OUT}/ablation_comparisons.csv)\n- [v1开发集搜索与确认CSV]({OUT}/v1_development_search_and_confirmation.csv)\n- [v1开发集固定基线CSV]({OUT}/v1_development_fixed_baselines.csv)\n- [30阶段实验索引CSV]({OUT}/experiment_inventory.csv)\n- [留出均值/P95图]({OUT}/heldout_comparison.png)\n- [来源哈希与核验记录]({OUT}/provenance.json)',
    ]
    (OUT / '实验结果总览.md').write_text('\n\n'.join(text)+'\n')
    make_plot(models, current)
    manifest = dict(scope='Saved experimental results, no new GPU runs',
        current_workloads=len(current), ablation_comparison_rows=len(ablation_rows),
        experiment_inventory_rows=len(inventory), main_counts=dict(summary['counts']),
        assertions='PASS: repetition means, deltas, source comparison counts, n=2, steady/completion',
        sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(SOURCES)],
        outputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(OUT.iterdir())
                 if p.is_file() and p.name != 'provenance.json'])
    (OUT/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('sources','outputs')},ensure_ascii=False))


def make_plot(models, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,1,figsize=(11.5,8.5),sharex=True)
    colors = ('#75849a','#176fbd','#cc8c34')
    x = np.arange(6)
    for ax,metric,label in zip(axes,('weighted_mean_e2e','weighted_p95_e2e'),
                              ('Arrival-weighted mean E2E (ms)','Arrival-weighted task P95 (ms)')):
        for i,(key,name,color) in enumerate(zip(('m1','m2','b2'),('v1','v2','Specialist-greedy B2'),colors)):
            values=[models[key][r['workload']]['replicate_statistics'][metric] for r in rows]
            ax.bar(x+(i-1)*.25,[v['mean'] for v in values],.23,color=color,label=name,
                   yerr=[v['sample_std'] for v in values],capsize=3,error_kw={'linewidth':1})
        ax.set_ylabel(label)
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False,ncol=3,loc='upper right')
    axes[0].set_title('Held-out validation: v1, v2 and B2',loc='left',fontsize=16,pad=15)
    axes[1].set_xticks(x,['Mix 01\nModerate','Mix 01\nHigh','Mix 02\nModerate','Mix 02\nHigh','Mix 03\nModerate','Mix 03\nHigh'])
    fig.text(.09,.02,'Independent final confirmation: n=2. Error bars = sample standard deviation, not confidence intervals.\nSame final states reuse exact-compatible measurements; intensities within each mixture are correlated.',fontsize=9,color='#445066')
    fig.tight_layout(rect=(0,.07,1,1))
    for suffix in ('png','svg'):
        fig.savefig(OUT/f'heldout_comparison.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
