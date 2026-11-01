"""Posthoc b_t sensitivity of saved six-term GPU pairs; never changes contracts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import mmap
from pathlib import Path
import re
import statistics

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('text', 'image', 'ocr', 'asr')
TERMS = ('Q', 'T', 'L', 'P', 'M', 'E')
SEEDS = (71, 72, 73, 81, 82, 109, 110, 111, 112, 205, 206, 207, 208, 303, 304)
BASE = dict(text=1000., image=250., ocr=10000., asr=3000.)
INDEX = ROOT / 'artifacts/unified-weight-study-20261023/pair-scores-v2/declared-contrasts.csv'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def sections(path):
    """Read only small top-level sections of pretty-printed large raw files."""
    result = {}
    with Path(path).open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for key in ('config', 'per_task_metrics', 'evaluation', 'plan', 'status', 'measurement_window'):
            start = re.search(rb'^  "' + key.encode() + rb'": ', mm, re.M)
            if not start:
                raise ValueError(f'Missing section {key}: {path}')
            end = re.compile(rb'^  "[^"\\]+": ', re.M).search(mm, start.end())
            stop = end.start() if end else len(mm) - 2
            result[key] = json.loads(mm[start.end():stop].rstrip().rstrip(b','))
    return result


def gain(pair, refs):
    return pair['constant'] + math.fsum(pair['task_coefficients'][t] * BASE[t] / refs[t] for t in TASKS)


def maximin(pairs, k=None):
    # x_t = old_b_t/new_b_t. A common b_t applies to BOTH L and P and all pairs.
    bounds = [(1/k, k)] * 4 if k else [(0, None)] * 4
    r = linprog([0, 0, 0, 0, -1],
                A_ub=[[-p['task_coefficients'][t] for t in TASKS] + [1] for p in pairs],
                b_ub=[p['constant'] for p in pairs], bounds=bounds + [(None, None)],
                method='highs')
    if not r.success:
        return {'status': int(r.status), 'message': r.message}
    x = dict(zip(TASKS, map(float, r.x[:4])))
    recomputed = {str(p['seed']): p['constant'] + sum(p['task_coefficients'][t]*x[t] for t in TASKS) for p in pairs}
    margin = min(recomputed.values())
    assert abs(margin-r.x[-1]) < 1e-8
    return dict(status=0, margin=margin, x=x, per_seed=recomputed,
                b_ms={t: BASE[t]/x[t] if x[t] > 0 else None for t in TASKS},
                feasibility_residual=float(max(np.asarray(r.ineqlin.residual) * -1, default=0)))


def csv_save(path, rows):
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    sources = {str(INDEX): sha(INDEX)}
    rows = [r for r in csv.DictReader(INDEX.open()) if r['scheme'] == 'original_equal'
            and int(r['seed']) in SEEDS
            and (int(r['seed']) < 303 or r['comparison_kind'] == 'policy_allowed_text_route_toggle_metadata_match')]
    assert len(rows) == len(SEEDS) and len({r['seed'] for r in rows}) == len(SEEDS)
    pairs = []; catalog = None; task_rows = []
    for row in rows:
        a, b = [], []
        raw = []
        for role in ('spill', 'control'):
            path = ROOT / 'artifacts' / row[role]
            sources[str(path)] = sha(path)
            raw.append(sections(path))
        spill, control = raw
        rates = spill['config']['workload']; total = math.fsum(rates.values())
        for r in raw:
            assert r['config']['latency_reference_ms'] == BASE
            assert r['config']['workload'] == rates
            assert all(abs(r['config']['weights'][t]-1/6) < 1e-14 for t in TERMS)
            assert r['evaluation']['utility']['eligible'] is True
        assert spill['plan']['deployment'] == control['plan']['deployment']
        if int(row['seed']) == 303:
            catalog = spill['config']['catalog']
        su, cu = [r['evaluation']['utility'] for r in raw]
        coeff = {}; lt = {}; pt = {}; details = {}
        for t in TASKS:
            sa, cb = [r['per_task_metrics'][t] for r in raw]
            alpha = rates[t]/total
            lt[t] = alpha*(cb['mean_e2e_latency_ms']-sa['mean_e2e_latency_ms'])/BASE[t]/6
            pt[t] = alpha*(cb['p95_e2e_latency_ms']-sa['p95_e2e_latency_ms'])/BASE[t]/6
            coeff[t] = lt[t]+pt[t]
            detail = dict(seed=int(row['seed']), task=t, alpha=alpha, old_b_ms=BASE[t],
                          mean_spill_ms=sa['mean_e2e_latency_ms'], mean_control_ms=cb['mean_e2e_latency_ms'],
                          p95_spill_ms=sa['p95_e2e_latency_ms'], p95_control_ms=cb['p95_e2e_latency_ms'],
                          old_L_gain=lt[t], old_P_gain=pt[t], old_combined_gain=coeff[t],
                          spill_observations=sa['completed_requests'], control_observations=cb['completed_requests'])
            details[t] = detail; task_rows.append(detail)
        constant_terms = {t: (cu[t+'_norm']-su[t+'_norm'])/6 for t in 'QTME'}
        constant = math.fsum(constant_terms.values())
        base_gain = constant+math.fsum(coeff.values())
        assert math.isclose(base_gain, float(row['delta_U']), abs_tol=2e-10)
        for values, term in ((lt, 'L'), (pt, 'P')):
            assert math.isclose(math.fsum(values.values()), (cu[term+'_norm']-su[term+'_norm'])/6, abs_tol=2e-10)
        scalar = math.fsum(coeff.values())
        pairs.append(dict(seed=int(row['seed']), study=row['study'], comparison_kind=row['comparison_kind'],
                          spill=row['spill'], control=row['control'], original_delta_U=base_gain,
                          constant=constant, constant_terms=constant_terms, task_coefficients=coeff,
                          uniform_threshold_k=(-scalar/constant if constant else None),
                          uniform_formula=f'{constant:.17g} + ({scalar:.17g})/k',
                          details=details))
    pairs.sort(key=lambda p: p['seed'])
    groups = {'fixed_validation': [109,110,111,112], 'queue_aware': [71,72,73,81,82],
              'refined': [205,206,207,208], 'coupled': [303,304]}
    # Diagnostic domains; these factors are NOT asserted to be reasonable SLOs.
    box_rows = []; box_solutions = {}
    for name, seeds in groups.items():
        selected = [p for p in pairs if p['seed'] in seeds]
        box_solutions[name] = {}
        for k in (1,2,5,10,20,50,100,250,500,1000):
            z = maximin(selected, k); box_solutions[name][str(k)] = z
            box_rows.append(dict(group=name, k=k, margin=z.get('margin'), **z.get('b_ms', {})))
        box_solutions[name]['unbounded'] = maximin(selected)
    recent = [p for p in pairs if p['seed'] in (303,304)]
    lo, hi = 1., 1000.
    for _ in range(65):
        mid = (lo+hi)/2
        if maximin(recent, mid)['margin'] > 0: hi=mid
        else: lo=mid
    threshold = dict(k=hi, solution=maximin(recent, hi),
                     positive_witness=maximin(recent, 250),
                     method='Numerical bisection of bounded LP; zero-margin boundary is not strict benefit')

    # Scales anchored to existing seven-model n=1 standalone service measurements.
    # They are alternative normalization diagnostics, NOT validated user SLOs.
    schemes = {'original': BASE}
    for model in ('generalist_2','generalist_3'):
        schemes[model+'_n1_mean'] = {t: catalog['models'][model]['service_time_ms'][t] for t in TASKS}
    for method in ('min','median','max'):
        fn = {'min':min,'median':statistics.median,'max':max}[method]
        schemes['supported_paths_n1_'+method] = {t:fn(m['service_time_ms'][t] for m in catalog['models'].values() if t in m['supported_tasks']) for t in TASKS}
    for k in (2,5,10,20,50,100):
        schemes['uniform_'+str(k)] = {t:k*BASE[t] for t in TASKS}
    scheme_rows = []
    for name, refs in schemes.items():
        for pair in pairs:
            scheme_rows.append(dict(scheme=name, seed=pair['seed'], delta_U=gain(pair,refs), **refs))
    uniform_rows = [dict(k=k, **{str(p['seed']):gain(p,{t:k*BASE[t] for t in TASKS}) for p in pairs})
                    for k in (1,2,5,10,20,50,100,1000)]
    analysis = dict(status='POSTHOC_MEASURED_REFERENCE_SENSITIVITY',
                    formula='DeltaU_i = constant_i + sum_t coefficient_it * old_b_t/new_b_t',
                    objective='six equal weights; each task b_t shared by mean and P95',
                    scope='Fixed measured routes only. Groups are not pooled as repeat trials. No new GPU run, search, or SLO validation.',
                    unknown=dict(seed209='UNKNOWN, not included or filled with zero'),
                    excluded='Historical five-term 401/402 and fixed development107/108 not analyzed here.',
                    base_b_ms=BASE, pairs=pairs, groups=groups, box_solutions=box_solutions,
                    coupled_smallest_symmetric_factor=threshold, schemes=schemes, uniform=uniform_rows,
                    sources=sources)
    (output/'analysis.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    csv_save(output/'task-gains.csv', task_rows)
    csv_save(output/'reference-schemes.csv', scheme_rows)
    csv_save(output/'uniform-scan.csv', uniform_rows)
    csv_save(output/'bounded-search.csv', box_rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig, axes = plt.subplots(1,2,figsize=(8.2,3.1),layout='constrained')
    xx = np.geomspace(1,1000,500)
    for p in recent:
        yy = [p['constant']+sum(p['task_coefficients'].values())/x for x in xx]
        axes[0].plot(xx,yy,label=f"Seed {p['seed']}")
    axes[0].axhline(0,color='black',lw=.7)
    axes[0].set(xscale='log',yscale='symlog',xlabel=r'Common scale $k$ ($b_t^{new}=k b_t^{old}$)',ylabel=r'Paired $\Delta U$',title='(a) Uniform reference scaling')
    axes[0].legend(frameon=False)
    for name in ('refined','coupled'):
        xx2=np.geomspace(1,1000,80)
        yy2=[maximin([p for p in pairs if p['seed'] in groups[name]],float(k))['margin'] for k in xx2]
        axes[1].plot(xx2,yy2,label=name.capitalize())
    axes[1].axhline(0,color='black',lw=.7)
    axes[1].set(xscale='log',yscale='symlog',xlabel=r'Allowed factor $k$: $b_t^{new}/b_t^{old}\in[1/k,k]$',ylabel=r'Best minimum paired $\Delta U$',title='(b) Independent task references')
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_yscale('symlog',linthresh=1e-4);ax.grid(alpha=.22,lw=.5)
    for ext in ('png','pdf','svg'):
        fig.savefig(output/f'reference-sensitivity.{ext}',dpi=240,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)

    lines=['# 参考延迟的实测事后敏感性分析','',
           '保留六项等权和 P95，仅重评分已完成实测，不改原始记录/合同/路由，不运行 GPU。',
           '每任务的同一个 b 同时进入平均与 P95；四任务的参考可分别调整，但所有方案/种子共用同一组参考。',
           '以下“配对全正”仅为已见数据上的确定性条件，不代表统计稳定或新目标下重新搜索最优。','',
           '## 最近耦合分流 303/304','',
           '|统一放大倍数 k|303 ΔU|304 ΔU|','|---:|---:|---:|']
    for r in uniform_rows:
        lines.append(f"|{r['k']}|{r['303']:.9g}|{r['304']:.9g}|")
    lines += ['',f"独立调四任务参考，限制每项新/旧参考位于 [1/k,k]：最小共同转正幅度边界 k≈{hi:.9g}；零边界不计收益。",'',
              '|边界/见证|QA ms|Image ms|OCR ms|ASR ms|303 ΔU|304 ΔU|','|---|---:|---:|---:|---:|---:|---:|']
    for label,z in [('边界',threshold['solution']),('k=250 可行见证',threshold['positive_witness'])]:
        lines.append('|'+label+'|'+ '|'.join(f"{z['b_ms'][t]:.8g}" for t in TASKS)+'|'+ '|'.join(f"{z['per_seed'][str(s)]:.9g}" for s in (303,304))+'|')
    lines += ['','这里 bQA/bImage 大幅增大、bOCR 大幅减小，是收益驱动的极端偏好见证，不能称合理 SLO。',
              '303/304 的 OCR 均只有很少请求，task-gains.csv 保留样本数；微小 OCR 改善被极小分母放大。','',
              '## 用既有单模型画像构造参考','',
              '固定模型的 n=1 平均服务时间以及各任务全池支持路径的 min/median/max，仅用作可解释替代尺度诊断；它们不是端到端 SLO。','',
              '|参考方案|QA ms|Image ms|OCR ms|ASR ms|303 ΔU|304 ΔU|','|---|---:|---:|---:|---:|---:|---:|']
    for name,refs in schemes.items():
        if name.startswith('uniform'):continue
        lines.append('|'+name+'|'+ '|'.join(f'{refs[t]:.6g}' for t in TASKS)+'|'+ '|'.join(f'{gain(p,refs):.9g}' for p in recent)+'|')
    lines += ['','## 历史配对与不可行性','',
              '205–208 与 303/304 分组研究，不跨策略阶段合并均值。208 的四任务合并延迟收益系数全部为负，非延迟常数也为负，故任何正的四任务 b 都不能让该配对转正。',
              '205 原六项聚合的支配证书不能直接外推到按任务分别改变分母的空间：它有少量 ASR/OCR 正向时延变化。本分析保留这些任务细节。',
              '209 仍为 UNKNOWN；401/402 五项历史结果未纳入。另附71/72/73/81/82与109–112完整数据，分组保留全部负结果。','',
              '统一放大 b 等价于共同降低 L/P 相对权重；它只能改变偏好，无法缩短实际时间或修复 GPU 竞争。若采纳新参考，应以业务时延要求或独立冻结的参考部署为依据，再重新优化部署/路由并验证。','',
              '![参考延迟敏感性](reference-sensitivity.png)','',
              '精确来源与 SHA256 见 analysis.json；逐任务时间及贡献见 task-gains.csv；所有方案重评分见 reference-schemes.csv。']
    (output/'report.zh.md').write_text('\n'.join(lines)+'\n')
    manifest={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    manifest['script_sha256']=sha(__file__)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(output),'pairs':len(pairs),'threshold':threshold,'checks':'15 baseline and per-term arithmetic checks passed'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
