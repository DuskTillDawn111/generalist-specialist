"""Seal the error-only Poisson report and its reusable scientific figures."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path);parser.add_argument('--figures',required=True,type=Path)
    args=parser.parse_args();base=args.input.resolve();out=args.figures.resolve()
    if (out/'manifest.json').exists():raise FileExistsError('Already sealed')
    summary=json.loads((out/'summary.json').read_text());figures=summary['figures'];protocol=summary['protocol']
    tests=(base/'reporting-tests.log').read_text();assert 'Ran 3 tests' in tests and tests.rstrip().endswith('OK')
    source=json.loads((out/'sources.json').read_text())
    for r in source:assert sha(Path(r['path']))==r['sha256']
    assert sha(Path(protocol['fit']['path']))==sha(Path(protocol['original_fit']['path']))
    old=ROOT/'artifacts/static-study-gpu-20261022/confirmation-runs/strongest_alternative_route-seed45-b9e202d7a5ea/collection-config.json'
    oldcfg=json.loads(old.read_text());assert oldcfg['arrival_process']=='poisson'
    comparison=dict(old_formal_gpu_config=dict(path=str(old),sha256=sha(old),arrival_process=oldcfg['arrival_process'],
        rates=oldcfg['workload'],horizon_seconds=oldcfg['horizon_seconds']),
        current_validation=dict(arrival_process='poisson',rates=protocol['task_rates_rps'],horizon_seconds=protocol['horizon_seconds']),
        same_arrival_family=True,identical_load_or_horizon=False,
        old_scoped_confirmation_families=['stratified_jitter','paired_bursts'])
    (out/'arrival-protocol-comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
    rate=next(iter(protocol['task_rates_rps'].values()));n=summary['requests']
    lines=['# 冻结预测器：泊松到达下的预测误差','',
        f'本次按请求补做两次固定 GPU 验证，合计 **{n} 个正式请求**；使用冻结预测器，不训练、不调参、不按结果更换种子。五模型全部驻留、各一副本，每任务按 50% G2 / 50% 专用模型进行原 SWRR 路由。', '',
        '[五页科研误差图集](poisson_prediction_errors.pdf) · [逐请求 CSV](request-errors.csv) · [聚合时延 CSV](aggregate-errors.csv) · [能耗/吞吐 CSV](episode-errors.csv)','',
        '## 到达方式与训练分开','',
        '训练中的同时提交、错开提交、短突发及持续到达用于覆盖执行状态；不能用训练回放误差代替部署条件下的验证。本图只使用两次新的泊松到达实测。两轮计划和全部预测在首次 GPU 采集前一起封存。', '',
        '泊松到达复用早期静态部署、分流和正式 GPU 对照所用的 `gsdeploy.trace.make_trace(..., "poisson")`：每任务独立指数间隔，不固定实际请求数，不事后调整间隔，不筛选种子。', '',
        f'到达族一致；本轮负载与窗口保持当前五模型范围：每任务 {rate:.9f} rps、每轮 {protocol["horizon_seconds"]:.3f} 秒、每任务期望 24 个请求。早期对照示例为每任务 {next(iter(oldcfg["workload"].values())):.9f} rps、{oldcfg["horizon_seconds"]} 秒，两者不是相同强度/时长的逐项性能对比。', '',
        '此前五模型三候选确认的两种到达实际为 `stratified_jitter` 与 `paired_bursts`，其数据未混入这里，也没有改标为泊松。', '',
        '## 主图：逐请求平均、最小与最大误差','',
        '`APE_i = 100 × |预测端到端时延_i − 实测端到端时延_i| / 实测端到端时延_i`。端到端时延从计划到达时刻算至完成，包含排队。', '',
        '圆点为逐请求 APE 的算术平均，横线两端为最小值和最大值；每个请求权重相同，所有请求均纳入。图右侧直接写出 Mean / Min / Max / n。这不是带符号误差均值，不是置信区间，也不是平均时延的误差。', '',
        '|任务|请求数|平均误差 %|最小误差 %|最大误差 %|','|---|---:|---:|---:|---:|']
    for r in figures[0]['groups']:lines.append(f'|{r["label"]}|{r["n"]}|{r["mean_pct"]:.4f}|{r["min_pct"]:.4f}|{r["max_pct"]:.4f}|')
    lines += ['', '![逐请求误差](fig01_request_errors.png)','',
        '样本数是请求数，不是独立输入数或独立运行数。使用已有画像输入池，两轮只有两个到达历史，不据此给出总体显著性或未见输入泛化结论。', '',
        '## 聚合误差与逐请求误差分别报告','',
        '图 3、4 先在每个任务/每轮分别计算预测和实测的平均时延、P95 时延，再计算相对误差。图中均值/最小/最大是在两轮聚合误差上统计，每任务 n=2。P95 时延的误差不是“逐请求误差的 P95”。', '',
        '|任务|平均时延误差：均值 / 最小 / 最大 %|P95 时延误差：均值 / 最小 / 最大 %|','|---|---|---|']
    for a,b in zip(figures[2]['groups'],figures[3]['groups']):
        fmt=lambda r:f'{r["mean_pct"]:.4f} / {r["min_pct"]:.4f} / {r["max_pct"]:.4f}'
        lines.append(f'|{a["label"]}|{fmt(a)}|{fmt(b)}|')
    lines += ['', '整段能耗与固定窗口吞吐同样报告两轮误差的均值、最小与最大。能耗是 GPU board 从启动至 observation_end，包含 priming、正式窗口、排空和尾部；不是整机能耗。吞吐是窗口内完成数除以同一窗口长度，不是最大稳定容量。','',
        '|指标|样本数|平均误差 %|最小误差 %|最大误差 %|','|---|---:|---:|---:|---:|']
    for r in figures[4]['groups']:lines.append(f'|{r["label"]}|{r["n"]}|{r["mean_pct"]:.4f}|{r["min_pct"]:.4f}|{r["max_pct"]:.4f}|')
    lines += ['', '## 五张独立图','',
        '|图|内容|文件|','|---|---|---|']
    names=['逐请求误差，按任务与总体','逐请求误差，按 G2/专用执行路径','任务平均时延误差','任务 P95 时延误差','整段 GPU 能耗与固定窗口吞吐误差']
    for i,(m,label) in enumerate(zip(figures,names),1):
        stem=m['name'];lines.append(f'|{i}|{label}|[PDF]({stem}.pdf) · [SVG]({stem}.svg) · [PNG]({stem}.png)|')
    lines += ['', '所有图保留极端值，使用线性误差轴，不删点、不裁剪，不用候选总效用或 SLO 达标率替代预测误差。采用论文常见的简洁分面/点区间呈现；这里的区间严格是样本最小—最大值。PDF/SVG 为矢量，PNG 为 600 dpi。','',
        '## 验证边界与复现','',
        f'正式请求数按两轮为 {" / ".join(str(r["n_requests"]) for r in summary["episodes"])}。请求身份、路由、输入及预测前置封存检查通过。运行/输出审计问题记录为 `{summary["integrity_issues"]}`；详细审计见 capture-audit.json。误差计算的三项测试通过，覆盖绝对误差不抵消、极值保留和非法分母拒绝。','',
        '混合到达训练是设计选择，本轮没有与“只用泊松训练”做对照，因此不能由此单独证明混合训练提升了鲁棒性。结果也不授予全部七模型、多副本、其他负载或未见输入的预测资格。','',
        '源码见 source/，封存协议与哈希见 protocol.json、sources.json、manifest.json。CSV 保留未四舍五入的请求级误差和正负方向，便于重画。两次固定实验已结束，不因某项误差大而自动重训或补采。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    doc=ROOT/'docs/predictor_poisson_errors_20261105.md'
    doc.write_text('\n'.join(lines).replace('](','](../artifacts/predictor-poisson-error-figures-20261105/')+'\n')
    for name in ('protocol.json','capture-audit.json','reporting-tests.log'):
        shutil.copyfile(base/name,out/name)
    for name in ('prepare_poisson_predictor_validation_20261105.py','plot_poisson_predictor_errors_20261105.py','finalize_poisson_predictor_figures_20261105.py'):
        dst=out/'source'/name;dst.parent.mkdir(exist_ok=True);shutil.copyfile(ROOT/'experiments'/name,dst)
    import pymupdf as fitz
    from PIL import Image
    for m in figures:
        with fitz.open(out/f'{m["name"]}.pdf') as d:
            assert len(d)==1 and d[0].get_drawings() and not d[0].get_images()
            assert all(d.extract_font(f[0])[-1] for f in d[0].get_fonts())
        with Image.open(out/f'{m["name"]}.png') as im:assert all(abs(x-600)<1 for x in im.info['dpi'])
    with fitz.open(out/'poisson_prediction_errors.pdf') as d:
        assert len(d)==5 and all(p.get_text() and not p.get_images() for p in d)
    manifest=dict(status='COMPLETE',new_gpu_episodes=2,predictor_refits=0,prospective_predictions=2,
        new_predictions_during_analysis=0,requests=n,reporting_tests_passed=3,arrival_process='poisson',
        figures=5,exports=15,pdf_pages=5,source_hashes_unchanged=True,fit_unchanged=True,
        integrity_issues=summary['integrity_issues'],remaining_automatic_experiments=[],files=[])
    for p in sorted(out.rglob('*')):
        if p.is_file():manifest['files'].append(dict(path=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size))
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(out.with_suffix('.zip'),'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out.parent))
    with zipfile.ZipFile(out.with_suffix('.zip')) as z:assert z.testzip() is None
    print(json.dumps(dict(status='COMPLETE',requests=n,pdf=str(out/'poisson_prediction_errors.pdf'),report=str(doc))))


if __name__=='__main__':main()
