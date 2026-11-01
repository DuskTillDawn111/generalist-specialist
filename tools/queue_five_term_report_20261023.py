"""Read-only Chinese five-term native report exporter; no frozen package imports.

Every invocation creates a new directory. Only raw receipts already cited by an
analysis are consumed; unmatched or unaudited pairs stay PENDING.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/queue-five-term-study-20261023'
TERMS = tuple('QTLME')
WEIGHTS = dict(Q=.2, T=.2, L=.2, P=0., M=.2, E=.2)
OBJECTIVE = 'gsdeploy.five_term_no_tail.v1'


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)


def receipt(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return dict(path=str(path), sha256=h.hexdigest())


class Inputs:
    def __init__(self):
        self.refs = {}

    def read(self, path, expected=None):
        path = Path(path).resolve()
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
        stamp = lambda s: (s.st_ino, s.st_size, s.st_mtime_ns)
        if stamp(before) != stamp(after):
            raise ValueError('Input changed while reading: ' + str(path))
        ref = dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())
        if expected is not None and ref['sha256'] != expected['sha256']:
            raise ValueError('Receipt mismatch: ' + str(path))
        if str(path) in self.refs and self.refs[str(path)] != ref:
            raise ValueError('Conflicting input version: ' + str(path))
        self.refs[str(path)] = ref
        return json.loads(payload)


def percentile(values, q=.95):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values)-1)*q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi]-values[lo])*(pos-lo)


def cohort_signature(rows):
    cohort = [[r.get(k) for k in ('request_id','task','pool_id','sample_id','scheduled_offset_seconds')] for r in rows]
    cohort.sort(key=lambda r: json.dumps(r, ensure_ascii=False))
    return hashlib.sha256(json.dumps(cohort, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def summarize_raw(raw, job, phase):
    ev = raw.get('evaluation') or {}
    u = ev.get('utility') or {}
    rows = raw.get('request_records') or []
    families = raw.get('replica_to_family') or {}
    routing = Counter(r.get('task','UNKNOWN')+'->'+r.get('model','UNKNOWN') for r in rows)
    family_routing = Counter(r.get('task','UNKNOWN')+'->'+families.get(r.get('model'),'UNKNOWN') for r in rows)
    per_task = {}
    quality = (u.get('quality_evidence') or {}).get('per_task') or {}
    for task in ('text','image','ocr','asr'):
        selected = [r for r in rows if r.get('task') == task]
        completed = [r for r in selected if r.get('status') == 'completed']
        latencies = [(r['completion_time_ns']-r['scheduled_time_ns'])/1e6 for r in completed
                     if r.get('completion_time_ns') is not None and r.get('scheduled_time_ns') is not None]
        loss = quality.get(task,{}).get('loss')
        per_task[task] = dict(requests=len(selected), completed=len(completed),
                              mean_e2e_ms=statistics.fmean(latencies) if latencies else None,
                              p95_e2e_ms=percentile(latencies), quality_loss=loss,
                              quality_metric={'text':'EM','image':'accuracy','ocr':'CER','asr':'WER'}[task],
                              quality_value=(1-loss if task in {'text','image'} else loss) if finite(loss) else None,
                              error_numerator=quality.get(task,{}).get('error_numerator'),
                              reference_denominator=quality.get(task,{}).get('reference_denominator'))
    text_total = per_task['text']['requests']
    spills = sum(n for route,n in family_routing.items() if route.startswith('text->generalist_'))
    stability = ev.get('stability') or {}
    se = ev.get('stability_evidence') or {}
    window = raw.get('measurement_window') or {}
    completed = [r for r in rows if r.get('status') == 'completed']
    within = sum(r.get('completion_time_ns',math.inf) <= window.get('end_time_ns',-math.inf) for r in completed)
    latencies = [(r['completion_time_ns']-r['scheduled_time_ns'])/1e6 for r in completed
                 if r.get('completion_time_ns') is not None and r.get('scheduled_time_ns') is not None]
    examples = []
    gates = Counter()
    for r in rows:
        routing_decision = r.get('routing_decision') or {}
        decision = routing_decision.get('decision') or {}
        gates.update(decision.get('gates',[]))
        if len(examples) >= 2 or r.get('task') != 'text' or not families.get(r.get('model'),'').startswith('generalist_'):
            continue
        snapshot = routing_decision.get('snapshot') or {}
        counts = {a:len(v.get('outstanding',[])) for a,v in snapshot.items()}
        specs = [a for a,f in families.items() if f == 'text']
        if len(specs) < 2 or not all(counts.get(a,0) >= 1 for a in specs) or counts.get(r['model'],1) != 0:
            continue
        examples.append(dict(label=job['label'], request_id=r['request_id'], selected_alias=r['model'],
                             outstanding_counts=counts, decision=decision,
                             scope='Host-visible queue mechanism only; not GPU idle or counterfactual benefit proof'))
    result = dict(phase=phase,label=job['label'],role=job['role'],seed=job['seed'],status=raw.get('status'),
                  eligible=u.get('eligible',False),U=u.get('U'),diagnostic_U=u.get('diagnostic_U'),
                  causal_verified=(ev.get('dynamic_audit') or {}).get('verified'),
                  stability=stability.get('stability_state_v2'),stability_reasons=stability.get('stability_reason_v2'),
                  raw_errors=raw.get('errors'), offered_requests=len(rows),completed_requests=len(completed),
                  completed_within_window=within,completed_during_drain=len(completed)-within,
                  mean_e2e_ms=statistics.fmean(latencies) if latencies else None,
                  actual_text_spills=spills,text_spill_fraction=spills/text_total if text_total else None,
                  text_0_requests=routing.get('text->text#0',0),text_1_requests=routing.get('text->text#1',0),
                  routing_counts=dict(routing),family_routing_counts=dict(family_routing),gates=dict(gates),
                  end_backlog=se.get('end_backlog_requests'),final_backlog_after_drain=se.get('final_backlog_after_drain'),
                  tail_backlog_slope=se.get('tail_backlog_slope'),drain_seconds=window.get('drain_seconds'),
                  energy_window_joules=(raw.get('energy_evidence') or {}).get('run_joules'),
                  startup_joules=(raw.get('energy_evidence') or {}).get('startup_joules'),
                  drain_energy_joules=(raw.get('cohort_energy_diagnostic') or {}).get('drain_joules'),
                  safe_memory_mib=(raw.get('memory_evidence') or {}).get('safe_memory_mib'),
                  measurement_window=window,cohort_signature=cohort_signature(rows),per_task=per_task,
                  mechanism_examples=examples,config=raw.get('config'),plan=raw.get('plan'))
    result.update({t+'_norm':u.get(t+'_norm') for t in (*TERMS,'P')})
    if raw.get('config',{}).get('weights') != WEIGHTS or u.get('objective_schema') != OBJECTIVE:
        raise ValueError('Raw objective identity mismatch: '+job['label'])
    if u.get('eligible') and (not close(u.get('U'),-math.fsum(.2*result[t+'_norm'] for t in TERMS))
                              or not close(u.get('P_contribution'),0.)):
        raise ValueError('Raw five-term arithmetic mismatch: '+job['label'])
    return result


def checked_pairs(audit, runs, phase):
    by_label = {r['label']:r for r in runs}
    result = []
    for p in audit['summary']['paired']:
        on = by_label.get(f"{p['strategy']}-seed{p['seed']}")
        off_role = {'coupled_spill':'matched_no_spill','legacy_latency':'legacy_no_spill'}[p['strategy']]
        off = by_label.get(f"{off_role}-seed{p['seed']}")
        valid = bool(p.get('both_evidence_valid_eligible') and on and off and on.get('audit_verified')
                     and off.get('audit_verified') and on.get('eligible') and off.get('eligible'))
        row = dict(phase=phase,strategy=p['strategy'],seed=p['seed'],status='VALID' if valid else 'PENDING_OR_INVALID',
                   same_cohort=p.get('same_cohort'),actual_text_spills=p.get('actual_text_spills'),
                   control_text_spills=p.get('control_text_spills'),delta_U=None)
        if valid:
            if p.get('actual_text_spills') != on.get('actual_text_spills') or p.get('control_text_spills') != off.get('actual_text_spills'):
                raise ValueError('Pair routing count mismatch')
            if not p.get('same_cohort') or on['cohort_signature'] != off['cohort_signature']:
                raise ValueError('Pair cohort mismatch')
            delta = on['U']-off['U']
            contributions = p.get('contribution_to_delta_U') or {}
            deltas = p.get('term_deltas_spill_minus_control') or {}
            for t in TERMS:
                expected = on[t+'_norm']-off[t+'_norm']
                if not close(deltas.get(t),expected) or not close(contributions.get(t),-.2*expected):
                    raise ValueError('Pair contribution mismatch: '+t)
                row['delta_'+t] = deltas[t]
                row['contribution_'+t] = contributions[t]
            if not close(math.fsum(contributions[t] for t in TERMS),delta) or not close(p.get('delta_U'),delta):
                raise ValueError('Pair delta_U mismatch')
            row.update(delta_U=delta,U_on=on['U'],U_off=off['U'],contribution_sum=math.fsum(contributions[t] for t in TERMS))
        result.append(row)
    return result


def phase_data(inputs, analysis_path, audit_path, expected_phase):
    analysis = inputs.read(analysis_path)
    audit = inputs.read(audit_path)
    packet = inputs.read(audit['packet']['path'],audit['packet'])
    if analysis.get('phase') != expected_phase or packet.get('phase') != expected_phase or analysis.get('weights') != WEIGHTS:
        raise ValueError('Analysis/packet phase or weight mismatch')
    observed = {r['label']:r for r in analysis['observations']}
    if len(observed) != len(analysis['observations']):
        raise ValueError('Duplicate analysis labels')
    audited = {r['label']:r for r in audit['observations']}
    if len(audited) != len(audit['observations']):
        raise ValueError('Duplicate audit labels')
    if set(observed)-{j['label'] for j in packet['jobs']}:
        raise ValueError('Analysis contains undeclared jobs')
    runs=[]
    for job in packet['jobs']:
        obs = observed.get(job['label'])
        ar = audited.get(job['label']) or {}
        if not obs or not obs.get('raw'):
            runs.append(dict(phase=expected_phase,label=job['label'],role=job['role'],seed=job['seed'],
                             status='PENDING',audit_verified=False,audit_status=ar.get('status','PENDING')))
            continue
        raw = inputs.read(obs['raw']['path'],obs['raw'])
        if raw.get('experiment_job_id') != job['id'] or raw.get('experiment_phase') != expected_phase:
            raise ValueError('Raw frozen job identity mismatch: '+job['label'])
        run = summarize_raw(raw,job,expected_phase)
        run['raw'] = obs['raw']
        run['audit_status'] = ar.get('status','PENDING')
        run['audit_errors'] = ar.get('errors',[])
        if ar.get('raw') and ar['raw'] != obs['raw']:
            raise ValueError('Analysis/audit raw receipt mismatch: '+job['label'])
        diagnostics = ar.get('diagnostics') or {}
        if ar.get('verified') and any(not close(diagnostics.get('terms',{}).get(t),run.get(t+'_norm')) for t in (*TERMS,'P')):
            raise ValueError('Audit/raw normalized term mismatch: '+job['label'])
        if ar.get('verified') and (diagnostics.get('eligible') != run['eligible'] or
                                  (run['eligible'] and not close(diagnostics.get('recorded_U'),run['U']))):
            raise ValueError('Audit/raw utility mismatch: '+job['label'])
        run['audit_verified'] = bool(ar.get('verified') and ar.get('raw') == obs['raw'] and audit.get('source_input_bindings',{}).get('verified'))
        if obs.get('eligible') != run['eligible'] or (run['eligible'] and not close(obs.get('U'),run['U'])):
            raise ValueError('Analysis/raw utility mismatch: '+job['label'])
        runs.append(run)
    pairs = checked_pairs(audit,runs,expected_phase)
    terminal = all(r['status'] not in {'PENDING','running','starting'} for r in runs)
    valid = terminal and all(r.get('audit_verified') and r.get('eligible') and r.get('causal_verified') for r in runs)
    primary = [p for p in pairs if p['strategy']=='coupled_spill']
    expected_pairs = 2 if expected_phase=='development' else 6
    threshold = .0001 if expected_phase=='development' else 0.
    endpoint = bool(valid and len(packet['jobs']) == (8 if expected_phase=='development' else 12) and len(primary)==expected_pairs and all(p['status']=='VALID' and p['delta_U']>threshold
                    and (p.get('actual_text_spills') or 0)>0 and p.get('control_text_spills')==0 for p in primary))
    audit_terminal = all(r.get('audit_status') == 'AUDITED' for r in runs)
    return dict(phase=expected_phase,audit_terminal=audit_terminal,runs=runs,pairs=pairs,terminal=terminal,valid=valid,endpoint=endpoint,
                expected_jobs=len(packet['jobs']),terminal_jobs=sum(r['status'] not in {'PENDING','running','starting'} for r in runs),
                audited_jobs=sum(r.get('audit_verified',False) for r in runs),packet=packet,audit=audit,analysis=analysis)


def fmt(value):
    if value is None:
        return '—'
    if isinstance(value,bool):
        return '是' if value else '否'
    if finite(value):
        return f'{value:.9g}'
    return str(value).replace('|','\\|').replace('\n',' ')


def table(headers, rows):
    return ['|'+'|'.join(headers)+'|','|'+'|'.join('---' for _ in headers)+'|',
            *('|'+'|'.join(fmt(v) for v in row)+'|' for row in rows),'']


def csv_text(rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    return stream.getvalue()


def role_rankings(runs, seeds):
    """Rank only after every declared development role/seed is validly audited."""
    roles=('coupled_spill','matched_no_spill','legacy_latency','legacy_no_spill')
    lookup={(r['role'],r['seed']):r for r in runs}
    rankings=[]
    for role in roles:
        selected=[lookup.get((role,seed)) for seed in seeds]
        complete=bool(seeds) and all(r and r.get('status')=='completed' and r.get('eligible')
                    and r.get('audit_verified') and r.get('causal_verified') and finite(r.get('U')) for r in selected)
        row=dict(phase='development',role=role,expected_seeds=list(seeds),complete_valid_audited=complete,
                 seed_U={str(seed):r.get('U') if r and r.get('eligible') and r.get('audit_verified') else None
                         for seed,r in zip(seeds,selected)},
                 mean_U=statistics.fmean(r['U'] for r in selected) if complete else None,rank=None,
                 ranking_status='PENDING_OR_INVALID',scope='Observed four-role fixed-resident development domain only')
        for seed in seeds:
            row['seed_'+str(seed)+'_U']=row['seed_U'][str(seed)]
        rankings.append(row)
    if all(r['complete_valid_audited'] for r in rankings):
        for row in rankings:
            row['rank']=1+sum(other['mean_U']>row['mean_U'] for other in rankings)
            row['ranking_status']='COMPLETE_FINITE_OBSERVATION'
    return rankings


def render(development, confirmation, cpu, status):
    phases=[development]+([confirmation] if confirmation else [])
    dev=development
    if status=='PENDING':
        opening=f"PENDING：开发终结记录{dev['terminal_jobs']}/{dev['expected_jobs']}，有效外审{dev['audited_jobs']}/{dev['expected_jobs']}；最终开发/确认结论尚未闭合。"
    elif not dev['endpoint']:
        opening='开发已终结，主策略未通过预声明门禁；本轮不进入独立确认，不能声明稳定分流收益。'
    else:
        opening='开发门禁通过；'+('独立确认达到全部六组有效、实际分流且ΔU>0终点。' if confirmation and confirmation['endpoint'] else '独立确认尚未达到预声明终点。')
    raw_run=next((r for r in dev['runs'] if r.get('config')),None)
    cfg=raw_run['config'] if raw_run else {}
    lines=['# 五项等权在线路由实验报告','',opening,'',
           '目标固定Q/T/L/M/E各0.2，P=0；P95只作诊断。效用较大较好，U为负加权成本和，负数本身不代表失败。配对ΔU=U_on−U_off>0才表示本目标改善，不使用负基线U的比值作为收益率。内部six_term_quality_v1只是兼容编码。','',
           '## 目标、负载与部署','',
           '来源池为generalist_1、generalist_2、generalist_3及text/image/ocr/asr四专用。本轮预先固定驻留text×2 + generalist_2×1 + ocr×1；没有逐轮搜索七池部署或比较纯通、纯专。',
           '负载text=1.85、image/asr/ocr各0.01 rps，Poisson到达；预热30秒、测量600秒、排空上限300秒。背景image/asr→G2、ocr→OCR；文本动态路由到两个text副本或G2。同种子cohort相同不代表执行历史相同。','',
           '设a_t=λ_t/Σλ、h=测量秒数，所有归一项是成本：Q=Σa_t loss_t；T=1−测量窗内成功完成数/测量cohort到达数；L=Σa_t mean_e2e_t/reference_t；M=安全系数×实测拥有显存峰值/GPU显存预算；E=(测量窗板卡J/h+启动J/摊销秒数)/参考W；U=−0.2(Q+T+L+M+E)。P=Σa_t p95_t/reference_t，权重0。排空能耗另报，不进入正式E；排空后全部完成不等于T=0。',
           'text原始指标是EM、image是accuracy，损失各为1−指标；OCR是CER、ASR是WER，损失直接为错误率，不能统称精度。归一Q是到达率加权成本，与某一任务原始质量指标不同。在线冻结族级质量代理也不同于本轮输出实测。原rho≤0.8先验、稳定性和质量证据要求保持，无绝对质量准入。','']
    if cfg:
        lines += ['实际模型标识：','',*['- '+m+'：'+v['name'] for m,v in cfg['catalog']['models'].items()], '',
                  '时延参考(ms)：'+json.dumps(cfg['latency_reference_ms'],ensure_ascii=False)+'；能耗参考W='+str(cfg['energy_reference_watts'])+'；启动摊销秒数='+str(cfg['startup_amortization_seconds'])+'。','']
    lines += ['## 四种路由角色','',*table(['角色','评分/控制','文本给G2'],[
        ['coupled_spill','新五项条件代理；mean+p90、margin0、cooldown30秒','允许；受暴露/证据门控'],
        ['matched_no_spill','相同耦合算法，仅关闭分流','禁止'],
        ['legacy_latency','冻结延迟启发式与guard','允许'],
        ['legacy_no_spill','与legacy相同延迟评分与guard，仅去G2候选','禁止']]),
        '主比较分别为coupled_spill−matched_no_spill、legacy_latency−legacy_no_spill；legacy更好也不能替换确认主策略。只用正式eligible U比较，diagnostic_U不排名。','']
    lines += ['## 四角色开发观测排名','',
              '每个角色只有全部声明seed完整、eligible、因果重放及外审有效才计算均值；四角色域全部完成前不给任何名次。较大U较优，精确同均值共享名次；该排名只描述固定驻留有限观测域，不是七池部署最优。','',
              *table(['角色',*[str(seed)+' U' for seed in dev['packet']['seeds']],'均值U','排名','状态'],[
                  [r['role'],*[r['seed_U'][str(seed)] for seed in dev['packet']['seeds']],r['mean_U'],r['rank'],r['ranking_status']]
                  for r in dev['rankings']])]
    lines += ['在线局部评分只预测已观察请求前缀与当前队列；未来到达不建模。物理峰值M随路由的变化未知，在线使用共同驻留size代理，不能声称真实M相同；共同过去能耗与启动能耗不参加局部差分。最终U使用本轮实测质量、截止窗完成、时延、显存和带启动摊销的能耗。','']
    for phase in phases:
        lines += ['## '+('开发' if phase is dev else '独立确认')+'逐轮结果','',
                  f"终结{phase['terminal_jobs']}/{phase['expected_jobs']}，有效外审{phase['audited_jobs']}/{phase['expected_jobs']}。",'',
                  *table(['角色/seed','状态/eligible/外审','U','Q','T','L','M','E','P诊断','文本→G2','text#0/#1','稳定性/因果重放'],[
                      [r['label'],str(r['status'])+'/'+fmt(r.get('eligible'))+'/'+fmt(r.get('audit_verified')),r.get('U'),
                       *[r.get(t+'_norm') for t in TERMS],r.get('P_norm'),str(r.get('actual_text_spills','—'))+'/'+fmt(r.get('text_spill_fraction')),
                       str(r.get('text_0_requests','—'))+'/'+str(r.get('text_1_requests','—')),str(r.get('stability','—'))+'/'+fmt(r.get('causal_verified'))] for r in phase['runs']]),
                  '### 实际请求量、原始质量和平均时延','',
                  *table(['角色/seed','任务','实际请求/完成','质量指标','原始值','mean e2e ms','p95 e2e ms'],[
                      [r['label'],t,str(p['requests'])+'/'+str(p['completed']),p['quality_metric'],p['quality_value'],p['mean_e2e_ms'],p['p95_e2e_ms']]
                      for r in phase['runs'] for t,p in r.get('per_task',{}).items()]),
                  '### 配对与五项贡献','',
                  '贡献=−0.2×(on成本−off成本)，贡献和已与ΔU核对。P不参与贡献。PENDING_OR_INVALID不宣布收益；完整个别配对也不代表开发终点。','',
                  *table(['策略/seed','状态/同cohort','U_on','U_off','ΔU','Q贡献','T贡献','L贡献','M贡献','E贡献'],[
                      [p['strategy']+'/'+str(p['seed']),p['status']+'/'+fmt(p['same_cohort']),p.get('U_on'),p.get('U_off'),p['delta_U'],
                       *[p.get('contribution_'+t) for t in TERMS]] for p in phase['pairs']])]
        valid_pairs=[p for p in phase['pairs'] if p['status']=='VALID']
        for strategy in ('coupled_spill','legacy_latency'):
            values=[p['delta_U'] for p in valid_pairs if p['strategy']==strategy]
            if values:
                lines += [strategy+f"：有效配对n={len(values)}，平均ΔU={statistics.fmean(values):.12g}，观测范围[{min(values):.12g}, {max(values):.12g}]。范围不是置信区间；小样本不据此证明可重复收益。",'']
        lines += ['### 分流决策实例','']
        examples=[e for r in phase['runs'] for e in r.get('mechanism_examples',[])]
        if not examples:
            lines += ['当前证据没有可展示的双专用outstanding且所选G2队列为空实例；不补造反事实。','']
        for e in examples[:4]:
            lines += ['- '+e['label']+' / '+e['request_id']+'：选择'+e['selected_alias']+'，可见outstanding='+json.dumps(e['outstanding_counts'],ensure_ascii=False)+'。完整mean/p90候选与贡献、门控、cooldown见mechanism-examples.json。']
        lines += ['', '实例只说明决策时host可见队列状态与主动分流，G2队列为空不等于整块GPU空闲；后验完成时间不能证明未选路径反事实时延。效用收益依赖上面自身配对。','']
        for r in phase['runs']:
            if r.get('raw_errors') or r.get('audit_errors') or r['status']=='PENDING':
                lines += ['- '+r['label']+'：'+json.dumps(dict(status=r['status'],raw_errors=r.get('raw_errors'),audit_status=r.get('audit_status'),audit_errors=r.get('audit_errors')),ensure_ascii=False)]
        lines += ['']
    lines += ['## 开发门禁与确认','',
              '开发门禁要求8轮全部完整有效、coupled两个自身配对都实际分流且ΔU>0.0001；该阈值是探索实用下限，不是噪声置信界。确认固定421–426六组，要求全部有效、实际分流且ΔU>0；不增补种子掩盖负组。',
              '独立确认状态：'+('未提供确认分析；开发门禁未过时不执行，门禁仍PENDING时尚待裁定。' if not confirmation else ('完成且达到终点。' if confirmation['endpoint'] else '已提供证据，但尚未达到终点；逐组状态如上。')),'']
    if confirmation and confirmation['endpoint']:
        lines += ['六组均正的名义单侧符号检验p=1/64=0.015625，依赖独立、对称符号等假设；共享硬件状态与短序列重复限制解释。','']
    lines += ['## CPU搜索与DFS：两个域分别解释','',
              '固定本轮驻留、种子413/414，24策略=分流开关×mean或mean+p90×margin(0/0.0005/0.002)×cooldown(0/30秒)。这是参数/路由域，不是七模型部署域；GPU只比较四角色。','',
              *table(['CPU域','DFS/优化调用','DFS/优化秒','可得结论'],[
                  [name,str(c['dfs_oracle_calls'])+'/'+str(c['optimized_oracle_calls']),str(c['dfs_seconds'])+'/'+str(c['optimized_seconds']),
                   '24/24 UNKNOWN，无非空最优' if c['unknown_candidate_count'] else '仅文本：唯一winner、全部ties与run字段一致']
                  for name,c in cpu['studies'].items()]),
              '48→28次节省20/48=41.67%，来自10个不分流候选的已证明margin/cooldown等价；严格上界剪枝0、缓存命中0。仅文本观察耗时比1.734457×，单次非独占进程DFS先运行，不是重复冷启动或GPU加速证据。','',
              '仅文本winner允许分流、mean+p90、margin0、cooldown0，代理均值U=-0.6763464326863999；seed413/414分别分流95/1055与89/1133。M为经验驻留常数、E缺启动且背景移除，不能与native物理U混排。完整混合CPU背景服务/共享功率缺证据，native使用预声明mean+p90/margin0/cooldown30默认，并非CPU自主找到完整负载最优参数。CPU v1源漂移保留；v2重跑为权威。','',
              '## 全池部署目标的边界','',
              '当前只比较固定混合驻留的有限路由角色，不证明七池最优，也不证明连续路由最优。旧静态七池每任务0.0748360842705697 rps、1800秒、六项等权、K8，不能代替本轮重文本五项目标。本阶段新静态入口仅只读检查，未执行新七池搜索。',
              '真实全池对照需冻结相同副本/全部支持路径/分母1–8路由域及predictor，并与显式dfs_bound_pruning=False的plain DFS比较；CLI DFS默认有界剪枝。须报告时间/状态上限、UNKNOWN、残余上界与ties；合成fixture不能作实际七池效率证据。即使精确完成也仅证有限predictor域，非全物理组合。',
              '历史handler重建与独立画像迁移改善覆盖，但背景缺上下文、零进展拟合、留seed降秩和稀少功率状态仍在；不能把host-pending与actual-handler模型拼成受控反事实环境，亦未补零/epsilon伪造闭合。','',
              '## 来源与交付','',
              '逐轮CSV包含真实路由、稳定性、请求量、质量与平均时延；配对CSV包含各项成本差与效用贡献。mechanism-examples.json保存完整候选预测及门控。manifest.json记录直接读取的analysis/audit/packet/raw/CPU/审计准备文件及SHA、报告源与输出SHA；旧文件不覆盖。本报告不重新执行因果重放，沿用对应不可变raw收据的独立外审结论。前置1187项通过是历史准备记录，不表示本次报告重跑了这些测试；首版审计器假失败/裁决见manifest附属来源。','']
    return '\n'.join(lines)


def build(args):
    output=Path(args.output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite report directory: '+str(output))
    if bool(args.confirmation_analysis) != bool(args.confirmation_audit):
        raise ValueError('Confirmation analysis and audit must be supplied together')
    inputs=Inputs()
    dev=phase_data(inputs,args.development_analysis,args.development_audit,'development')
    confirm=phase_data(inputs,args.confirmation_analysis,args.confirmation_audit,'independent_confirmation') if args.confirmation_analysis else None
    cpu=inputs.read(args.cpu_summary)
    if cpu.get('weights') != WEIGHTS or cpu.get('objective_schema') != OBJECTIVE:
        raise ValueError('CPU objective mismatch')
    for path in args.evidence:
        inputs.read(path)
    dev['rankings']=role_rankings(dev['runs'],dev['packet']['seeds'])
    phases=[dev]+([confirm] if confirm else [])
    pending = not dev['terminal'] or not dev['audit_terminal'] or (dev['endpoint'] and (not confirm or not confirm['terminal'] or not confirm['audit_terminal']))
    status = 'PENDING' if pending else ('TERMINAL' if dev['valid'] and (not confirm or confirm['valid']) else 'TERMINAL_WITH_INVALID_EVIDENCE')
    runs=[r for p in phases for r in p['runs']]
    csv_runs=[]
    for r in runs:
        flat={k:v for k,v in r.items() if k not in {'config','plan','per_task','mechanism_examples'}}
        for task,stats in r.get('per_task',{}).items():
            flat.update({task+'_'+k:v for k,v in stats.items()})
        csv_runs.append(flat)
    pairs=[p for phase in phases for p in phase['pairs']]
    examples=[e for r in runs for e in r.get('mechanism_examples',[])]
    summary=dict(schema='gsdeploy.five_term_report_summary.v1',status=status,objective_schema=OBJECTIVE,weights=WEIGHTS,
                 development=dict(terminal=dev['terminal'],valid=dev['valid'],gate_passed=dev['endpoint'],terminal_jobs=dev['terminal_jobs'],audited_jobs=dev['audited_jobs']),
                 rankings=dev['rankings'],
                 confirmation=dict(terminal=confirm['terminal'],valid=confirm['valid'],endpoint=confirm['endpoint']) if confirm else None,
                 physical_seven_pool_optimality=False,GPU_started=False,frozen_sources_modified=False)
    payloads={'report.zh.md':render(dev,confirm,cpu,status),'runs.csv':csv_text(csv_runs),'paired-contributions.csv':csv_text(pairs),'rankings.csv':csv_text(dev['rankings']),
              'mechanism-examples.json':json.dumps(examples,ensure_ascii=False,indent=2,allow_nan=False)+'\n',
              'summary.json':json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n'}
    manifest=dict(schema='gsdeploy.five_term_report_manifest.v1',status=status,report_source=receipt(__file__),
                  inputs=list(inputs.refs.values()), frozen_packet_binding_status=[p['audit'].get('source_input_bindings') for p in phases],
                  output_receipts=[dict(path=str((output/name).resolve()),sha256=hashlib.sha256(content.encode()).hexdigest()) for name,content in payloads.items()],
                  command_arguments=vars(args),GPU_started=False,frozen_sources_modified=False,
                  scope='Native fixed-resident finite comparisons; CPU full workload UNKNOWN; text-only DFS equality only')
    payloads['manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    output.mkdir(parents=True,exist_ok=False)
    for name,content in payloads.items():
        (output/name).write_text(content,encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-analysis',required=True)
    parser.add_argument('--development-audit',required=True)
    parser.add_argument('--confirmation-analysis')
    parser.add_argument('--confirmation-audit')
    parser.add_argument('--cpu-summary',default=str(BASE/'cpu-search-v2/summary.json'))
    parser.add_argument('--evidence',action='append',default=[],help='Additional JSON evidence to bind in manifest')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(build(args),ensure_ascii=False))


if __name__=='__main__':
    main()
