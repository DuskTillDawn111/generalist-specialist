"""Report limitations and uncertainty without editing frozen execution code."""
from .common import *
import numpy as np
from scipy.stats import t as student_t


def main():
    gate=read(OUT/'module33_gate.json')
    dev=read(OUT/'development_observations.json')
    pairs=read(OUT/'pair_observations.json')
    chosen=read(OUT/'independent_pair_selection.json')['development_best']
    ranking=read(OUT/'GPU_ranking_analysis.json')
    attempts=[read(p) for p in (OUT/'gpu_observations').glob('*.json')]
    failures=[dict(run_id=o['run_id'],candidate_id=o['candidate_id'],U=o['U'],raw=o['raw'],checks=o['checks'],
                  actual_launch_manifest=ref(OUT/('frozen_gpu_manifest_v2.json' if o['run_id']=='DEV_C6494564b5e815e73' else 'frozen_gpu_manifest_v4.json'))) for o in attempts if not o['technical_valid']]
    save('GPU_execution_accounting.json',dict(execution_attempts=len(attempts),
         valid_measured_runs=sum(o['technical_valid'] for o in attempts),
         development_candidates=len(dev),independent_request_traces=len(pairs),
         initialization_failures_retained=failures,
         no_extra_GPU_run_for_journal_reassessment=True,
         syntax_failure_before_workers='pre_GPU_syntax_failure/',
         scope='Technical failures do not get legal U; development candidates each have one valid measured run. Initialization-only relaunch recorded separately.'))
    valid=[o for o in dev if o['eligible']]
    ordered=sorted(valid,key=lambda o:o['U'],reverse=True)
    rank_pairs=list(csv.DictReader((OUT/'gpu_ranking_pairs.csv').open()))
    top_ci=None
    if len(ordered)>=2:
        a,b=ordered[:2]
        row=next(r for r in rank_pairs if {r['A'],r['B']}=={a['candidate_id'],b['candidate_id']})
        lo,hi=float(row['CI95_low']),float(row['CI95_high'])
        if row['A']!=a['candidate_id']:lo,hi=-hi,-lo
        top_ci=dict(best=a['candidate_id'],runner_up=b['candidate_id'],delta_U=a['U']-b['U'],
                    CI95_low=lo,CI95_high=hi,strict_ranking_supported=lo>0,
                    scope='Unadjusted paired 30s block diagnostic on one development trajectory. Memory fixed; no between-run uncertainty.')
    uv=[o['U'] for o in pairs if o['eligible'] and o['candidate_id']==chosen]
    n=len(uv);mean=float(np.mean(uv)) if n else None
    half=float(student_t.ppf(.975,n-1)*np.std(uv,ddof=1)/np.sqrt(n)) if n>1 else None
    repeated=dict(candidate_id=chosen,n=n,observed_U=uv,mean_U=mean,
                  offered_completed=[dict(run_id=o['run_id'],offered=o['components']['offered'],completed=o['components']['completed'],within=o['components']['within'],T_norm=o['components']['T_norm']) for o in pairs if o['candidate_id']==chosen],
                  CI95_low=mean-half if half is not None else None,CI95_high=mean+half if half is not None else None,
                  scope='Three independent frozen request traces; descriptive t interval, same deployment. Does not validate other candidates independently.')
    speeds=list(csv.DictReader((OUT/'native_optimizer_speedup.csv').open()))
    by={alg:[float(r['speedup']) for r in speeds if r['algorithm']==alg and r['pool']=='LARGE7']
        for alg in ['FORMAL_V3_CPU_BACKEND','20A_NATIVE_CONTINUOUS','20A_FINITE_DOMAIN_ADAPTER']}
    route_audit=[]
    sp=read(OUT/'candidate_space_definition.json');K=sp['route_grid_denominator']
    for oracle in read(OUT/'cpu_oracles.json'):
        pool,load=oracle['pool'],oracle['load'];native=read(OUT/f'cpu/{pool}_{load}_formal_V3.json')
        rates=sp['loads'][load]['rates'];off=[];ids=[]
        finite_ids={identity(json.loads(r['deployment']),json.loads(r['routing'])) for r in csv.DictReader((OUT/f'cpu/{pool}_{load}_all_candidates.csv').open()) if r['eligible']=='True'}
        for call in native['backend_calls']:
            x={m:n for m,n in call['deployment'].items() if n};rr=call['routing'];ids.append(identity(x,rr))
            if any(abs(q/rates[t]*K-round(q/rates[t]*K))>1e-7 for t,row in rr.items() for q in row.values()):
                off.append(dict(deployment=x,routing=rr))
        final_decision=native['decision'];final_routes=final_decision['routing']
        final_on_grid=all(abs(q/rates[t]*K-round(q/rates[t]*K))<=1e-7 for t,row in final_routes.items() for q in row.values())
        matches_U=abs(final_decision['prediction']['U']-oracle['U_oracle']['predicted_U'])<1e-9
        recorded=next(r for r in read(OUT/'predicted_regret.json') if r['pool']==pool and r['load']==load and r['algorithm']=='FORMAL_V3_CPU_BACKEND')
        route_audit.append(dict(pool=pool,load=load,runtime_backend_calls=len(ids),
            unique_runtime_routes=len(set(ids)),off_grid_runtime_routes=off,
            explicit_unique_legal_grid_routes_visited=len(set(ids)&finite_ids),
            feasible_grid_routes=len(finite_ids),
            finite_access_ratio=len(set(ids)&finite_ids)/len(finite_ids),
            finite_access_ratio_scope='Explicit observed unique native runtime candidates falling in finite legal domain / legal grid candidates; not continuous-domain coverage or MILP internal branch count.',
            all_prediction_scenario_calls=native['objective_calls'],
            J_route_cache_entries=native['J_route_solves'],
            original_formal_counters={k:native['result'][k] for k in ['lp_solve_count','milp_solve_count','runtime_evaluation_count','search_iterations','num_candidates_generated','num_candidates_lp_evaluated','num_runtime_candidates','visited_deployment_count']},
            formal_native_enforces_finite_route_domain=False,
            final_plan_in_finite_domain=final_on_grid and recorded['finite_scope'],final_U_equals_finite_oracle=matches_U,
            grid_adapter20A_enforces_finite_route_domain=True,
            scope='Final-output finite-grid comparison is valid; native V3 internally searches continuous routes and is not an identical-grid search experiment.'))
    save('formal_route_domain_audit.json',route_audit)
    csvsave('formal_search_access_and_calls.csv',route_audit)
    contrasts=[]
    co=next(o for o in dev if o['candidate_id']==chosen)
    for r in csv.DictReader((OUT/'gpu_measured_baseline_comparison.csv').open()):
        r=dict(r)
        r['selected_minus_baseline_U']=co['U']-float(r['measured_development_U']) if r['measured_development_U'] else None
        contrasts.append(r)
    save('gpu_baseline_regret_comparison.json',contrasts)
    evidence=dict(development_point_ranking=ranking,top_two_uncertainty=top_ci,
                  independent_chosen_plan_utility=repeated,
                  native_first_run_LARGE7_speedups=by,
                  formal_V3_full_GPU_adaptive_search_evaluated=False,
                  formal_V3_same_finite_internal_search_domain=False,
                  formal_route_domain_audit=route_audit,
                  formal_V3_decision_scope='Original create/search/acceptance with explicit unchanged20A CPU forecast RuntimeEvaluator; selected plan then measured on GPU. Not a full real-GPU-feedback search timing experiment.',
                  original20A_holdout_support_rule_satisfied=False,
                  original20A_holdout_reason='This benchmark has one load/model-pool combination and zero independent competing-deployment contrasts (best and optimizer are identical). Does not satisfy >=3 matched groups across >=2 model combinations.',
                  optimizer_superiority_over_fastest_static_demonstrated=False,
                  benchmark_restriction='SMALL6 coverage forces5 resident families; selection chooses the optional largeASR specialist, OCR1/2 counts and ASR routing grid. LARGE7 adds alternative Generalist and measured replica choices. These are explicit restricted spaces, not the original full pool.',
                  postmeasurement_algorithm_changes=False,
                  performance_error_causal_decomposition_identified=False,
                  performance_error_reason='Input distribution, Poisson vs periodic arrival, native frontend, co-location and hardware state jointly differ. Absolute U residual is measured; their individual causal shares are not identified.',
                  journal_assessment_correction=read(OUT/'journal_assessment_correction.json'))
    save('final_scientific_evidence.json',evidence)
    gate.update(top_two_uncertainty=top_ci,independent_chosen_plan_utility=repeated,
                native_first_run_LARGE7_speedups=by,
                GPU_any_point_ranking_mismatch=ranking.get('point_reversals',0)>0,
                GPU_RANKING_MISMATCH_definition='Optimizer selected plan differs from full finite development point maximum; all pairwise inversions separately recorded.',
                OPTIMIZER_SEARCH_GAP_scope='20A finite adapter has exact same-grid U regret0; native optimizer final admitted outputs match grid reference. No continuous-space search optimum certificate.',
                original20A_holdout_support_rule_satisfied=False,
                independent_competing_deployment_contrasts=0,
                formal_V3_full_GPU_adaptive_search_evaluated=False,
                FORMAL_V3_SAME_FINITE_SEARCH_DOMAIN=False,
                strict_all_requested_conditions_met=False,
                strict_unmet_condition='Stage C formal native V3 did not enforce the identical finite route grid internally; only its output lies on the grid. Algorithm not amended after GPU outcomes.',
                formal_route_domain_audit=route_audit,
                MIXED_OPTIMALITY_SUPPORTED_definition='Scoped finite-grid predicted U optimum and all8 development GPU empirical maximum matched by the unchanged CPU-backed formal decision, with3 independent same-plan remeasurements. Not a strict population-ranking certificate.')
    save('module33_gate.json',gate)
    text=['','## 最终证据边界补充','',
          '正式V3保留原create/search/接纳逻辑，性能接口明确使用冻结20A CPU预测；随后独立测量输出部署。没有开展完整的GPU反馈式V3搜索，因此下面的速度只指CPU预测后端的求解开销，不能解释成完整在线GPU搜索加速。', '',
          'Stage C范围限制：原生正式V3路由器仍是连续变量，最终结果都落入冻结网格，但搜索中存在网格外路由（formal_route_domain_audit.json逐项列出）。它与网格Oracle的最终输出regret比较有效，却不是严格相同有限路由域的搜索比较；只有20A有限域适配这一开发算法执行了相同路由网格。正式V3的J和噪声规则未改，不能用20A适配结果冒称正式V3的同域验收。本轮保留该未满足条件，不后验修改正式路由器重新验收。', '',
          f"在GPU启动前记录的首次CPU计时，大池正式V3耗时比{min(by['FORMAL_V3_CPU_BACKEND']):.2f}–{max(by['FORMAL_V3_CPU_BACKEND']):.2f}×，原生连续20A耗时比{min(by['20A_NATIVE_CONTINUOUS']):.2f}–{max(by['20A_NATIVE_CONTINUOUS']):.2f}×，20A网格适配同域加速{min(by['20A_FINITE_DOMAIN_ADAPTER']):.2f}–{max(by['20A_FINITE_DOMAIN_ADAPTER']):.2f}×。原始耗时见native_optimizer_speedup.csv。正式V3/原生连续20A与网格枚举不是相同内部路由域，耗时比仅供描述。上表V3三次中位数包含与首次GPU启动/warmup重叠的后两次计时，保留但不能视为严格隔离的CPU效率实验；枚举每格只计时一次，所有速度均为单机描述性结果。两池不仅模型数量从6增至7，副本上限也扩大；不能把开销变化唯一归因于增加一个模型，更不声称渐近复杂度。", '',
          f"最高两项不确定性：{json.dumps(top_ci,ensure_ascii=False)}。最高点差约0.0021，低于原20A正收益门槛0.01，且区间跨0；不能据此宣称OCR一副本严格优于二副本，也不能宣称优化器严格优于J-routing基线。", '',
          f"获选部署独立轨迹复测的实际U：{json.dumps(repeated,ensure_ascii=False)}。Oracle与算法为同一配置时，经验regret=0是身份关系；三条独立轨迹复测提供该部署的效用与波动数据，不构成竞争候选的独立排名确认。", '',
          '独立轨迹均无请求丢失。第三条Poisson轨迹只有326个到达，少于240秒×1.46 QPS=350.4，原20A固定QPS分母导致T_norm=0.069635；保留此原定义，不能解释成系统丢失请求，也不后验改写分母。三次窗口/样本量不同于开发120秒窗口，开发与复测U差值不能直接归因于单一机制。', '',
          '本Module不满足20A冻结的通用holdout支持规则（至少3个独立匹配部署/负载组、跨至少2模型组合等）；20A历史UTILITY_MODEL_INSUFFICIENT未被改写或升级。CPU四格搜索U regret为0，性能预测U MAE已量化，但不识别各性能机制的单独因果贡献。', '',
          '基线比较：ASR32B的小池中，fastest-service静态启发式和最佳一副本不分流静态类也选择同一个观测最佳部署，差值为0；本轮不证明优化器优于这些简单基线。最小显存启发式实测U=-2.514951，获选方案U=-1.042465；J网格参考U=-1.044539，其与获选方案差距无法可靠区分。BEST_ONE_REPLICA_NO_SPLIT是受限静态类别内的参考最大，不能当作无需搜索的廉价启发式。所有相同配置复用该配置原始测量，没有额外伪造独立试验。', '',
          '空间限制：SMALL6的六任务覆盖强制5个模型族驻留；自由度主要是额外ASR Specialist是否驻留、OCR1/2副本及ASR离散路由。LARGE7增加替代Generalist和实测副本选择。不能把这个充分测量的小问题代表为原始完整模型选择问题。', '',
          '首个有效GPU记录曾因journal缺少phase末端才派生的3个字段被评估器误判。用原native annotate_records从相同时间戳在内存重建后，完整记录逐项相等；原raw字节、初始失败评估与修正说明均保留，未重跑GPU或修改U/门槛。原native harness的raw provenance引用初始v1清单；run_manifest_provenance.json提供实际v3/v4启动清单映射，原raw保持原样。', '',
          '统一三结构选择框架可以继续扩展，但原合法Specialist集合没有visual_qa路径，六任务纯S在当前质量准入下不可行；不能把仍驻留VQA Generalist的ASR S-only路由改称纯S。纯G须纳入已准入的OCR Generalist并重新冻结预算/容量。详见unified_structure_extension_readiness.json。本轮未启动Module34。','',
          '[实验图PNG](/root/autodl-tmp/generalist-specialist/results/module33_mixed_oracle/module33_benchmark.png) · [可导出PDF](/root/autodl-tmp/generalist-specialist/results/module33_mixed_oracle/module33_benchmark.pdf)','']
    for name in ['module33_summary.md','ranking_error_analysis.md']:
        p=OUT/name;body=p.read_text()
        body=body.replace('CPU U搜索误差为0。','CPU有限网格参考下的输出regret为0，原生连续空间搜索误差未获证明。')
        body=body.replace('U搜索regret=0。','输出相对U网格Oracle的regret=0；原生连续空间最优性未知。')
        body=body.replace('原生20A和正式V3的CPU预测后端结果均找到U最优','原生20A和正式V3的CPU预测后端输出均匹配有限网格U最优')
        body=body.replace('20A自己的U目标regret=0。','20A有限域适配自己的网格U目标regret=0；原生连续域全局最优性未知。')
        if name=='module33_summary.md':
            lines=body.splitlines();lines[4:4]=['证据限定：正式V3最终输出匹配有限网格Oracle，但内部仍使用连续路由；Stage C严格同有限搜索域条件未完全满足。完整8配置GPU观测最大与独立同方案复测成立，不是跨候选独立排名或通用全局最优证明。详情见文末与final_scientific_evidence.json。','']
            body='\n'.join(lines)+'\n'
        p.write_text(body+'\n'.join(text))
    print('FINAL_EVIDENCE',n,'independent traces',flush=True)


if __name__=='__main__':main()
