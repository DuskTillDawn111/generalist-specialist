import sys
from pathlib import Path
OUT=Path('/root/autodl-tmp/generalist-specialist/results/module32_heavy_specialist');sys.path[:0]=[str(OUT),str(OUT.parents[1])]
from m32.common import *
import numpy as np
rows=list(csv.DictReader((OUT/'native_latency_decomposition.csv').open()));pairs=read(OUT/'pair_comparisons_user_gate.json');gate=read(OUT/'module32_gate.json')['gate'];obs=read(OUT/'run_observations.json');by={o['run_id']:o for o in obs};inter=list(csv.DictReader((OUT/'worker_interference_analysis.csv').open())) if (OUT/'worker_interference_analysis.csv').exists() else []
lines=['# Stage D：Heavy Specialist 机制判断','',f'Gate：**{gate}**。仅解释本次匹配探索轨迹。',''];audit=[]
for aid,old in [('A1',1.472),('A2',4.816)]:
 row=next((r for r in rows if r['arm']==aid),None);p=next((p for p in pairs if p['arm']==aid),None)
 if not row or not p or not p['legal_matched_pair']:
  lines +=[f'{aid}：缺少合法稳定匹配对照，不计算ΔU或收益判断。',''];continue
 reduction=float(row['S_wait_reduction_mean_ms']);gain=float(row['S_wait_reduction_total_seconds']);gcost=float(row['G_execution_delta_vs_same_input_S_total_seconds']);extra=float(row['G_absolute_execution_total_seconds']);a=p.get('ASR_L_P_delta');b=p.get('background_L_P_delta');ratio=reduction/old
 infl=next((r for r in inter if r['arm'] in [aid,'split05' if aid=='A1' else 'split10'] and r['task']=='asr' and r['worker'].startswith(S+'#')),None)
 inflratio=float(infl['paired_service_inflation_ratio']) if infl and infl.get('paired_service_inflation_ratio') else None
 gpath=next((r for r in by[aid]['paths'] if r['task']=='asr' and r['model']==by[aid]['Generalist']),None)
 lines +=[f'## {aid}','',f'同输入保留S等待减少{reduction:.3f}ms/请求，总{gain:.3f}s。Module31A对应值{old:.3f}ms，因此观测减少量为原来的{ratio:.3f}倍。'+('本次量级更大，但不宣称跨模块独立统计显著性。' if reduction>old else '本次未增加对应的可释放排队收益。'),'']
 if gpath:lines +=[f"G原生ASR执行mean/P95={gpath['service']['mean']:.3f}/{gpath['service']['p95']:.3f}ms，等待mean/P95={gpath['wait']['mean']:.3f}/{gpath['wait']['p95']:.3f}ms；新增G绝对执行{extra:.3f}s，相对同输入原S执行额外{gcost:.3f}s。排队减少与额外执行的全请求加总见五项分解，P95不能作线性时间加总。",'']
 if a is not None and b is not None:lines +=[f'ASR mean/P95效用变化{a:+.6f}；五背景mean/P95变化{b:+.6f}；总ΔU={p["delta_U"]:+.6f}。'+('ASR层面有正贡献，背景损失抵消其中收益。' if a>0 and b<0 else 'ASR层面本身也未形成净正贡献，背景变化不能解释为唯一抵消因素。' if a<=0 else '背景效应与ASR贡献均按完整六任务保留。'),'']
 if inflratio is not None:lines +=[f'保留S的同输入服务比GS/S-only={inflratio:.6f}，变化{(inflratio-1)*100:+.3f}%。这是真实共置/运行条件下的配对服务变化，不能唯一归因为GPU因果。','']
 audit.append(dict(arm=aid,S_wait_reduction_mean_ms=reduction,module31A_reduction_mean_ms=old,observed_reduction_ratio_vs31A=ratio,S_wait_total_saved_seconds=gain,G_absolute_added_execution_seconds=extra,G_extra_execution_vs_S_seconds=gcost,ASR_L_P_delta=a,background_L_P_delta=b,delta_U=p['delta_U'],paired_S_service_inflation_ratio=inflratio,independent_significance=False))
lines +=['历史standalone重型S约494.913ms，Module32共置短容量段分别约530.901ms(C1)与1669.372ms(C2)，C1高负载后段又出现更大服务膨胀。模块间G、背景路径、输入顺序、负载和运行状态不同，因此仅据配对正式数据判断本负载点；不将一次结果推广为heavy S普遍有效或无效。','', '未修改正式优化器；只在探索正例Gate时由总结报告提出不含模型名称/QPS/比例硬编码的建议。没有独立长窗口复现。']
(OUT/'heavy_specialist_mechanism_analysis.md').write_text('\n'.join(lines)+'\n');save('heavy_specialist_mechanism_analysis.json',audit)
with (OUT/'module32_summary.md').open('a') as f:f.write('\n详细Stage D四项判断：heavy_specialist_mechanism_analysis.md / .json。\n')
print('STAGE_D_ANALYSIS',audit)
