"""Pair archived fig08 route rows with their exact final utility; offline only."""
from pathlib import Path
import hashlib
import json
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/fig08-matching-utility-20261105'


def main():
    records=[];sources=[]
    route_path=ROOT/'artifacts/research-figures-20261105/plot-data.json'
    route_data=json.loads(route_path.read_text())
    for seed in (11,29,47):
        for name,folder,stem in [('Legacy','bilevel-scalability-ablation-20261105','BO-MADS'),
                                 ('Balanced','bilevel-scheduler-improvement-20261105','balanced')]:
            path=ROOT/f'artifacts/{folder}/14-{stem}-{seed}.json'
            r=json.loads(path.read_text());w=r['winner']
            for row in route_data:
                if row['figure']=='fig08_routes' and row['seed']==seed and row['scheduler']==name:
                    assert abs(w['routing_shares'][row['task']].get(row['model'],0)-row['share'])<1e-12
            records.append(dict(seed=seed,scheduler=name,U=w['best_feasible_utility'],
                deployment=w['deployment'],routing_shares=w['routing_shares']))
            sources.append(path)
    OUT.mkdir(exist_ok=True)
    (OUT/'data.json').write_text(json.dumps(records,indent=2)+'\n')
    with (OUT/'utility.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','scheduler','U']);writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in records)
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(7.1,3.7));fig.subplots_adjust(left=.23,right=.94,bottom=.18,top=.85)
    for j in range(3):
        a,b=records[2*j:2*j+2]
        ax.plot([a['U'],b['U']],[2*j,2*j+1],color='#BBBBBB',lw=1,zorder=1)
    for i,r in enumerate(records):
        color='#0072B2' if r['scheduler']=='Legacy' else '#D55E00'
        ax.scatter(r['U'],i,color=color,marker='o' if r['scheduler']=='Legacy' else 's',s=38,zorder=3)
        ax.annotate(f"{r['U']:.6f}",(r['U'],i),xytext=(8,0),textcoords='offset points',va='center',fontsize=9)
    ax.set_yticks(range(6),[f"{r['seed']} / {r['scheduler']}" for r in records])
    ax.set_ylim(5.5,-.5);ax.set_xlim(-.3843,-.3780);ax.set_xticks([-.384,-.382,-.380,-.378])
    ax.set_xlabel('Final utility U (higher is better)');ax.set_title('14-model pool: utility corresponding to Fig. 8',loc='left',pad=14)
    ax.grid(axis='x',alpha=.2,lw=.6)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'fig08_final_utility.{ext}',dpi=600)
    plt.close(fig)
    receipt=lambda p:dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (OUT/'README.md').write_text('''# Fig. 8 路由的逐行效用对应图

六行顺序严格对应 research-figures-20261105/fig08_routes.png，已将所有绘图路由比例与六份原始搜索胜者逐项核对。Legacy 与 Balanced 都是 BO–MADS 的调度版本，不是 Joint-K2 与 BO–MADS 的比较。数字 11/29/47 是搜索随机种子。

这是 14 个候选模型、两任务、最多两副本的合成凸排队成本实验，不是实际 GPU 测量；U 越大越好。seed11/29 部署均为 s_shared+s_text，改善来自同部署的文本分流；seed47 从 g+x00 改为 s_shared+s_text，包含模型组合变化，不可全部归因于比例精化。

本图只整理历史实验。原批次上限为 64 次预测、32 次外层评价、128 次 MILP、60 秒；新版本实际预测53次，旧版本分别49/50/48次，不能由效用改善宣称工作量更少。

无新增搜索、预测或 GPU 实验。PNG 600 dpi，PDF/SVG 矢量。data.json 保存完整路由与部署，utility.csv 保存逐行效用。相同批次的收敛图为 ../research-figures-20261105/fig04_convergence.png（右侧14模型面板）。
''')
    (OUT/'manifest.json').write_text(json.dumps(dict(sources=[receipt(p) for p in sources+[route_path]],
        script=receipt(Path(__file__)),outputs=[receipt(p) for p in sorted(OUT.iterdir()) if p.name!='manifest.json'],
        routes_match_original_figure=True,new_searches=0,new_predictions=0,new_gpu_runs=0),indent=2)+'\n')
    print(OUT)


if __name__=='__main__':main()
