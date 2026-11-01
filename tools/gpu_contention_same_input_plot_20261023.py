"""Plot same-input cases from saved interval partitions, no native raw reread."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.gpu_contention_forensics_20261023 import BASE,receipt


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    out=BASE/'gpu-contention-forensics-v1'
    if (out/'same-input-handler-timelines.png').exists():raise SystemExit('Refusing overwrite')
    analysis_path=out/'analysis.json';analysis=json.loads(analysis_path.read_text())
    prior_path=out/'seed402-good-measurement-1039-timeline.json';prior=json.loads(prior_path.read_text())
    event=prior['events'][0];origin=event['wall_ns']-round(event['measurement_offset_seconds']*1e9)
    spills={r['request_id']:r for run in analysis['results'] if run['seed']==402 for r in run['spill_requests']}
    ids=['measurement:613','measurement:483'];rows=[spills[i] for i in ids]
    fig,axes=plt.subplots(2,1,figsize=(13,6.8),constrained_layout=True)
    lanes={'generalist_2#0':3,'text#0':2,'text#1':1,'ocr#0':0}
    reconstructions=[]
    for ax,r in zip(axes,rows):
        gs,ge=r['start_ns'],r['end_ns'];union={}
        for seg in r['segments']:
            for alias,rid in seg['all_other_active'].items():
                key=(alias,rid)
                if key not in union:union[key]=[seg['start_ns'],seg['end_ns']]
                else:
                    assert union[key][1]==seg['start_ns']
                    union[key][1]=seg['end_ns']
        pad=(ge-gs)*.08
        low,high=gs-pad,ge+pad
        ax.broken_barh([((gs-origin)/1e9,(ge-gs)/1e9)],(2.72,.56),facecolors='#d87539',edgecolors='white',linewidth=.6)
        starts=[]
        for (alias,rid),(left,right) in union.items():
            color='#467caa' if alias.startswith('text#') else '#769e55'
            ax.broken_barh([((left-origin)/1e9,(right-left)/1e9)],(lanes[alias]-.28,.56),facecolors=color,edgecolors='white',linewidth=.6)
            if alias.startswith('text#') and left>gs:
                ax.plot((left-origin)/1e9,lanes[alias]+.34,marker='v',color='#242424',markersize=4)
                starts.append(dict(alias=alias,request_id=rid,inferred_service_start_ns=left))
        assert len(starts)==r['later_specialist_service_start_count']
        for t in (gs,ge):ax.axvline((t-origin)/1e9,color='#b85d29',linestyle='--',linewidth=.8)
        ax.set_xlim((low-origin)/1e9,(high-origin)/1e9)
        ax.set_yticks(list(lanes.values()),list(lanes));ax.set_ylim(-.6,3.65)
        ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
        ax.set_xlabel('Seconds since run start (Unix wall timestamps; bars clipped to G2 service window)')
        frac=sum(s['duration_ms'] for s in r['segments'] if s['specialist_active_count']==2)/r['on_service_ms']
        ax.set_title(f"Seed402 {r['request_id']}; same G2 input4007/output5 tokens; service{r['on_service_ms']:.1f}ms; two-specialist overlap{frac*100:.1f}%; later specialist starts={len(starts)}",fontsize=10,loc='left')
        reconstructions.append(dict(request_id=r['request_id'],G2_service_start_ns=gs,G2_service_end_ns=ge,
                                    clipped_other_service_intervals=[dict(alias=k[0],request_id=k[1],start_ns=v[0],end_ns=v[1]) for k,v in union.items()],
                                    later_specialist_starts=starts))
    fig.suptitle('Same-input host handler timelines — CUDA kernel overlap is UNKNOWN',fontsize=13)
    axes[0].legend(handles=[Patch(facecolor='#d87539',label='G2 text spill service'),Patch(facecolor='#467caa',label='Specialist service intersected with G2 window'),Line2D([0],[0],marker='v',color='none',markerfacecolor='#242424',label='Later specialist handler start')],loc='upper right',fontsize=8,ncol=2)
    fig.savefig(out/'same-input-handler-timelines.png',dpi=180);fig.savefig(out/'same-input-handler-timelines.pdf');plt.close(fig)
    (out/'same-input-handler-timelines.json').write_text(json.dumps(dict(source=receipt(__file__),sources=[receipt(analysis_path),receipt(prior_path)],
                                                                        measurement_start_unix_ns=origin,rows=reconstructions,
                                                                        scope='Other-worker bars reconstructed from exact saved partition intersections and CLIPPED to G2 service window; full G2 durations preserved. No raw reread. Zero additional starts in613,seven in483. No kernel claim.'),indent=2)+'\n')
    print('same-input plot complete')


if __name__=='__main__':main()
