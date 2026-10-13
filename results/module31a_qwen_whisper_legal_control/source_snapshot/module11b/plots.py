"""Standalone SVG panels keep service demand, GPU memory, and RSS separate."""
import csv,html
from .common import *
def main():
    with (OUT/'specialist_low_qps_tradeoff.csv').open() as f:rows=list(csv.DictReader(f))
    groups={k:[r for r in rows if r['task']==k] for k in sorted({r['task'] for r in rows})};colors=['#2563eb','#d97706','#059669','#dc2626','#9333ea']
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="420" viewBox="0 0 1100 420"><rect width="1100" height="420" fill="white"/>']
    panels=[('service_demand_saving_seconds_per_second','Service-demand saving (s/s)',1),('measured_S_GPU_mib','Resident Specialist GPU memory (MiB)',1),('measured_S_host_RSS_bytes','Resident Specialist host RSS (GiB)',1/2**30)]
    for pi,(field,title,scale) in enumerate(panels):
        offset=30+pi*365;values=[float(r[field])*scale for r in rows if r[field]];maximum=max(values,default=1)
        parts.append(f'<text x="{offset}" y="25" font-size="13" font-family="sans-serif">{html.escape(title)}</text>')
        parts.append(f'<path d="M{offset+40} 45 V315 H{offset+325}" fill="none" stroke="#555"/>')
        for tick in range(4):
            yy=315-tick*90
            parts.append(f'<text x="{offset+35}" y="{yy+4}" text-anchor="end" font-size="10">{maximum*tick/3:.3g}</text>')
        for color,(task,rr) in zip(colors,groups.items()):
            pts=[]
            for r in rr:
                lam=float(r['lambda_rps'])
                if lam<=0 or not r[field]:continue
                xx=offset+40+(math.log10(lam)-(-7))/10*285;yy=315-float(r[field])*scale/maximum*270
                pts.append(f'{xx:.2f},{yy:.2f}')
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2"/>')
        for exponent in [-6,-3,0,3]:
            xx=offset+40+(exponent+7)/10*285
            parts.append(f'<text x="{xx}" y="335" text-anchor="middle" font-size="10">10^{exponent}</text>')
        parts.append(f'<text x="{offset+150}" y="355" font-size="12">Arrival rate (requests/s; log scale)</text>')
    for i,(color,task) in enumerate(zip(colors,groups)):
        parts.append(f'<text x="{80+i*180}" y="387" font-family="sans-serif" font-size="13" fill="{color}">{html.escape(task)}</text>')
    parts.append('<text x="35" y="410" font-family="sans-serif" font-size="11">Separate resource axes; no weighted score or equivalent memory. Cost observations use frozen resident ladders.</text></svg>')
    (OUT/'low_qps_separate_resource_curves.svg').write_text(''.join(parts))
if __name__=='__main__':main()
