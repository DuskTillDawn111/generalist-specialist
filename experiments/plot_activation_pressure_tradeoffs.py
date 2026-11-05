"""Plot paired finite-pressure measurements; no formal utility or ranking."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--five-model', type=Path, required=True)
    parser.add_argument('--replicas', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    five = json.loads(args.five_model.read_text())
    replicas = json.loads(args.replicas.read_text())
    controls, rows = {}, []
    def add(run, quality, label, batch):
        baseline = controls[run['seed'],batch['shared_trace_sha256']]
        old, old_quality = baseline
        row = dict(seed=run['seed'], asr_rate=batch['rates']['asr'], method=label,
            source=run['source'], trace=batch['shared_trace_sha256'],
            energy_ratio=batch['arrival_through_drain_joules']/old['arrival_through_drain_joules'],
            delta_Q_norm=quality['observed_probe_Q_norm']-old_quality['observed_probe_Q_norm'])
        row.update({task+'_e2e_ratio':batch['task_metrics'][task]['e2e_ms']['mean']
                    /old['task_metrics'][task]['e2e_ms']['mean'] for task in ('text','asr','image','ocr')})
        rows.append(row)
    for record in five['runs']:
        control = record['matched_control']
        quality = {batch['id']:batch for batch in record['matched_control_quality']['batches']}
        for batch in control['batches']:
            controls[control['seed'],batch['shared_trace_sha256']] = (batch,quality[batch['id']])
            add(control,quality[batch['id']],'4S',batch)
        run = record['measurements']
        quality = {batch['id']:batch for batch in record['actual_output_quality']['batches']}
        for batch in run['batches']:
            label=f"G+4S {batch['text_generalist_share']:g}/{batch['asr_generalist_share']:g}"
            add(run,quality[batch['id']],label,batch)
    for record in replicas['runs']:
        run = record['measurements']
        label = '4S + ASR' if run['deployment']['asr']==2 else '4S + text'
        quality = {batch['id']:batch for batch in record['actual_output_quality']['batches']}
        for batch in run['batches']:
            add(run,quality[batch['id']],label,batch)
    output=args.output
    output.mkdir(parents=True,exist_ok=False)
    methods=['4S','4S + ASR','4S + text','G+4S 0/0','G+4S 0.1/0',
             'G+4S 0/0.1','G+4S 0.1/0.1','G+4S 0.25/0.25']
    seeds=sorted({row['seed'] for row in rows})
    if len(seeds)!=2:
        raise ValueError('Exactly two paired cohorts expected')
    with (output/'paired-components.csv').open('x') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    panels=[('text_e2e_ratio','Text mean E2E / control'),('asr_e2e_ratio','ASR mean E2E / control'),
            ('image_e2e_ratio','Image mean E2E / control'),('ocr_e2e_ratio','OCR mean E2E / control'),
            ('energy_ratio','Arrival-through-drain energy / control'),('delta_Q_norm','Actual normalized quality loss change')]
    for rate in sorted({row['asr_rate'] for row in rows}):
        selected=[row for row in rows if row['asr_rate']==rate]
        if len(selected)!=len(methods)*len(seeds):
            raise ValueError('Complete method-by-round component roster required')
        fig,axes=plt.subplots(2,3,figsize=(17,8),layout='constrained')
        for axis,(key,title) in zip(axes.flat,panels):
            for index,seed in enumerate(seeds):
                series=[next(row for row in selected if row['method']==method and row['seed']==seed) for method in methods]
                axis.scatter([i+(-.09 if index==0 else .09) for i in range(len(methods))],
                    [row[key] for row in series],label=f'Round {index+1}',marker='o' if index==0 else 's',s=30)
            axis.axhline(0 if key=='delta_Q_norm' else 1,color='gray',linestyle='--',linewidth=.8)
            axis.set_title(title,fontsize=11)
            axis.set_xticks(range(len(methods)),methods,rotation=35,ha='right',fontsize=8)
            axis.grid(axis='y',alpha=.2)
        axes.flat[0].legend(fontsize=9)
        fig.suptitle(f'45 s exploratory probes: text 1.0 rps, ASR {rate:g} rps\n'
                     'G+4S labels give text/ASR offload shares; lower values are better',fontsize=13)
        fig.supxlabel('Paired offered cohorts; historical control chronology differs. Startup excluded. No formal utility or stability qualification.',fontsize=9)
        for extension in ('png','pdf'):
            fig.savefig(output/f'asr-{rate:g}-components.{extension}',dpi=160)
        plt.close(fig)
    manifest=dict(schema='gsdeploy.pressure_component_plot_receipts.v1',rows=len(rows),
        inputs={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (args.five_model,args.replicas,Path(__file__))},
        utility_qualified=False,scope='component ratios/differences only; no extrapolation to complete episodes')
    with (output/'manifest.json').open('x') as stream:
        json.dump(manifest,stream,indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(output),rows=len(rows))))


if __name__=='__main__':
    main()
