"""Audit existing per-input output/stopping variation; never use future tokens as features."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.fit_activation_native import load_bundle
from experiments.analyze_activation_semantic import output_identity
from gsdeploy.forecasting.native_training import stage_layout


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();groups=defaultdict(list);sources=[]
    for directory in args.capture:
        bundle=load_bundle(directory.resolve());raw=bundle['raw'];sources.append(bundle['source'])
        if raw['status']!='completed' or any(raw.get(k) for k in ('errors','runtime_errors','telemetry_errors')):
            raise ValueError('Completed clean original output receipts required')
        for row in raw['request_records']:
            key=(raw['execution_context_sha256'],raw['resident_aliases'][row['model']],row['task'],row['pool_id'])
            w=row['worker_response'];layout=stage_layout(row)
            groups[key].append(dict(run_id=raw['run_id'],request_id=row['request_id'],job=row['profile_job_id'],
                output_sha256=output_identity(w),prediction=w.get('prediction'),generated_tokens=w.get('generated_tokens'),
                finish_reason=w.get('finish_reason'),success=w.get('success'),
                handler_ms=(w['end_perf_ns']-w['start_perf_ns'])/1e6,
                native_ms=sum(s['end_ns']-s['start_ns'] for s in layout if s['kind']=='gpu')/1e6))
    varied=[];summary=defaultdict(lambda:dict(inputs=0,repeated_inputs=0,inputs_with_output_variation=0,
                                           inputs_with_generated_length_variation=0))
    for (context,family,task,pool),rows in sorted(groups.items()):
        item=summary[context+':'+family+'/'+task];item['inputs']+=1
        if len(rows)<2:continue
        item['repeated_inputs']+=1;outputs=Counter(r['output_sha256'] for r in rows)
        lengths={r['generated_tokens'] for r in rows if r['generated_tokens'] is not None}
        item['inputs_with_output_variation']+=len(outputs)>1
        item['inputs_with_generated_length_variation']+=len(lengths)>1
        if len(outputs)>1 or len(lengths)>1:
            varied.append(dict(execution_context_sha256=context,family=family,task=task,pool_id=pool,
                               output_counts=dict(outputs),generated_lengths=sorted(lengths),observations=rows))
    report=dict(schema='gsdeploy.native_output_stopping_variation.v1',sources=sources,summary=dict(summary),varied_inputs=varied,
                diagnostic_only=True,physical_predictor_qualified=False,
                limitations=['mixed solo/co-run/cold histories; observed variation is not attributed causally to load or instrumentation',
                             'generated tokens and output text are post-completion diagnostics, forbidden as online causal predictor inputs',
                             'variation requires an empirical stopping/latency distribution or a separately validated aggregate decode stage; no fixed future token count assumed'])
    with args.output.open('x') as out:json.dump(report,out,ensure_ascii=False,indent=2);out.write('\n')
    print(json.dumps(dict(output=str(args.output),summary=dict(summary),varied_inputs=len(varied)),ensure_ascii=False))


if __name__=='__main__':main()
