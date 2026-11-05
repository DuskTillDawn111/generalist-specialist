"""Paired runtime-scope measurements; neither fitting nor predictor admission."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.fit_activation_native import load_bundle
from experiments.diagnose_activation_semantic_repeats import keyed_rows
from experiments.analyze_activation_semantic import output_identity, summarize
from experiments.activation_semantic_bands import band_layout
from gsdeploy.forecasting.native_training import validate_capture, stage_layout


def analyze(base):
    names=['strict-round1','scoped-round1','scoped-round2','strict-round2']
    bundles={n:load_bundle(base/n) for n in names}; keyed={};issues=[]
    for name,b in bundles.items():
        validate_capture(b,allowed_splits=('development_runtime_scope_diagnostic',))
        raw=b['raw'];scope=b['plan']['runtime_scope']
        if any(s.get('foreign_pids') for s in raw['board_samples']):
            raise ValueError('Foreign GPU processes in diagnostic')
        for alias,family in raw['resident_aliases'].items():
            e=raw['ready'][alias].get('runtime_scope',{});strict=scope=='all_strict' or family=='generalist_2'
            if (e.get('scope')!=scope or e.get('slot')!=family or e.get('strict_requested') is not strict
                    or e.get('deterministic_algorithms') is not strict
                    or e.get('cublas_workspace_config')!=(':4096:8' if strict else None)):
                raise ValueError('Actual runtime settings do not match declared scope')
            if strict and (e.get('deterministic_warn_only') is not False or e.get('cudnn_benchmark') is not False
                           or e.get('cudnn_deterministic') is not True):
                raise ValueError('Incomplete strict runtime receipt')
        for row in raw['priming_request_records']+raw['request_records']:
            if row['worker_response'].get('success') is not True:raise ValueError('Failed output')
            family=raw['resident_aliases'][row['model']]
            if family=='generalist_2':
                e=row['worker_response'].get('memory_optimizations',{}).get('exllama_reconstruct_runtime',{})
                if e.get('matmul_recons_thd')!=1 or not e.get('configured_after_model_load'):
                    raise ValueError('G2 reconstruction path not verified')
        for row in raw['request_records']:
            stage_layout(row);band_layout(row,raw['resident_aliases'][row['model']])
        keyed[name]=keyed_rows(b)
    if any(set(v)!=set(keyed[names[0]]) for v in keyed.values()):raise ValueError('Matched cohorts differ')
    comparisons=[]
    pairs=[('strict-round1','scoped-round1'),('strict-round2','scoped-round2'),
           ('strict-round1','strict-round2'),('scoped-round1','scoped-round2')]
    for left,right in pairs:
        cross=left.startswith('strict') and right.startswith('scoped')
        if cross and bundles[left]['plan']['matched_job_content_sha256']!=bundles[right]['plan']['matched_job_content_sha256']:
            raise ValueError('Within-round input order mismatch')
        groups=defaultdict(lambda:dict(ratios=[],left=[],right=[],bands=defaultdict(list)))
        changes=[]
        for key,old in keyed[left].items():
            new=keyed[right][key];family=bundles[left]['raw']['resident_aliases'][old['model']]
            kind='solo' if key[0].startswith('solo:') else 'corun';path=kind+':'+family+'/'+old['task']
            a=old['worker_response']['worker_latency_ms'];b=new['worker_response']['worker_latency_ms']
            g=groups[path];g['ratios'].append(b/a);g['left'].append(a);g['right'].append(b)
            if output_identity(old['worker_response'])!=output_identity(new['worker_response']):
                changes.append(dict(key=list(key),family=family,task=old['task']))
            bands=[]
            for row in (old,new):
                bands.append({s['id']:(s['end_ns']-s['start_ns'])/1e6 for s in band_layout(row,family) if s['kind']=='gpu'})
            if set(bands[0])!=set(bands[1]):issues.append(right+':'+str(key)+':BAND_LAYOUT_CHANGED')
            else:
                for band in bands[0]:g['bands'][band].append(bands[1][band]/bands[0][band])
        if changes:issues.append(left+' vs '+right+':OUTPUT_CHANGED')
        comparisons.append(dict(left=left,right=right,cross_scope=cross,output_changes=changes,
            paths={k:dict(n=len(g['ratios']),right_over_left=summarize(g['ratios']),
                left_mean_ms=statistics.mean(g['left']),right_mean_ms=statistics.mean(g['right']),
                band_right_over_left={b:summarize(v) for b,v in g['bands'].items()}) for k,g in sorted(groups.items())}))
    return dict(schema='gsdeploy.runtime_scope_diagnostic.v1',sources={n:b['source'] for n,b in bundles.items()},
        runtime_receipts={n:b['raw']['ready'] for n,b in bundles.items()},comparisons=comparisons,issues=issues,
        output_repeat_diagnostic_passed=not issues,physical_predictor_qualified=False,formal_utility_qualified=False,
        limitations=['paired ABBA runtime bundle contrast; individual deterministic flags not causally separated',
            'two development rounds and selected fixed inputs do not establish full-pool or held-out accuracy',
            'no observer/control overhead comparison; any adopted new context needs new qualification',
            'GPU timing and CUDA events remain mixed native work, not measured kernel-only work'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();r=analyze(a.base.resolve())
    r['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with a.output.open('x') as f:json.dump(r,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(dict(output=str(a.output),issues=r['issues'],physical_predictor_qualified=False)))
