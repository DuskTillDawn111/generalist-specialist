"""Receipt-bound post-hoc workload/service decomposition, CPU read-only only."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv,hashlib,io,json,math,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/queue-five-term-study-20261023'
FEATURES=('input_tokens','generated_tokens','effective_generation','finish_reason','request_sha256','protocol_sha256','prediction')


def digest(data):return hashlib.sha256(data).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def mean(values):return statistics.fmean(values) if values else None

def describe(values):
    values=sorted(values)
    return dict(n=len(values),mean=mean(values),median=statistics.median(values) if values else None,min=values[0] if values else None,max=values[-1] if values else None)

class Inputs:
    def __init__(self):self.refs={}
    def read(self,path,expected=None):
        p=Path(path).resolve();before=p.stat();data=p.read_bytes();after=p.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Changed while reading: '+str(p))
        ref=dict(path=str(p),sha256=digest(data))
        if expected and expected!=ref:raise ValueError('Receipt mismatch: '+str(p))
        self.refs[str(p)]=ref
        return json.loads(data)


def read_native_stream(inputs,ref):
    """Read pretty-printed terminal request records one object at a time."""
    path=Path(ref['path']).resolve();before=path.stat();h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
    actual=dict(path=str(path),sha256=h.hexdigest())
    if actual!=ref:raise ValueError('Native receipt mismatch: '+str(path))
    inputs.refs[str(path)]=actual
    families=None;rows=[];section=None;buf=[]
    with path.open() as stream:
        for line in stream:
            if section=='families':
                buf.append(line)
                if line.startswith('  },'):
                    families=json.loads(''.join(buf).rstrip().rstrip(','));buf=[];section=None
                continue
            if section=='requests':
                if line.startswith('  ],'):break
                if line.startswith('    {'):buf=[line]
                elif buf:buf.append(line)
                if buf and (line.startswith('    },') or line.rstrip()=='    }'):
                    r=json.loads(''.join(buf).rstrip().rstrip(','))
                    rows.append({**{k:r[k] for k in ('request_id','pool_id','model','task','sample_id','scheduled_offset_seconds')},
                                 'worker_response':r['worker_response']})
                    buf=[]
                continue
            if line.startswith('  "replica_to_family": {'):
                section='families';buf=['{\n']
            elif line.startswith('  "request_records": ['):
                section='requests'
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Native changed during stream')
    if families is None or not rows:raise ValueError('Unrecognized terminal JSON formatting')
    return dict(replica_to_family=families,request_records=rows)


def compact(row):
    w=row['worker_response']
    return dict(request_id=row['request_id'],pool_id=row['pool_id'],alias=row['model'],
                input_tokens=w.get('input_tokens'),generated_tokens=w.get('generated_tokens'),
                effective_generation=w.get('effective_generation'),finish_reason=w.get('finish_reason'),
                service_ms=w.get('service_ms'),worker_latency_ms=w.get('worker_latency_ms'),
                worker_queue_ms=w.get('queue_wait_ms'),cache_reclamation=w.get('cache_reclamation'),
                raw_output_characters=len(w.get('raw_output') or ''),prediction_characters=len(w.get('prediction') or ''),
                **{k:w.get(k) for k in ('request_sha256','protocol_sha256')})


def same_work(a,b):
    wa,wb=a['worker_response'],b['worker_response']
    return {k:wa.get(k)==wb.get(k) for k in FEATURES}


def csv_dump(rows):
    keys=list(dict.fromkeys(k for r in rows for k in r));s=io.StringIO(newline='');w=csv.DictWriter(s,keys);w.writeheader()
    for row in rows:w.writerow({k:canonical(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    return s.getvalue()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',default=str(BASE/'independent-runtime-audit-stream-v1/audit-snapshot-09.json'))
    p.add_argument('--analysis',default=str(BASE/'native-development-execution-v1/analysis.json'))
    p.add_argument('--calibration-audit',default=str(ROOT/'artifacts/queue-coupled-study-20261023/controlled-calibration-independent-audit-v1/audit.json'))
    p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError('Refusing to overwrite '+str(out))
    inputs=Inputs();audit=inputs.read(args.audit);analysis=inputs.read(args.analysis)
    audited={r['label']:r for r in audit['observations']};observed={r['label']:r for r in analysis['observations']}
    raw={}
    for seed in (401,402):
        for role in ('coupled_spill','matched_no_spill'):
            label=f'{role}-seed{seed}';a=audited[label];ref=observed[label]['raw']
            if not a['verified'] or a['raw']!=ref:raise ValueError('Unaudited or mismatched '+label)
            raw[label]=read_native_stream(inputs,ref)
    ca=inputs.read(args.calibration_audit)
    if not ca['verified']:raise ValueError('Calibration audit failed')
    cal=inputs.read(ca['raw']['path'],ca['raw'])
    cal_by=defaultdict(list)
    for r in cal['request_records']:
        family=cal['replica_to_family'][r['model']]
        cal_by[(family,r['pool_id'],r['calibration_context'])].append(r)
    # S0 and S1 denote isolated active aliases under the same PRIMARY residency.
    solo={'generalist_2':('G',),'text':('S0','S1','S')}
    contexts=sorted({r['calibration_context'] for r in cal['request_records']})
    if not any(k[2] in solo['text'] for k in cal_by):
        raise ValueError('No recognized isolated specialist calibration context: '+str(contexts))
    spills=[];same_family=[];seed_stats=[];cal_refs=[]
    for (family,pool,context),rows in sorted(cal_by.items()):
        token_sets={k:sorted({canonical(r['worker_response'].get(k)) for r in rows}) for k in FEATURES}
        cal_refs.append(dict(family=family,pool_id=pool,context=context,requests=len(rows),
                             service_ms=describe([r['worker_response']['service_ms'] for r in rows]),
                             work_feature_values=token_sets,calibration_seeds=sorted({r['calibration_seed'] for r in rows})))
    for seed in (401,402):
        on=raw[f'coupled_spill-seed{seed}'];off=raw[f'matched_no_spill-seed{seed}']
        off_by={r['request_id']:r for r in off['request_records']}
        local_spills=[];local_same=[]
        for a in on['request_records']:
            b=off_by[a['request_id']]
            if any(a[k]!=b[k] for k in ('task','pool_id','sample_id','scheduled_offset_seconds')):raise ValueError('Cohort mismatch')
            fa=on['replica_to_family'][a['model']];fb=off['replica_to_family'][b['model']]
            if fa==fb:
                flags=same_work(a,b)
                row=dict(seed=seed,task=a['task'],family=fa,request_id=a['request_id'],pool_id=a['pool_id'],
                         alias_on=a['model'],alias_off=b['model'],work_equal=flags,
                         all_work_equal=all(flags.values()),service_on_ms=a['worker_response']['service_ms'],
                         service_off_ms=b['worker_response']['service_ms'],
                         service_ratio_on_over_off=a['worker_response']['service_ms']/b['worker_response']['service_ms'])
                same_family.append(row);local_same.append(row)
            if a['task']!='text' or fa!='generalist_2':continue
            caa,cbb=compact(a),compact(b)
            row=dict(seed=seed,request_id=a['request_id'],pool_id=a['pool_id'],
                     **{'G_'+k:v for k,v in caa.items() if k not in ('request_id','pool_id')},
                     **{'S_off_'+k:v for k,v in cbb.items() if k not in ('request_id','pool_id')})
            row['G_over_S_service_ratio']=caa['service_ms']/cbb['service_ms']
            row['G_minus_S_generated_tokens']=caa['generated_tokens']-cbb['generated_tokens']
            row['G_minus_S_input_tokens']=caa['input_tokens']-cbb['input_tokens']
            for family,prefix,runtime in [('generalist_2','G',a),('text','S',b)]:
                refs=[r for ctx in solo[family] for r in cal_by.get((family,a['pool_id'],ctx),[])]
                row[prefix+'_same_input_solo_calibration_n']=len(refs)
                row[prefix+'_same_input_solo_service_ms']=mean([r['worker_response']['service_ms'] for r in refs])
                matched=[r for r in refs if all(same_work(runtime,r).values())]
                row[prefix+'_same_work_solo_calibration_n']=len(matched)
                row[prefix+'_same_work_solo_service_ms']=mean([r['worker_response']['service_ms'] for r in matched])
                base=row[prefix+'_same_work_solo_service_ms']
                row[prefix+'_native_over_same_work_solo_service_ratio']=runtime['worker_response']['service_ms']/base if base else None
            gs,ss=row['G_same_work_solo_service_ms'],row['S_same_work_solo_service_ms']
            row['G_over_S_same_work_solo_service_ratio']=gs/ss if gs and ss else None
            # Descriptive additive service accounting, not causal attribution.
            row['cross_family_solo_reference_gap_ms']=gs-ss if gs and ss else None
            row['G_native_minus_G_solo_ms']=caa['service_ms']-gs if gs else None
            row['S_off_native_minus_S_solo_ms']=cbb['service_ms']-ss if ss else None
            row['accounting_identity_residual_ms']=(caa['service_ms']-cbb['service_ms']-
                ((gs-ss)+(caa['service_ms']-gs)-(cbb['service_ms']-ss))) if gs and ss else None
            spills.append(row);local_spills.append(row)
        spec=[r for r in local_same if r['family']=='text']
        seed_stats.append(dict(seed=seed,spills=len(local_spills),
            G_generated_tokens=describe([r['G_generated_tokens'] for r in local_spills]),
            S_off_generated_tokens=describe([r['S_off_generated_tokens'] for r in local_spills]),
            G_input_tokens=describe([r['G_input_tokens'] for r in local_spills]),
            S_off_input_tokens=describe([r['S_off_input_tokens'] for r in local_spills]),
            G_output_longer=sum(r['G_minus_S_generated_tokens']>0 for r in local_spills),
            G_output_equal=sum(r['G_minus_S_generated_tokens']==0 for r in local_spills),
            G_output_shorter=sum(r['G_minus_S_generated_tokens']<0 for r in local_spills),
            G_service_ms=describe([r['G_service_ms'] for r in local_spills]),
            S_off_service_ms=describe([r['S_off_service_ms'] for r in local_spills]),
            G_over_S_service_ratios=describe([r['G_over_S_service_ratio'] for r in local_spills]),
            same_family_text_requests=len(spec),same_family_text_all_work_equal=sum(r['all_work_equal'] for r in spec),
            same_family_text_feature_mismatch_counts={k:sum(not r['work_equal'][k] for r in spec) for k in FEATURES},
            same_family_text_service_ratios_all=describe([r['service_ratio_on_over_off'] for r in spec]),
            same_family_text_service_ratios_same_work=describe([r['service_ratio_on_over_off'] for r in spec if r['all_work_equal']]),
            G_native_over_same_work_solo_ratios=describe([r['G_native_over_same_work_solo_service_ratio'] for r in local_spills if r['G_native_over_same_work_solo_service_ratio'] is not None]),
            S_native_over_same_work_solo_ratios=describe([r['S_native_over_same_work_solo_service_ratio'] for r in local_spills if r['S_native_over_same_work_solo_service_ratio'] is not None])))
    # Within-input, within-family controlled context/solo comparisons; retain token mismatches.
    cal_comparisons=[]
    for r in cal['request_records']:
        family=cal['replica_to_family'][r['model']]
        refs=[v for ctx in solo.get(family,()) for v in cal_by.get((family,r['pool_id'],ctx),[])
              if v['calibration_seed']==r['calibration_seed'] and all(same_work(r,v).values())]
        baseline=mean([v['worker_response']['service_ms'] for v in refs])
        cal_comparisons.append(dict(request_id=r['request_id'],family=family,pool_id=r['pool_id'],context=r['calibration_context'],
                calibration_seed=r['calibration_seed'],input_tokens=r['worker_response'].get('input_tokens'),
                generated_tokens=r['worker_response'].get('generated_tokens'),finish_reason=r['worker_response'].get('finish_reason'),
                service_ms=r['worker_response']['service_ms'],same_work_same_seed_solo_n=len(refs),
                same_work_same_seed_solo_ms=baseline,ratio_to_same_work_same_seed_solo=r['worker_response']['service_ms']/baseline if baseline else None))
    grouped=[]
    for family,context in sorted({(r['family'],r['context']) for r in cal_comparisons}):
        rs=[r for r in cal_comparisons if r['family']==family and r['context']==context]
        ratios=[r['ratio_to_same_work_same_seed_solo'] for r in rs if r['ratio_to_same_work_same_seed_solo'] is not None]
        grouped.append(dict(family=family,context=context,requests=len(rs),exact_work_same_seed_matched=len(ratios),ratio=describe(ratios)))
    summary=dict(schema='gsdeploy.five_term_token_service_decomposition.v1',status='POST_HOC_DESCRIPTIVE_NOT_CAUSAL_GPU_ATTRIBUTION',
                 native_seeds=[401,402],native_roles=['coupled_spill','matched_no_spill'],controlled_calibration_seeds=cal['seeds'],
                 resident=cal['deployment'],native_seed_summaries=seed_stats,controlled_contexts=contexts,
                 controlled_same_input_same_work_context_ratios=grouped,
                 prefill_decode_timing_available=False,GPU_started=False,frozen_sources_modified=False,
                 limitations=['Same pool_id across G and S does not imply same tokenization, prompt, parameters or generation work.',
                    'generated_tokens includes terminal tokens; tokens/s over whole handler service is not decode throughput.',
                    'ProfileAdapter service includes infer preprocessing, generation, output processing, device synchronization and cache reclamation; no per-token or TTFT timestamps.',
                    'Same input/family/work solo reference removes observed token/prompt changes but not thermal/clock/cache/drift differences between calibration301/302 and native401/402.',
                    'Controlled contexts are launch designs, not guaranteed whole-request kernel overlap; compare with independent interval study.',
                    'No GPU kernel trace, SM occupancy, memory bandwidth or prefill/decode split is identified.',
                    'Service accounting identity is descriptive; neither residual is automatically causal GPU contention.'])
    source=ROOT/'artifacts/retained-replica-profiles-20261021/source-qa/gsdeploy/backends/compact_native.py'
    inputs.refs[str(source.resolve())]=dict(path=str(source.resolve()),sha256=digest(source.read_bytes()))
    lines=['# 分流请求工作量与服务减速拆分','',
           '本分析只读已审401/402四轮主策略/自身关闭分流对照，以及同驻留受控301/302校准。跨模型service比同时含模型/提示/输入token/输出token和竞争状态差异，不能全部归于GPU干扰。','',
           '|seed|分流数|G输出token均值|S输出token均值|G更长/相等/更短|G/S服务比均值|同族S配对工作量完全一致|',
           '|---|---|---|---|---|---|---|']
    for s in seed_stats:
        lines.append(f"|{s['seed']}|{s['spills']}|{s['G_generated_tokens']['mean']}|{s['S_off_generated_tokens']['mean']}|{s['G_output_longer']}/{s['G_output_equal']}/{s['G_output_shorter']}|{s['G_over_S_service_ratios']['mean']}|{s['same_family_text_all_work_equal']}/{s['same_family_text_requests']}|")
    lines+=['','逐分流spill-work.csv保留G/off-S实际input_tokens、generated_tokens、effective_generation、finish_reason、service、cache回收与同输入独占参考；same-family-pairs.csv保存全部同request同family on/off工作量核对。generated_tokens包含终止token，跨模型tokenizer/提示不同，输入token数不可当相同FLOPs。','',
        '受控参照为同一个text×2+G2×1+OCR×1驻留，G/S0/S1表示单活跃handler设计，SS/SG/SSG按原校准context名称保存；每个上下文/seed/pool_id只与同family、同input/输出token、同有效generation、同finish/request/protocol/prediction的solo记录比较。不匹配就保留缺失，不以不同输入平均替代。','',
        'service accounting：G_native−S_off=(G_solo−S_solo)+(G_native−G_solo)−(S_off−S_solo)。该恒等式只分开跨模型独占参考差与各自相对参考残差；残差仍含时钟、热状态、cache、CPU等跨阶段变化，不能全部称因果竞争成本。','',
        'prefill与decode不可从现有字段识别：没有TTFT、逐token时间或kernel trace。源码ProfileAdapter.run的service_ms包住infer、同步和cache回收；infer又含准备/完整generate/解码输出处理，因此service/generated_tokens不是decode每token时延。G的repetition_penalty等有效generation也与S不同。独立interval分析才能描述handler重叠，仍不等于GPU kernel并发或带宽瓶颈。','',
        '## 同输入同工作量受控上下文参考','',
        '|family|context|匹配数/总数|service/同seed solo均值|中位数|','|---|---|---|---|---|']
    for g in grouped:lines.append(f"|{g['family']}|{g['context']}|{g['exact_work_same_seed_matched']}/{g['requests']}|{g['ratio']['mean']}|{g['ratio']['median']}|")
    lines+=['','此表是校准阶段whole-handler比，不是native中每次分流的物理反事实。401/402原生相对301/302参考比及匹配数在summary.json与spill-work.csv；完整受控输入/工作量集合在calibration-work.json。','']
    outputs={'summary.json':json.dumps(summary,indent=2,ensure_ascii=False)+'\n','spill-work.csv':csv_dump(spills),
             'same-family-pairs.csv':csv_dump(same_family),'calibration-work.json':json.dumps(cal_refs,indent=2,ensure_ascii=False)+'\n',
             'calibration-same-work-ratios.csv':csv_dump(cal_comparisons),'report.zh.md':'\n'.join(lines)}
    manifest=dict(schema='gsdeploy.token_decomposition_manifest.v1',inputs=list(inputs.refs.values()),
                  source=dict(path=str(Path(__file__).resolve()),sha256=digest(Path(__file__).read_bytes())),
                  outputs=[dict(path=str((out/n).resolve()),sha256=digest(v.encode())) for n,v in outputs.items()],
                  GPU_started=False,frozen_sources_modified=False)
    outputs['manifest.json']=json.dumps(manifest,indent=2,ensure_ascii=False)+'\n'
    out.mkdir(parents=True,exist_ok=False)
    for n,v in outputs.items():(out/n).write_text(v)
    print(json.dumps(seed_stats,ensure_ascii=False))

if __name__=='__main__':main()
