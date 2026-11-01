"""Read-only evidence verification followed by publishing current configuration.

No GPU work, no rewriting frozen packets, no pool-admission override.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/baichuan-no-synthesis-20261021'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def memory_observation(folder):
    p=read(folder/'profile.json')
    tel=read(folder/'telemetry.json')
    resident=p['resident_after_warmup_mib']
    # Bind to the actual warm-resident observation, rather than another later
    # request that happens to have the same whole-board memory reading.
    end_ns=p['startup_energy']['end_ns']
    snapshots=[t for t in tel if t['time_ns']==end_ns]
    assert len(snapshots)==1 and snapshots[0]['processes'],'Missing exact warm-resident process sample'
    warm=snapshots[0]
    assert abs(warm['used_mib']-resident)<.01
    return dict(warm_board_used_GiB=resident/1024,
        warm_process_used_GiB=sum(x['used_mib'] for x in warm['processes'])/1024,
        peak_board_used_GiB=p['peak_board_used_mib']/1024,
        peak_process_used_GiB=max(sum(x['used_mib'] for x in t['processes']) for t in tel)/1024,
        board_baseline_before_load_GiB=tel[0]['used_mib']/1024,
        scope='Warm residency is post-warmup; peak includes loading and complete measured requests; NVML sampled.')

def main():
    state=read(BASE/'state.json')
    assert state['status']=='COMPLETE_FOUR_TASK_EQUIVALENCE'
    assert set(state['completed'])=={'text','image','ocr','asr'}
    release=read(BASE/'release.json')
    for p,digest in release['source_hashes'].items():assert sha(p)==digest,p
    assert sha(ROOT/'scripts/verify_baichuan_no_synthesis_20261021.py')==release['script_sha256']
    tasks={};bindings={str(BASE/'release.json'):sha(BASE/'release.json')}
    total=0
    for e in release['jobs']:
        task=e['task'];folder=Path(e['folder']);c=read(folder/'comparison.json')
        assert c['unchanged'] and c['valid']==c['count']==e['count'] and not c['changes']
        newraw=folder/f'results/{task}-round0.jsonl';oldraw=Path(e['baseline'])/(task+'-round0.jsonl')
        assert sha(newraw)==c['raw_sha256'] and sha(oldraw)==e['baseline_raw_sha256']
        assert sha(folder/'worker.json')==e['spec_sha256'] and sha(folder/'job.json')==e['job_sha256']
        before=memory_observation(Path(e['baseline']))
        after=memory_observation(folder/'results')
        c.update(memory_before=before,memory_after=after,
            resident_board_reduction_GiB=before['warm_board_used_GiB']-after['warm_board_used_GiB'],
            resident_process_reduction_GiB=before['warm_process_used_GiB']-after['warm_process_used_GiB'])
        tasks[task]=c;total+=c['count']
        for p in [folder/'worker.json',folder/'job.json',folder/'comparison.json',newraw,oldraw,
                  folder/'results/profile.json',folder/'results/telemetry.json']:
            bindings[str(p)]=sha(p)
    assert total==376
    ocr=ROOT/'artifacts/ocr-small-candidate-screen-20261021/retained-ppocrv6-evaluation200-v1'
    assessment=read(ocr/'assessment.json')
    assert all(m['valid']==200 for m in assessment['models'].values())
    assert read(ocr/'resource-output-equivalence.json')['passed']
    for m in assessment['models'].values():assert sha(m['raw'])==m['raw_sha256']
    for p in [ocr/'assessment.json',ocr/'resource-output-equivalence.json']:
        bindings[str(p)]=sha(p)
    summary=dict(status='COMPLETE_USER_REQUEST_OCR_AND_BAICHUAN_MEMORY',
        requests_verified=total,all_outputs_and_termination_unchanged=True,tasks=tasks,
        ocr_complete_evaluation=str(ocr/'report.zh.md'),
        no_new_independent_confirmation_claim=True,pool_admitted=False,bindings=bindings)
    save(BASE/'assessment.json',summary)
    cfg=ROOT/'configs/baichuan_no_synthesis_20261021.json'
    config=read(cfg);config.update(state='VERIFIED_FOUR_TASK_EQUIVALENCE',
        verification_assessment=str(BASE/'assessment.json'),
        verification_assessment_sha256=sha(BASE/'assessment.json'),
        observed_memory_by_task={t:c['memory_after'] for t,c in tasks.items()})
    save(cfg,config)
    lines=['# 百川移除语音合成分支：完成报告','',
        '376/376条四任务复验有效，原始输出、预测、生成token数和终止信息逐项保持一致。',
        'QA80是已有确认集的等价性回放；未调参、未训练、不增加独立样本数。图像200、OCR32、ASR64为既有校准。','',
        '| 任务 | 请求 | 原整卡预热驻留GiB | 新整卡预热驻留GiB | 原进程预热驻留GiB | 新进程预热驻留GiB | 新整卡采样峰值GiB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for t,c in tasks.items():
        b,a=c['memory_before'],c['memory_after']
        lines.append(f'| {t} | {c["count"]} | {b["warm_board_used_GiB"]:.4f} | {a["warm_board_used_GiB"]:.4f} | {b["warm_process_used_GiB"]:.4f} | {a["warm_process_used_GiB"]:.4f} | {a["peak_board_used_GiB"]:.4f} |')
    lines+=['','整卡与进程数值分开列出，未用减去空闲基线的方法伪造进程驻留。峰值包含加载和完整任务请求，采样可能遗漏更短尖峰。',
        '仅移除audio_head、audio_tokenizer.audio_decoder、audio_tokenizer.audio_flow_matching_decoder；实际移除参数存储3,466,782,512字节。音频输入编码器、桥接、视觉和语言主干保留。',
        '原始checkpoint未删；若调用已停用的语音合成路径，将显式报错。加载阶段仍先加载原checkpoint，故加载峰值并未消除。',
        '当前执行配置：configs/baichuan_no_synthesis_20261021.json；绑定的source_root和任务protocol均已固定。之前的冻结实验文件不改写。',
        '所有本轮任务已结束。未开始新的QA实验、匹配成本、共驻副本或系统矩阵；不能据此宣布整池准入。','']
    (BASE/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))

if __name__=='__main__':main()
