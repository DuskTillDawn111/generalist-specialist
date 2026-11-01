"""Offline output equivalence audit; never called by inference workers."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/retained-replica-profiles-20261021'
QA=ROOT/'artifacts/qa-development-revision-20261020/tydiqa-new-native-confirmation80-v2'
FRESH=ROOT/'artifacts/remaining-task-confirmation-20261020'


def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(v) for v in Path(p).read_text().splitlines() if v]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def baseline(e):
    task,slot=e['task'],e['slot']
    if slot=='ocr':
        return ROOT/'artifacts/ocr-small-candidate-screen-20261021/results/retained-ppocrv6-full200-v1/outputs.jsonl'
    if slot=='asr':
        return ROOT/'artifacts/chandra-replacement-20261020/results/granite-full/granite-full/asr-round0.jsonl'
    if slot=='generalist_1' and task!='text':
        return FRESH/'results'/f'fresh-{task}-{slot}'/f'fresh-{task}-{slot}'/(task+'-round0.jsonl')
    if task=='text':
        if slot=='generalist_1':
            return ROOT/'artifacts/baichuan-no-synthesis-20261021/text/results/text-round0.jsonl'
        return QA/('qa80-'+slot)/'results/text-round0.jsonl'
    return FRESH/'results'/f'fresh-{task}-{slot}'/f'fresh-{task}-{slot}'/(task+'-round0.jsonl')


def main():
    results=[]
    packages=[OUT,OUT/'protocol-binding-correction-v2']
    entries=[(package,e) for package in packages if (package/'release.json').exists()
             for e in read(package/'release.json')['entries']]
    for package,e in entries:
        original=baseline(e)
        old={v['sample_id']:v for v in rows(original)}
        single={}
        fields=['raw_output','prediction','success','finish_reason','generated_tokens','terminal_token_id','eos_token_ids']
        if e['slot']=='ocr':fields=['raw_output','success']
        for n in e['grid']:
            folder=package/'runs'/f'{e["id"]}-n{n}'
            path=folder/'profile.json'
            if not path.exists() or read(path)['status']!='COMPLETE':continue
            data=[v for p in sorted(folder.glob(e['task']+'-round*.jsonl')) for v in rows(p)]
            missing=[v['sample_id'] for v in data if v['sample_id'] not in old]
            changes=[];vs_single=[]
            for v in data:
                sid=v['sample_id'];now={k:v.get(k) for k in fields}
                if sid in old:
                    changed=[k for k in fields if v.get(k)!=old[sid].get(k)]
                    if changed:changes.append(dict(sample_id=sid,replica=v['replica'],fields=changed))
                if n==1:single.setdefault(sid,now)
                elif single.get(sid)!=now:vs_single.append(dict(sample_id=sid,replica=v['replica']))
            results.append(dict(id=e['id'],replicas=n,requests=len(data),package=str(package),
                baseline=str(original),baseline_sha256=digest(original),
                missing_baseline_ids=sorted(set(missing)),changed_requests_vs_prior=len(changes),
                changed_unique_ids_vs_prior=len({v['sample_id'] for v in changes}),
                changes_vs_prior=changes,changed_requests_vs_n1=len(vs_single),changes_vs_n1=vs_single,
                fields=fields,source_raw_sha256={str(p):digest(p) for p in folder.glob(e['task']+'-round*.jsonl')}))
    target=OUT/'output-equivalence.json';tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(scope='Offline comparison only; no output-dependent cohort selection',
        rows=results),ensure_ascii=False,indent=2)+'\n');tmp.replace(target)
    print('Audited',len(results),'profiles;',sum(v['changed_requests_vs_n1'] for v in results),
        'request differences vs n1;',sum(v['changed_requests_vs_prior'] for v in results),'vs prior')


if __name__=='__main__':main()
