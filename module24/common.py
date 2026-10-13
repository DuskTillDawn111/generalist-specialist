from pathlib import Path
import json, csv, hashlib, time, itertools, copy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/module24_utility_calibration'
G = 'qwen'
S = 'openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1'
TEXT = 'roberta_sst2'
IMAGE = 'convnext_tiny'
OCR = 'baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt'
MODELS = [G, S, TEXT, IMAGE, OCR]
BUDGET = 34567.5534
RATES = {s:dict(text=4.,image=2.,ocr=5.,asr=q,lid=.03,visual_qa=.2)
         for s,q in [('LOW',.02),('MID',.80),('HIGH',1.52)]}
SOURCE = ROOT/'results/module14f_memory_ocr_x3/heavy_empirical_x3_reconciled.json'

def read(p): return json.loads(Path(p).read_text())
def save(n,d):
    p=OUT/n; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvsave(n,rows,fields=()):
    p=OUT/n; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys([*fields,*(k for r in rows for k in r)])))
        w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def ref(p):
    p=Path(p).resolve(); h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size)
def identity(v):return json.dumps(v,sort_keys=True,separators=(',',':'))
def combo(x):return identity(sorted(x))
def group(x,rates):return identity([sorted(x),rates])
def metadata():
    files=list((ROOT/'results').rglob('*'))
    files += [p for d in ROOT.glob('module*') if d.is_dir() and d.name!='module24' for p in d.rglob('*')]
    return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in files
            if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts}
def inputs():
    d=read(SOURCE)
    d=copy.deepcopy(d)
    for key in ['models','capacity_factors','memory_table_mib','evidence_bounds','model_evidence']:
        d[key]={m:d[key][m] for m in MODELS}
    refs=read(ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json')['latency_reference_ms']
    weights=read(ROOT/'results/module20_utility_optimizer/utility_definition.json')['weights']
    assert weights==dict(Q=0.,T=.25,L=.3,P=.3,M=.15)
    return d,refs,weights

def route(d,x,rates,z):
    fractions=dict(zip(['text','image','asr'],map(float,z)))
    rr={t:{} for t in rates}
    for t,q in rates.items():
        paths=[m for m in x if t in d['models'][m]['supported_tasks']]
        if t in fractions and len(paths)==2:
            f=fractions[t];sp=next(m for m in paths if m!=G)
            if f>1e-12:rr[t][G]=q*f
            if f<1-1e-12:rr[t][sp]=q*(1-f)
        else:
            assert len(paths)==1,(t,x)
            rr[t][paths[0]]=q
    return rr

def summarize_samples(v):
    return dict(mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95)),
                p50_ms=float(np.percentile(v,50)),min_ms=float(np.min(v)),max_ms=float(np.max(v)),count=len(v)) if len(v) else None
