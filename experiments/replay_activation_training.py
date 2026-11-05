"""Replay original training receipts; this command cannot qualify a predictor."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.fit_activation_native import load_bundle
from gsdeploy.forecasting.native_validation import replay_native_training


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();fit=json.loads(args.fit.read_text())
    bundles=[load_bundle(Path(s.rsplit(':sha256:',1)[0]).parent) for s in fit['source_receipts']]
    paths=sorted((ROOT/'gsdeploy/forecasting').glob('*.py'))+[Path(__file__)]
    sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report=replay_native_training(fit,bundles)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha for path,sha in sources.items()):
        raise ValueError('Training replay source changed during execution')
    report['prediction_sources']=sources
    with args.output.open('x') as out:json.dump(report,out,ensure_ascii=False,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('runs','prediction_sources')},ensure_ascii=False))
    for run in report['runs']:
        print(json.dumps({k:run[k] for k in ('run_id','latency_errors','native_work_errors','unsupported')},ensure_ascii=False))


if __name__=='__main__':main()
