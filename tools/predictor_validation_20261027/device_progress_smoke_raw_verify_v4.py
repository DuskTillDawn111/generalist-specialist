"""Descriptor-bound raw review using frozen v3 probe/output algorithms."""
import argparse
import json
from pathlib import Path

from . import device_progress_smoke_raw_verify_v3 as core
from .device_progress_review_contract_v4 import V6,qualify_roles,monitoring_boundaries
from .implementation_source_closure_v27 import implementation_sources,strict_references


def run(raw_path,output,contract=V6):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    raw_ref=core.receipt(raw_path);raw=json.loads(Path(raw_path).read_text());role=qualify_roles(raw,contract)
    output_review=core.reparse_outputs(raw);result=core.validate_raw(raw,output_review['policies']);result['issues'].extend(output_review['issues'])
    monitor=monitoring_boundaries(raw,contract);result['issues'].extend(monitor['issues'])
    if result['issues']:result['status']='FAIL'
    sources= strict_references([role['exact_pipeline_sources'],output_review['sources'],implementation_sources(__file__)])
    result.update(schema='gsdeploy.device_progress_smoke_CPU_raw_contract_review.v4',raw=raw_ref,qualified_role=role,
        output_protocol_review=output_review,monitoring_review=monitor,verifier_role_binding_revision=4,sources=sources,
        GPU_started_by_verifier=False,validation_labels_read=False,device_API_executed_by_verifier=False,
        limitations=['Probe and original CPU protocol parsing reuse frozen v3 algorithms; execution schema/source/scope comes exclusively from the exact descriptor.',
            'JSON consistency does not independently authenticate native CUDA Event calls. Native API evidence and conditional source associations require separate trace review.',
            'Three language owners use recorded-stream probes. OCR progress, other streams, global request completion, exact device completion time and SM service remain unknown.',
            'Monitoring covers startup through the final drain sample before worker close. Shutdown memory is not sampled; worker/owned-process exits require supervisor provenance review.',
            'Legal explicit no-answer checks protocol validity only. No task quality, speedup, paired equality or predictor qualification is asserted.'])
    if core.receipt(raw_path)!=raw_ref:raise ValueError('Raw changed during review')
    for ref in sources:
        if core.receipt(ref['path'])!=ref:raise ValueError('Source changed during review')
    out.mkdir(exist_ok=False);core.save_synced(out/'report.json',result);return core.receipt(out/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.raw,args.output)))
