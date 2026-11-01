"""Read-only CPU inventory of eight completed captures and lossless mappings."""
from pathlib import Path
import json
import shutil
import time
from . import whole57_joint_training_v23 as c
from . import unknown_preflight_archive_v1 as a


def inventory(output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);rows=[]
    for domain in (c.LIGHT,c.NATURAL):
        for route in ('balanced','skewed','spill_low','spill_high'):
            run=a.ROOT/'artifacts/predictor-whole57-joint-training-v23-20261028-prepared03'/domain/route
            rd=a.ROOT/'artifacts/predictor-whole57-joint-training-v23-20261028-reviewed01'/domain/route
            plan=c.read(run/'plan.json');raw=c.read(run/'raw.json');e=c.read(run/'execution.json');r=c.read(rd/'report.json')
            audit=c.audit(run)
            refs={n:a.receipt(run/n) for n in ('plan.json','context.json','execution-start.json','execution.json','raw.json','stdout.log','stderr.log')}
            for key,name in (('plan','plan.json'),('context','context.json')):assert raw[key]==refs[name] and e[key]==refs[name]
            assert e['raw']==refs['raw.json']
            for name,ref in r['evidence'].items():assert ref==refs[name]
            assert r['status']=='PASS' and not r['issues'] and r['requests']==57
            assert e['status']=='COLLECTED_TRAINING_REQUIRES_INDEPENDENT_REVIEW' and raw['status']=='completed'
            assert len(raw['request_records'])==57 and len(raw['priming_request_records'])==12
            assert not raw['errors'] and not raw['runtime_errors'] and not raw['telemetry_errors']
            assert all(x['status']=='completed' for x in raw['request_records']+raw['priming_request_records'])
            assert raw['planned_public_cohort']==plan['trace'] and len(plan['trace'])==57
            w=e['watchdog'];assert w['status']=='EXITED' and w['return_code']==0 and not w['cleanup']['live_survivors']
            assert e['final_source_audit']==raw['final_source_audit']=='PASS' and 'exception_cleanup' not in e
            assert all(x['exit_code']==0 and not x['forced'] and x['bye_received'] for x in raw['shutdown'].values())
            assert r['host_lifecycle']==a.receipt(rd/'host-lifecycle.json')
            independent=a.ROOT/('artifacts/predictor-whole57-joint-'+('light' if domain==c.LIGHT else 'natural')+'-'+route.replace('_','-')+'-independent-review-20261028-cpu01/report.json')
            independent_value=c.read(independent)
            assert independent_value['status'].startswith('PASS')
            trace=None
            if domain==c.LIGHT:
                assert r['trace']['sqlite']==a.receipt(rd/'trace-whole57.sqlite')
                for name in ('native-first-api-kernel.json','session-clock.json','session-metadata.json'):
                    refs['review/'+name]=a.receipt(rd/name)
                refs['trace-whole57.nsys-rep']=a.receipt(run/'trace-whole57.nsys-rep')
                manifest=a.ROOT/'artifacts/predictor-v23-light-balanced-derived-lossless-archive-20261028-cpu01/manifest.json' if route=='balanced' else a.ROOT/('artifacts/predictor-whole57-derived-lossless-v32-'+route+'-20261028/manifest.json')
                retirement=a.ROOT/'artifacts/predictor-v23-light-balanced-derived-lossless-retirement-20261028-cpu01/retirement-receipt.json' if route=='balanced' else a.ROOT/('artifacts/predictor-whole57-derived-retirement-v32-'+route+'-20261028-01/receipt.json')
                m=c.read(manifest);record=m['originals'][0];ret=c.read(retirement)
                assert m['source_report']==a.receipt(rd/'report.json')
                assert record['original']==r['trace']['parsed'] and a.receipt(record['archive']['path'])==record['archive']
                assert not Path(record['original']['path']).exists() and ret['manifest']==a.receipt(manifest)
                trace=dict(nsys=refs['trace-whole57.nsys-rep'],sqlite=r['trace']['sqlite'],
                    parsed_original_receipt_preserved=r['trace']['parsed'],parsed_original_path_present=False,
                    lossless_manifest=a.receipt(manifest),retirement_receipt=a.receipt(retirement),archive=record['archive'],
                    original_bytes=record['original_size'],gzip_bytes=record['archive_size'],
                    original_receipt_not_fabricated=True,full_roundtrip_verified_by_retirement=True,restore_command=ret['restore_command'])
            origin=raw['phase_start_ns'];assert raw['deadline_ns']==origin+30_000_000_000
            rows.append(dict(domain=domain,route=route,run=str(run),refs=refs,review=a.receipt(rd/'report.json'),
                independent_review=a.receipt(independent),independent_review_status=independent_value['status'],
                review_status=r['status'],source_audit=audit,request_count=57,priming_count=12,all69_completed=True,
                planned_complete_cohort_preserved=True,watchdog=dict(status=w['status'],return_code=0,
                    sample_count=len(w['samples']),minimum_free_bytes=min(x['free_bytes'] for x in w['samples']),
                    elapsed_seconds=w['samples'][-1]['elapsed_seconds'],live_survivors=w['cleanup']['live_survivors']),
                errors=raw['errors'],runtime_errors=raw['runtime_errors'],telemetry_errors=raw['telemetry_errors'],shutdown=raw['shutdown'],
                original_window_seconds=30.,offer_loop_end_seconds=(raw['arrival_loop_end_ns']-origin)/1e9,
                observation_end_seconds=(raw['observation_end_ns']-origin)/1e9,trace=trace,training_only=True,validation_started=False))
    report=dict(schema='gsdeploy.whole57_training_eight_run_actual_inventory.v23',
        status='ALL_EIGHT_ACTUAL_COLLECTED_FROZEN_AND_INDEPENDENT_REVIEW_PASS',
        prepared_campaign=a.receipt(a.ROOT/'artifacts/predictor-whole57-joint-training-v23-20261028-prepared03/campaign.json'),
        collector=a.receipt(c.__file__),reviewer=a.receipt(Path(c.__file__).with_name('whole57_joint_review_v23.py')),
        runs=rows,total_measured_requests=456,total_priming_requests=96,all_eight_source_audits_pass=True,
        all_eight_zero_errors=True,all_eight_no_owned_survivors=True,completed_capture_repeated=False,
        uncapped_shared=True,batch_size=1,weight_reuse=False,six_equal_weights_retained=True,
        final_free_bytes=shutil.disk_usage(a.ROOT).free,GPU_started_by_inventory=False,validation_started=False,
        predictor_accuracy_established=False,tool=a.receipt(__file__),created_unix_ns=time.time_ns())
    a.save_synced(out/'report.json',report);return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    result=inventory(args.output);print(json.dumps(dict(status=result['status'],runs=len(result['runs']),GPU_started=False)))
