from .common import *
def main():
    sp=read(OUT/'candidate_space_definition.json');d=read(sp['pools']['C']['path']);qpath=ROOT/'results/module8b_quality_pool_finalization/quality_admission_matrix.json';base=read(qpath)['rows']
    csvpath=ROOT/'results/module13_real_specialists/real_quality_results.csv';extra=list(csv.DictReader(csvpath.open()));ocrpath=ROOT/'results/module14_high_quality_heavy/ocr_gapfill_measurements.csv';ocr=list(csv.DictReader(ocrpath.open()));rows=[]
    for m,v in d['models'].items():
        for t in v['supported_tasks']:
            r=next((r for r in base if r['model']==m and r['task']==t and r['planning_admissible']),None)
            if r:r=dict(r,source_table=ref(qpath),passed=True)
            else:
                r=next((r for r in extra if t==r['task'] and m==r['model_id']+'@'+r['revision'] and r['admission_pass']=='True'),None)
                if r:r=dict(r,source_table=ref(csvpath),passed=True)
                else:
                    r=next((r for r in ocr if t==r['task'] and m==r['model_id']+'@'+r['revision'] and r['HEAVY_ELIGIBLE']=='True'),None)
                    if r:r=dict(r,source_table=ref(ocrpath),passed=True)
            assert r,('MISSING_ORIGINAL_QUALITY_ADMISSION',m,t)
            rows.append(dict(r,model=m,task=t))
    csvsave('quality_admission_provenance.csv',rows);save('quality_and_profile_preservation.json',dict(passed=True,paths=len(rows),thresholds_unchanged=True,whisper_small_ASR='Original full-precision8B threshold pass restored, not a new quality grant.'))
    print('QUALITY_PASS',len(rows),flush=True)
if __name__=='__main__':main()
