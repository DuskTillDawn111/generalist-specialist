"""Read existing offline fingerprints only. Never invoke nsys or profile workers."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAR = ROOT/'benchmark/results/characterization'
FEATURE_CSV = CHAR/'c1_analysis_v1/c1_service_path_features.csv'


def source(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load_resource_profiles(catalog, csv_path=FEATURE_CSV):
    """Each path exposes dynamic numeric values, nulls, units, provenance and limitations.

    CSV was derived by build_c1_feature_dataset.py. NVML identity/count/summary
    values are cross-checked against its pinned inputs. Kernels reuse the existing
    derived CSV with Nsight report paths; no profiling/export subprocess is run.
    """
    csv_path=Path(csv_path)
    data=[]
    if csv_path.exists():
        with csv_path.open(newline='') as handle:
            data=list(csv.DictReader(handle))
    categories=sorted(k for k in (data[0] if data else []) if k.startswith('kernel_') and k.endswith('_pct'))
    fields={'offline_gpu_util_pct':'gpu_util_pct','offline_memory_util_pct':'mem_util_pct',
            'offline_power_w':'power_w','offline_sm_clock_mhz':'sm_clock_mhz',
            'offline_device_vram_used_mib':'vram_used_mib','offline_energy_j_per_request':'energy_j_per_req',
            'kernel_time_ms_per_request':'kernel_gpu_ms_per_req'}
    fields.update({'kernel_fraction_'+k[len('kernel_'):-len('_pct')].lower():k for k in categories})
    by_pair={}
    for row in data:
        pair=(row['task'],row['model'])
        if pair in by_pair: raise ValueError('Duplicate resource profile: '+str(pair))
        by_pair[pair]=row
    profiles={}
    for m,model in sorted(catalog.items()):
        for k in sorted(model.supported_tasks):
            pair=(k,m); row=by_pair.get(pair)
            if m=='qwen': folder='standalone_v1'; count=30
            elif m=='whisper' and k=='asr': folder='specialist_nvml_v1'; count=30
            else: folder='specialist_nvml_v2'; count=1000
            prefix=CHAR/folder/f'{m}_{k}_nvml_n{count}'
            nvml_path=Path(str(prefix)+'.nvml_summary.json')
            identity_path=CHAR/folder/f'{m}_{k}_nvml_profiled_n{count}.jsonl.summary.json'
            nsys=CHAR/('qwen_nsys_v1' if m=='qwen' else 'specialist_nsys_v1')/f'{m}_{k}_n30.nsys-rep'
            values={f:None for f in fields}; reasons={}; evidence=[]
            nvml_ok=False
            if row and nvml_path.exists() and identity_path.exists():
                nvml=json.loads(nvml_path.read_text()); identity=json.loads(identity_path.read_text())
                if (identity.get('target_worker'),identity.get('target_task'),identity.get('target_count'),identity.get('protocol'))!=(m,k,count,'active_single_profile_v1'):
                    raise ValueError('Resource identity mismatch: '+str(identity_path))
                nvml_ok=nvml.get('samples',0)>0 and nvml.get('duration_s',0)>0
                evidence += [source(nvml_path),source(identity_path)]
            for feature,column in fields.items():
                is_kernel=feature.startswith('kernel_')
                available=bool(row) and (nsys.exists() if is_kernel else nvml_ok)
                if available and row.get(column) not in ('',None):
                    value=float(row[column])
                    if not math.isfinite(value) or value<0: raise ValueError('Invalid fingerprint value')
                    if feature.startswith('kernel_fraction_'): value/=100.
                    if not is_kernel:
                        expected=(nvml['estimated_energy_j']/count if column=='energy_j_per_req' else nvml[column]['mean'])
                        if not math.isclose(value,expected,rel_tol=1e-9,abs_tol=1e-9):
                            raise ValueError('Derived NVML CSV disagrees with source')
                    values[feature]=value
                else: reasons[feature]='No verified offline service-path measurement'
            if row: evidence.append(source(csv_path))
            fractions=[v for f,v in values.items() if f.startswith('kernel_fraction_')]
            if fractions and all(v is not None for v in fractions) and not math.isclose(sum(fractions),1.,abs_tol=1e-6):
                raise ValueError('Kernel category fractions do not sum to one')
            profiles[pair]=dict(values=values,missing=reasons,sources=evidence,
                kernel_report_path=str(nsys.resolve()) if nsys.exists() else None,
                derivation_script=source(ROOT/'joint_deploy/build_c1_feature_dataset.py'),
                nvml_sample_count=nvml.get('samples') if nvml_ok else None,
                nvml_duration_s=nvml.get('duration_s') if nvml_ok else None,
                limitations=['Offline device-wide means; not runtime state or isolated per-process utilization.',
                    'NVML sample windows are short and protocols/counts differ across paths.',
                    'Kernel fractions are time fractions by existing name taxonomy; kernel time is not wall-clock GPU busy time.',
                    'Fingerprint metrics and Stage 1 service demand use separate experiments; do not replace s0.'])
    return profiles
