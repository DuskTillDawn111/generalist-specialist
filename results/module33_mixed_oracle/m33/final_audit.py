"""Scientific provenance/integrity, independent of whether optimizer succeeds."""
from .common import *

def main():
    f=read(OUT/'frozen_gpu_manifest_v4.json');checks={};failed=[]
    for label,refs in [('frozen_sources',f['sources']),('historical_native_sources',f['native_sources']),('frozen_request_inputs',f['inputs'])]:
        bad=[r['path'] for r in refs if ref(r['path'])['sha256']!=r['sha256']];checks[label]=not bad;failed+=bad
    problem=read(OUT/'frozen_problem_manifest.json');sp=read(problem['space']['path'])
    refs=[problem['space'],problem['utility_definition'],problem['utility_reference'],*sp['pools'].values()]
    checks['frozen_problem_inputs']=all(ref(r['path'])['sha256']==r['sha256'] for r in refs)
    checks['history_SHA256']=read(OUT/'history_integrity_after.json')['passed']
    checks['original_quality_admission']=read(OUT/'quality_and_profile_preservation.json')['passed']
    checks['CPU_independent_enumeration']=read(OUT/'CPU_oracle_correctness.json')['passed']
    checks['CPU_frozen_reproduction']=read(OUT/'CPU_frozen_reproduction_acceptance.json')['passed']
    gate=read(OUT/'module33_gate.json');dev=read(OUT/'development_observations.json');pairs=read(OUT/'pair_observations.json');candidates=read(OUT/'gpu_candidates.json')
    checks['GPU_scope_honest']=gate['GPU_ORACLE_INCOMPLETE']==(len({o['candidate_id'] for o in dev})!=len(candidates) or not all(o['technical_valid'] for o in dev))
    checks['NULL_for_invalid_or_unstable']=all(o['U'] is None for p in (OUT/'gpu_observations').glob('*.json') for o in [read(p)] if not o['eligible'])
    mapping=[];slotmap={s['candidate_id']:s for s in f['slots']}
    for o in dev+pairs:
        raw=read(o['raw']['path']);slot=raw['slot'];candidate=next(c for c in candidates if c['candidate_id']==o['candidate_id'])
        assert identity(slot['deployment'],slot['routing'])==identity(candidate['deployment'],candidate['routing'])
        assert slot['lambda_rps']==sp['loads']['ASR32B']['rates'] and slot['safe_memory_budget_mib']==sp['budget_mib']
        # Reused native harness's raw provenance field references v1. This sidecar
        # names the actual launch amendment, retaining raw bytes and proving that
        # scientific configurations/trajectories are unchanged across versions.
        actual=OUT/('frozen_gpu_manifest_v3.json' if slot['run_id'].startswith('DEV_CACHE_READY') else 'frozen_gpu_manifest_v4.json')
        mapping.append(dict(run_id=slot['run_id'],raw_ref=raw['frozen_manifest'],actual_launch_manifest=ref(actual),
            independent_selection_manifest=ref(OUT/'independent_pair_selection.json') if slot['stage']=='INDEPENDENT_PAIR' else None,
            scientific_configuration_matches_frozen_candidate=True,
            raw_reference_limitation='Raw reused-harness frozen_manifest references initial v1; actual launch/cache/assessment amendments are provided here without rewriting measurement records.'))
    save('run_manifest_provenance.json',mapping)
    required=['frozen_problem_manifest.json','candidate_space_definition.json','feasible_deployments.csv','cpu_oracle_results.csv','optimizer_decisions.csv','predicted_regret.csv','optimizer_runtime_benchmark.csv','gpu_oracle_runs.csv','gpu_empirical_regret.csv','ranking_error_analysis.md','module33_gate.json','module33_summary.md']
    checks['required_artifacts']=all((OUT/n).is_file() for n in required)
    checks['no_Module34']=gate['Module34_started'] is False
    # All evidence/configurations may be complete even if a ranking mismatch is
    # observed. Audit scientific validity, never manufacture a success gate.
    save('final_artifact_review.json',dict(passed=all(checks.values()),checks=checks,failed_refs=failed,
        scope='Provenance/feasibility/measurement accounting audit, not an additional optimizer success criterion',gate=gate['primary_gate']))
    print('FINAL_AUDIT',all(checks.values()),checks,flush=True)
    if not all(checks.values()):raise SystemExit(1)
if __name__=='__main__':main()
