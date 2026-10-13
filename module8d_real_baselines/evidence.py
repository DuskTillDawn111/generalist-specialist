"""Lazy exact-cache sharing; serving, measurement and aggregation remain original."""
import copy, time
from pathlib import Path
from deployment_search.runtime_cache import save_json, file_hash, digest
from deployment_core import Workload
from integer_replica_pilot.evaluator import ReplicaRuntimeEvaluator, ReplicaRuntimeEvaluation, PilotExecutionBlocked, runner, summarize, aggregate
from joint_deploy.measurement.schema import MeasurementConfig
from module8c_runtime_compat.runner import runner as compatible_runner
from .common import ROOT,C,F,OUT,read,ref,validate_raw,verify_inputs


class EvidenceBank:
    def __init__(self):
        self.search={};self.confirmations=[]
        self.reuses=read(OUT/'reuse_execution_log.json')['events'] if (OUT/'reuse_execution_log.json').exists() else []
        self.new_runs=read(OUT/'new_measurement_manifest.json')['runs'] if (OUT/'new_measurement_manifest.json').exists() else []
        self.refresh()

    def refresh(self):
        # Original 8C search entries plus completed new campaign cache entries.
        for root in (C,OUT):
            for path in sorted(root.glob('**/search_cache/runtime_cache_manifest.json')):
                for key,entry in read(path)['configurations'].items():
                    if entry['status']=='complete':
                        assert digest(entry['identity'])==key
                        candidate={'entry':entry,'manifest':str(path),'key':key}
                        if key not in self.search:self.search[key]=candidate
            # Exact confirmation identities are reconstructed from config + frozen evaluator protocol.
            for path in sorted(root.glob('**/confirmation/*.summary.json')):
                rep=read(path)
                if not rep.get('valid_runtime_repetition'):continue
                raw_path=Path(rep['raw_result_path']);config_path=raw_path.with_suffix('.config.json')
                if not config_path.exists():continue
                raw_config=read(config_path)
                existing={r['path'] for r in self.confirmations}
                if str(raw_path) in existing:continue
                role='seed' if raw_path.name.startswith('seed_rep') else 'final' if raw_path.name.startswith('final_rep') else 'state'
                self.confirmations.append({'path':str(raw_path),'summary':str(path),'config':raw_config,
                    'role':role,'origin_campaign':'8C' if C in raw_path.parents else '8D-2'})

    def persist(self):
        save_json(OUT/'reuse_execution_log.json',{'events':self.reuses,'references':len(self.reuses),
            'unique_raw_paths':len({r['original_run']['path'] for r in self.reuses}),
            'shared_measurements_are_not_independent_samples':True})
        save_json(OUT/'new_measurement_manifest.json',{'runs':self.new_runs,'physical_attempts':len(self.new_runs),
            'valid_timed_runs':sum(r.get('valid_timed_execution',False) for r in self.new_runs),
            'failed_attempts_preserved':True})

    def log_reuse(self,owner,phase,rep_index,proof,role=None):
        self.reuses.append({'owner':owner,'phase':phase,'repetition':rep_index,'confirmation_role':role,
            'measurement_source':'reused','original_run':proof['original_run'],
            'original_campaign':'8C' if C in Path(proof['original_run']['path']).parents else '8D-2',
            'all_12_identity_conditions_equal':True,'proof':proof})
        self.persist()

    def run(self,owner,workload,phase):
        def tracked(config,raw_path,log_path,config_path,context_path):
            verify_inputs()
            cap=read(OUT/'campaign_manifest.json')['physical_run_cap']
            if len(self.new_runs)>=cap:raise RuntimeError('Frozen campaign new physical-run ceiling exhausted')
            record={'owner':owner,'workload':workload,'phase':phase,'measurement_source':'newly_measured',
                'raw_result_path':str(raw_path),'config':ref(config_path),'started_unix':time.time(),'status':'started'}
            self.new_runs.append(record);self.persist()
            print(__import__('json').dumps({'event':'physical_start','owner':owner,'phase':phase,
                 'new_physical_attempt':len(self.new_runs),'measurement_seed':config['random_seed']}),flush=True)
            selected=compatible_runner if workload=='no_vqa_phase' else runner
            outcome=selected(config,raw_path,log_path,config_path,context_path)
            record['outcome']=outcome
            if Path(raw_path).exists():
                raw=read(raw_path);record['raw']=ref(raw_path)
                record['valid_timed_execution']=bool(raw.get('valid_execution') and raw.get('measurement_window'))
                record['status']=raw.get('runtime_resource_status') if raw.get('runtime_resource_status')=='RUNTIME_RESOURCE_INFEASIBLE' else raw.get('execution_status','external_interruption')
            else:record.update(status='external_interruption',valid_timed_execution=False)
            self.persist()
            return outcome
        return tracked

    def search_donor(self,identity,workload):
        key=digest(identity)
        donor=self.search.get(key)
        if not donor:return None
        entry=donor['entry'];assert entry['identity']==identity
        assert file_hash(entry['aggregate'])==entry['aggregate_sha256']
        for i,rep in enumerate(entry['repetitions'],1):
            assert file_hash(rep['runtime']['raw_result_path'])==rep['runtime']['raw_sha256']
            validate_raw(rep['runtime']['raw_result_path'],identity,rep['seed'],workload)
        return donor

    def confirmation(self,directory,owner,workload,identity,role,index,banned):
        directory=Path(directory)/'confirmation';directory.mkdir(parents=True,exist_ok=True)
        label=f'{role}_rep{index}';result_path=directory/(label+'.summary.json')
        definitions=read(F/'baseline_definitions.json');seed=definitions['B4']['confirmation_seeds'][index-1]
        config=MeasurementConfig(identity['deployment'],identity['workload'],identity['routing'],
            random_seed=seed,**definitions['measurement_protocol']['settings']).as_dict()
        if result_path.exists():
            rep=read(result_path);assert file_hash(rep['raw_result_path'])==rep['raw_sha256'];return rep
        self.refresh()
        candidates=[r for r in self.confirmations if r['config']==config and r['path'] not in banned]
        candidates.sort(key=lambda r:(r['origin_campaign']!='8C',r['role']!='seed',r['path']))
        if candidates:
            donor=candidates[0];proof=validate_raw(donor['path'],identity,seed,workload)
            rep=copy.deepcopy(read(donor['summary']))
            rep.update(measurement_source='reused',original_source_path=donor['path'],
                       repetition=index,random_seed=seed,post_search_confirmation=True)
            save_json(result_path,rep);self.log_reuse(owner,'confirmation',index,proof,role)
        else:
            raw_path=directory/(label+'.json');config_path=directory/(label+'.config.json')
            process_path=raw_path.with_suffix('.process.json')
            if raw_path.exists() or process_path.exists():
                raise PilotExecutionBlocked('Prior unfinished confirmation attempt: no retry or duplicate launch')
            save_json(config_path,config)
            outcome=self.run(owner,workload,'confirmation')(config,raw_path,directory/(label+'.log'),config_path,C/workload/'runtime_context.json')
            if not raw_path.exists():raise PilotExecutionBlocked('No confirmation raw; preserve external interruption')
            catalog=__import__('module8c_final_real_search.common',fromlist=['inputs']).inputs()[0]
            rep=summarize(read(raw_path),Workload(identity['workload']),catalog,outcome,raw_path)
            rep.update(measurement_source='newly_measured',original_source_path=str(raw_path),
                       repetition=index,random_seed=seed,post_search_confirmation=True)
            save_json(result_path,rep)
            if rep['valid_runtime_repetition']:validate_raw(raw_path,identity,seed,workload)
        if not rep['valid_runtime_repetition']:raise PilotExecutionBlocked(rep['run_status'])
        self.refresh();return rep


class SharedEvidenceEvaluator(ReplicaRuntimeEvaluator):
    """The original evaluator, with lazy imports of exactly identical completed entries."""
    def __init__(self,directory,catalog,context_path,settings,seeds,max_unique,bank,owner,workload):
        self.bank,self.owner,self.workload_name=bank,owner,workload
        super().__init__(directory,catalog,context_path,settings,seeds,max_unique,
            run=bank.run(owner,workload,'search'))
        self.journal_path=Path(directory).parent/'logical_evaluations.json'
        self.journal=read(self.journal_path)['evaluations'] if self.journal_path.exists() else []

    def evaluate(self,deployment,workload,routing):
        identity=self.identity(deployment,workload,routing);key=digest(identity)
        self.bank.refresh()
        donor=self.bank.search_donor(identity,self.workload_name)
        imported=False
        if donor:
            with self.cache.locked() as cache:
                cache_key,entry=cache.reserve(identity)
                if entry['status']!='complete':
                    cache.data['configurations'][cache_key]=copy.deepcopy(donor['entry']);cache.flush();imported=True
        result=super().evaluate(deployment,workload,routing)
        source='reused' if donor else 'newly_measured'
        if donor:
            for i,rep in enumerate(result.repetitions,1):
                proof=validate_raw(rep['raw_result_path'],identity,self.seeds[i-1],self.workload_name)
                self.bank.log_reuse(self.owner,'search',i,proof)
        else:
            for i,rep in enumerate(result.repetitions,1):
                validate_raw(rep['raw_result_path'],identity,self.seeds[i-1],self.workload_name)
        self.journal.append({'logical_index':len(self.journal)+1,'cache_key':key,'identity':identity,
            'measurement_source':source,'lazy_cache_import':imported,
            'raw_sources':[ref(r['raw_result_path']) for r in result.repetitions],
            'runtime':result.__dict__})
        save_json(self.journal_path,{'evaluations':self.journal,'logical_count':len(self.journal),
            'unique_state_count':len({r['cache_key'] for r in self.journal}),
            'cached_states_count_against_search_budget':True})
        self.bank.refresh()
        print(__import__('json').dumps({'event':'logical_state_complete','owner':self.owner,'logical_count':len(self.journal),
            'measurement_source':source,'stability':result.stability_state_v2,'mean_ms':result.weighted_mean_e2e}),flush=True)
        return result
