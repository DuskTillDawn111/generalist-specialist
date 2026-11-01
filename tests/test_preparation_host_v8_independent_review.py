"""Independent arithmetic/boundary review of frozen eight-cell host hypothesis.

The separate audit runs full original-evidence verification. Constructor tests
patch that expensive verifier only after loading the frozen model; they assess
provider dispatch and refusal independently, not provenance qualification.
"""
import copy
import json
import math
from pathlib import Path
import sqlite3
import statistics
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import dag_preparation_host_release_v8 as host
from tools.predictor_validation_20261027.dag_host_release_v5 import structure, training_critical_path
from tools.predictor_validation_20261027.trace_export import linux_pid

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'artifacts/predictor-preparation-host-supply-v8-20261028-training01/model.json'


class PreparationHostIndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=json.loads(MODEL.read_text());cls.dag=json.loads(Path(cls.model['dag']['path']).read_text())
        cls.plans={(p['alias'],p['task'],p['pool_id']):p for p in cls.dag['profiles']}
        cls.bindings={(b['alias'],b['task'],b['pool_id']):b for b in cls.model['public_bindings']}
        cls.aliases={b['alias']:dict(family=b['family']) for b in cls.bindings.values()}
        parameters=json.loads(Path(cls.model['parameters']['path']).read_text())
        cls.captures={}
        for reference in parameters['solo_light_reviews']:
            review=json.loads(Path(reference['path']).read_text());raw=json.loads(Path(review['raw']['path']).read_text())
            for alias,owner in review['owners'].items():
                parsed=json.loads(Path(owner['parsed']['path']).read_text())
                with sqlite3.connect(Path(owner['sqlite']['path']).as_uri()+'?mode=ro',uri=True) as connection:
                    connection.row_factory=sqlite3.Row
                    runtime=[dict(r) for r in connection.execute('select r.*,s.value as api_name from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id')]
                for row in raw['request_records']:
                    if row['model']!=alias:continue
                    phases=[p for p in parsed['phases'] if p['request_id']==row['request_id']]
                    pid=phases[0]['pid'];offset=parsed['clock_mappings'][str(pid)]['offset_ns']
                    apis=sorted([a for a in runtime if linux_pid(a['globalTid'])==pid and any(p['nsys_start_ns']<=a['start']<=p['nsys_end_ns'] for p in phases)],key=lambda a:(a['start'],a['end'],a['correlationId']))
                    semantic=row['worker_response'].get('semantic_forward_observations',{}).get('records',[])
                    first=next(p['nsys_start_ns'] for p in phases if p['segment_id']==semantic[0]['id']) if semantic else None
                    key=alias,row['task'],row['pool_id']
                    cls.captures.setdefault(key,[]).append(dict(apis=apis,handler=row['worker_response']['start_perf_ns']-offset,semantic=first,sqlite=owner['sqlite']))

    def provider(self,**updates):
        args=dict(document=self.model,context=self.model['execution_context_sha256'],plans=self.plans,
            aliases=self.aliases,scenario=host.SCENARIO,allow_preparation_alignment_hypothesis=True,
            public_bindings=self.bindings);args.update(updates)
        with patch.object(host,'verify_model',return_value=self.model):return host.DagPreparationHostReleaseProvider(**args)

    def test_all_prefix_ordinals_offsets_and_two_capture_reference_domains(self):
        self.assertEqual(len(self.model['groups']),8)
        for group in self.model['groups']:
            key=group['alias'],group['task'],group['pool_id'];plan=self.plans[key]
            captures=self.captures[key];self.assertEqual(len(captures),2)
            source=next(c for c in captures if c['sqlite']==plan['source_sqlite'])
            signatures=[(a['start'],a['end'],a['api_name']) for a in source['apis']]
            self.assertEqual(len(signatures),len(set(signatures)))
            source_ord={signature:i for i,signature in enumerate(signatures)}
            nodes={n['id']:n for n in plan['nodes']}
            for node_id,evidence in group['prefix_offset_evidence'].items():
                node=nodes[node_id];ordinal=source_ord[node['issue_start_ns'],node['issue_end_ns'],node['api_name']]
                self.assertEqual(ordinal,evidence['source_runtime_ordinal'])
                use_handler=group['task']=='ocr' or node_id in group['preactivity_ids']
                offsets=[c['apis'][ordinal]['start']-c['handler' if use_handler else 'semantic'] for c in captures]
                self.assertEqual(offsets,evidence['reference_offsets_ns'])
                self.assertEqual(evidence['reference'],'handler_start' if use_handler else 'first_semantic_NVTX_start')
                for observation in group['observations']:
                    release=statistics.mean(offsets)+(0 if use_handler else observation['first_semantic_start_offset_ns'])
                    self.assertEqual(release,observation['prefix_release_ns'][node_id])

    def test_budget_subtracts_anchor_once_and_complete_DAG_bound(self):
        for group in self.model['groups']:
            key=group['alias'],group['task'],group['pool_id'];plan=self.plans[key]
            features=structure(plan);_,tails=training_critical_path(plan,features)
            for obs in group['observations']:
                anchor=obs['prefix_release_ns'][group['anchor_id']]
                self.assertEqual(anchor,obs['anchor_release_ns'])
                candidates=[(obs['completion_upper_bound_offset_ns']-anchor-tails[n])/(depth-group['anchor_depth'])
                            for n,depth in features['issue_depth'].items() if depth>group['anchor_depth']]
                step=min(candidates)
                self.assertEqual(step,obs['post_preparation_ns_per_step'])
                releases={n:(obs['prefix_release_ns'][n] if n in obs['prefix_release_ns'] else anchor+(depth-group['anchor_depth'])*step)
                          for n,depth in features['issue_depth'].items() if n!=features['terminal']}
                self.assertTrue(all(math.isfinite(r) and r>=0 for r in releases.values()))
                self.assertLessEqual(max(releases[n]+tails[n] for n in releases),obs['completion_upper_bound_offset_ns']+1e-6)

    def test_provider_all_nodes_uses_prefix_or_shifted_depth_and_terminal_tail_once(self):
        provider=self.provider()
        for group in self.model['groups']:
            key=group['alias'],group['task'],group['pool_id'];features=structure(self.plans[key]);par=group['joint_mean']
            row=dict(zip(('alias','task','pool_id'),key))
            for node in self.plans[key]['nodes']:
                actual=provider(node,row)
                if node['id']==features['terminal']:
                    self.assertEqual(actual,dict(not_before_offset_ns=0,after_predecessors_ns=round(par['terminal_tail_ns'])))
                else:
                    expected=group['prefix_release_ns'].get(node['id'],par['anchor_release_ns']+(features['issue_depth'][node['id']]-group['anchor_depth'])*par['post_preparation_ns_per_step'])
                    self.assertEqual(actual,dict(not_before_offset_ns=round(expected),after_predecessors_ns=0))

    def test_unknown_cell_node_and_changed_public_binding_rejected(self):
        provider=self.provider();key=next(iter(self.plans));row=dict(zip(('alias','task','pool_id'),key))
        with self.assertRaises(UnknownEvidence):provider(dict(id='foreign'),row)
        with self.assertRaises(UnknownEvidence):provider(self.plans[key]['nodes'][0],dict(row,pool_id='unseen'))
        changed=copy.deepcopy(self.bindings);changed[key]['input_content_sha256']='changed'
        with self.assertRaises(ValueError):self.provider(public_bindings=changed)
        with self.assertRaises(ValueError):self.provider(allow_preparation_alignment_hypothesis=False)
        with self.assertRaises(ValueError):self.provider(context='other')

    def test_changed_DAG_structure_is_rejected_before_cached_dispatch(self):
        plans=copy.deepcopy(self.plans);key=next(iter(plans));row=dict(zip(('alias','task','pool_id'),key))
        plans[key]['nodes'][0]['id']='changed-node'
        provider=self.provider(plans=plans)
        with self.assertRaises(ValueError):provider(plans[key]['nodes'][0],row)


if __name__=='__main__':unittest.main()
