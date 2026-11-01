"""Derive a new composite model identity for an explicit pooled handoff model.

Original execution IDs, the old contract and original component artifacts stay
unchanged. The new contract binds the new lifecycle artifact and pooling
assumption rather than pretending it is the old exact-input lifecycle model.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from gsdeploy.resources import identity
from .composite_execution_v2 import SCHEMA as COMPOSITE_SCHEMA,verify_target_runtime
from .native_collect import receipt,save
from .pooled_lifecycle_v4 import SCHEMA as POOLED_SCHEMA,PooledLifecycleProvider


def derive(base_contract_path,pooled_model_path,output,*,allow_cpu_fixture=False):
    base_ref=receipt(base_contract_path);model_ref=receipt(pooled_model_path)
    base=json.loads(Path(base_contract_path).read_text());model=json.loads(Path(pooled_model_path).read_text())
    if (base.get('schema')!=COMPOSITE_SCHEMA or identity({k:v for k,v in base.items()
            if k!='composite_execution_sha256'})!=base.get('composite_execution_sha256')):
        raise ValueError('Unchanged original composite contract required')
    original=base['components']['lifecycle']['original_context_sha256']
    if model.get('schema')!=POOLED_SCHEMA or model.get('execution_context_sha256')!=original:
        raise ValueError('Pooled lifecycle must retain original component execution identity')
    if not model.get('sources') or any(receipt(ref['path'])!=ref for ref in model['sources']):
        raise ValueError('Unchanged pooled lifecycle training and implementation lineage required')
    PooledLifecycleProvider(model,original,allow_cpu_fixture=allow_cpu_fixture,allow_pooling_hypothesis=True)
    if not allow_cpu_fixture:verify_target_runtime(base,base['target']['context']['path'])
    result=copy.deepcopy(base);result.pop('composite_execution_sha256')
    result['components']['lifecycle']['model']=model_ref
    result['components']['lifecycle']['adaptation']='training_family_task_joint_mean_pooling_v4'
    result['lifecycle_model_adaptation']=dict(base_contract=base_ref,
        original_composite_execution_sha256=base['composite_execution_sha256'],
        original_lifecycle_artifact=base['components']['lifecycle']['model'],
        pooled_model=model_ref,allow_pooling_hypothesis_required=True,
        pooling_assumption=model['pooling_assumption'],
        unvalidated_dimensions=model['unvalidated_dimensions'],
        original_component_execution_identity_retained=True,target_collector_execution_identity_retained=True,
        sources=[receipt(__file__),receipt(Path(__file__).with_name('pooled_lifecycle_v4.py'))])
    result['composite_execution_sha256']=identity(result)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);save(output/'contract.json',result)
    if receipt(base_contract_path)!=base_ref or receipt(pooled_model_path)!=model_ref:
        raise ValueError('Composite inputs changed during derivation')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-contract',required=True);parser.add_argument('--pooled-model',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();result=derive(args.base_contract,args.pooled_model,args.output)
    print(json.dumps(dict(composite_execution_sha256=result['composite_execution_sha256'],
                          original_target_execution_sha256=result['target']['execution_context_sha256'])))


if __name__=='__main__':main()
