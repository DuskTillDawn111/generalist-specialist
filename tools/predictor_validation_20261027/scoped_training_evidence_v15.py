"""Exact operational evidence closure with declared failed-history boundaries.

An explicitly qualified training context's prior_failed_evidence and separately
reviewed exact preparation-inventory edges can be opaque. Their target bytes
are verified and materialized. Any normal reachability upgrades the target to
full traversal. No historical receipt is changed or missing bytes invented.
"""
from __future__ import annotations

import json
import copy
from pathlib import Path

from .complete_training_index_v12 import located_references
from .historical_evidence_v4 import HistoricalReceiptResolver, _reference
from .native_collect import receipt
from .profiling import digest

SCHEMA='gsdeploy.scoped_operational_training_evidence_closure.v15'
BOUNDARY_REASON='EXPLICIT_STRICT_TRAINING_CONTEXT_PRIOR_FAILED_EVIDENCE_HISTORICAL_AUDIT_ONLY'
METADATA_REASON='EXACT_AUDITED_ORIGINAL_PREPARATION_SOURCE_INVENTORY_NOT_AN_OPERATIONAL_DEPENDENCY'


def operational_metadata_boundary_policy():
    """The one independently reviewed preparation-audit edge, never a glob."""
    base=Path(__file__).resolve().parents[2]/'artifacts/predictor-corun-training-evidence-plan-v8-20261028-final01'
    body=dict(schema='gsdeploy.explicit_operational_metadata_boundary_policy.v15',
        implementation=receipt(__file__),boundaries=[dict(
            parent=dict(path=str(base/'prepared-repeat.json'),sha256='2a0e88c09dbc80ac8ccfee96350838dd544236607fe08948c63801ad3f681e2d'),
            jsonpath=['source_before'],
            ref=dict(path=str(base/'source-before.json'),sha256='2fb58b4158870fdedb669c831f5de5c032d28adc848fe3253ffea87448257cd1'),
            reason=METADATA_REASON)])
    return dict(body,policy_sha256=digest(body))


def training_json_read_guard(training_raw_references,*,inherited_smoke_references=()):
    """Permit parsed raw only from audited training or explicit smoke provenance.

    This is a caller-selected read guard, not a recursive boundary. A normal
    nonindexed raw is rejected even if an opaque historical edge also names it.
    """
    allowed=set()
    for reference in [*training_raw_references,*inherited_smoke_references]:
        if not _reference(reference) or Path(reference['path']).name!='raw.json':
            raise ValueError('Exact audited raw.json receipts required for JSON read guard')
        allowed.add((reference['path'],reference['sha256']))
    def guard(reference):
        path=Path(reference['path']);name=path.name.lower();lower=str(path).lower()
        if name=='raw.json' and (reference['path'],reference['sha256']) not in allowed:
            raise ValueError('Normal nonindexed raw JSON parse refused before opening: '+str(path))
        if (name in ('observed.json','observed-metrics.json','predictions.json')
                or ('validation' in lower and any(t in name for t in ('labels','observed','outcomes')))):
            raise ValueError('Validation outcome JSON parse refused before opening: '+str(path))
    return guard


def resolve_closure(paths,manifest_path=None,*,training_context_references=(),before_json_read=None,
                    metadata_boundary_policy=None):
    """Resolve sources exactly; do not parse declared failed-history targets.

    ``training_context_references`` must come from the caller's independently
    verified strict training captures. A model merely spelling the same field
    does not qualify. ``before_json_read`` is an optional outcome-read guard;
    it runs before every normal JSON parse, including an opaque target upgrade.
    It is not invoked for byte verification of opaque audit targets.
    """
    if before_json_read is not None and not callable(before_json_read):
        raise ValueError('Callable pre-JSON outcome-read guard required')
    eligible=set()
    for reference in training_context_references:
        if not _reference(reference) or Path(reference['path']).name!='context.json':
            raise ValueError('Explicit exact strict-training context.json receipts required')
        eligible.add((reference['path'],reference['sha256']))
    authorized_metadata={}
    if metadata_boundary_policy is not None:
        policy=copy.deepcopy(metadata_boundary_policy)
        if (policy.get('schema')!='gsdeploy.explicit_operational_metadata_boundary_policy.v15'
                or policy.get('policy_sha256')!=digest({k:v for k,v in policy.items() if k!='policy_sha256'})
                or not _reference(policy.get('implementation'))
                or receipt(policy['implementation']['path'])!=policy['implementation']):
            raise ValueError('Source-bound exact operational metadata boundary policy required')
        for edge in policy['boundaries']:
            if (set(edge)!={'parent','jsonpath','ref','reason'} or not _reference(edge['parent'])
                    or not _reference(edge['ref']) or edge['reason']!=METADATA_REASON
                    or edge['jsonpath']!=['source_before']
                    or Path(edge['parent']['path']).name!='prepared-repeat.json'
                    or Path(edge['ref']['path']).name!='source-before.json'):
                raise ValueError('Exact reviewed preparation source-inventory boundary required')
            key=(edge['parent']['path'],edge['parent']['sha256'],tuple(edge['jsonpath']),edge['ref']['path'],edge['ref']['sha256'])
            if key in authorized_metadata:raise ValueError('Repeated exact metadata boundary')
            authorized_metadata[key]=edge
    else:policy=None
    resolver=HistoricalReceiptResolver(manifest_path)
    # The manifest is itself a normal metadata root. Its snapshot field can
    # arrive before any model edge naming the historical identity. Preserve
    # the historical read restriction before that first snapshot parse too.
    alias_parents={}
    for (old_path,old_sha),snapshot in resolver.mapping.items():
        alias_parents.setdefault((snapshot['path'],snapshot['sha256']),[]).append(
            dict(path=old_path,sha256=old_sha))
    def guarded_aliases(reference):
        pending_aliases=[reference];seen_aliases=set();result=[]
        while pending_aliases:
            alias=pending_aliases.pop();alias_key=alias['path'],alias['sha256']
            if alias_key in seen_aliases:continue
            seen_aliases.add(alias_key);result.append(alias)
            pending_aliases.extend(alias_parents.get(alias_key,[]))
        return result
    roots=[receipt(path) for path in paths]
    pending=[(ref,None,None,None) for ref in roots]
    if resolver.manifest_receipt:pending.append((resolver.manifest_receipt,None,None,None))
    if policy:pending.append((policy['implementation'],None,None,None))
    materialized={};substitutions={};normal_seen=set();parsed=set();boundaries=[];boundary_seen=set()
    # Many model receipts recur through inherited source inventories. Verify an
    # exact original receipt once, then verify every materialized source again
    # at the end. This cache never marks a source parsed or suppresses upgrade.
    resolved_receipts={}
    first_incoming={}
    qualified_contexts=[]
    while pending:
        original,opaque_reason,parent,jsonpath=pending.pop();opaque=opaque_reason is not None
        key=original['path'],original['sha256']
        first_incoming.setdefault(key,dict(parent=parent,jsonpath=jsonpath,ref=original,opaque=opaque))
        if key not in resolved_receipts:
            try:resolved_receipts[key]=resolver.resolve(original)
            except (ValueError,OSError) as exc:
                chain=[];cursor=key;visited=set()
                while cursor in first_incoming and cursor not in visited:
                    visited.add(cursor);edge=first_incoming[cursor];chain.append(edge)
                    previous=edge['parent']
                    if previous is None:break
                    cursor=previous['path'],previous['sha256']
                exc.scoped_closure_diagnostic=dict(failing_reference=original,parent=parent,
                    jsonpath=jsonpath,opaque_edge=opaque,root_to_failure=list(reversed(chain)),
                    already_materialized_sources=len(materialized),boundaries=boundaries,
                    materialized_JSON_paths=[r['path'] for k,r in materialized.items() if k in parsed])
                raise
        resolved=resolved_receipts[key]
        resolved_key=resolved['path'],resolved['sha256']
        materialized[resolved_key]=resolved
        if original!=resolved:substitutions[key]=dict(historical=original,snapshot=resolved)
        if opaque:
            boundary_key=(parent['path'],parent['sha256'],tuple(jsonpath),*key)
            if boundary_key not in boundary_seen:
                boundaries.append(dict(ref=original,materialized_ref=resolved,parent=parent,
                    jsonpath=jsonpath,reason=opaque_reason,target_bytes_verified=True))
                boundary_seen.add(boundary_key)
            # A separate normal request still expands this source; opaque
            # processing never adds the target to normal_seen or parsed.
            continue
        # Read permission is origin-sensitive even when a snapshot has another
        # filename or this resolved source was already parsed via another edge.
        # Check both identities before normal_seen can deduplicate traversal.
        if before_json_read is not None and (Path(original['path']).suffix=='.json'
                or Path(resolved['path']).suffix=='.json'):
            guard_refs={}
            for identity in (original,resolved):
                for alias in guarded_aliases(identity):guard_refs[alias['path'],alias['sha256']]=alias
            for alias in guard_refs.values():before_json_read(dict(alias))
        if resolved_key in normal_seen:continue
        normal_seen.add(resolved_key)
        if Path(resolved['path']).suffix!='.json':continue
        value=json.loads(Path(resolved['path']).read_text());parsed.add(resolved_key)
        qualified=key in eligible
        if qualified:
            if (not isinstance(value,dict) or value.get('training_only') is not True
                    or value.get('split') not in (None,'training')
                    or not isinstance(value.get('execution_sources'),list)
                    or not isinstance(value.get('residents'),dict)):
                raise ValueError('Supplied boundary parent is not an explicit actual training context')
            if 'prior_failed_evidence' in value and not isinstance(value['prior_failed_evidence'],list):
                raise ValueError('Explicit top-level prior_failed_evidence list required')
            qualified_contexts.append(original)
        for location,child in located_references(value):
            historical=qualified and bool(location) and location[0]=='prior_failed_evidence'
            edge_key=(*key,tuple(location),child['path'],child['sha256'])
            metadata=edge_key in authorized_metadata
            if metadata and (value.get('schema')!='gsdeploy.corun_exact_repeat_preparation.v8'
                    or value.get('training_only') is not True
                    or value.get('status')!='CPU_PREPARED_SOURCE_RESOURCE_AUDIT_PASS'):
                raise ValueError('Exact metadata boundary parent does not have reviewed preparation-audit semantics')
            reason=BOUNDARY_REASON if historical else METADATA_REASON if metadata else None
            pending.append((child,reason,original,list(location)))
    for boundary in boundaries:
        resolved=boundary['materialized_ref'];key=resolved['path'],resolved['sha256']
        boundary['also_normally_reachable']=key in normal_seen
        boundary['effective_opaque_boundary']=key not in normal_seen
        boundary['normal_JSON_traversal_performed']=key in parsed
    refs=list(materialized.values())
    if any(receipt(ref['path'])!=ref for ref in refs):
        raise ValueError('Scoped operational evidence closure changed during traversal')
    return dict(schema=SCHEMA,status='OPERATIONAL_TRAINING_CLOSURE_VERIFIED_WITH_DECLARED_HISTORICAL_BOUNDARIES',
        roots=roots,materialized_sources=refs,historical_substitutions=list(substitutions.values()),
        resolution_manifest=resolver.manifest_receipt,qualified_training_contexts=qualified_contexts,
        metadata_boundary_policy=policy,
        boundaries=boundaries,effective_opaque_boundary_edges=sum(r['effective_opaque_boundary'] for r in boundaries),
        normal_JSON_sources=[materialized[key] for key in materialized if key in parsed],
        pre_JSON_read_guard_used=before_json_read is not None,
        operational_model_and_training_closure_complete_under_declared_historical_boundaries=True,
        full_historical_transitive_closure_verified=False,historical_artifacts_rewritten=False,
        current_execution_equivalence_claimed=False,historical_direct_path_closure_pass_claimed=False,
        scope='Exact operational model/training sources, with verified but unparsed prior_failed_evidence audit targets from explicitly qualified strict training contexts; not an exhaustive historical closure.')
