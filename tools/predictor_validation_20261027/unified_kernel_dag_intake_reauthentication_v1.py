"""CPU reauthentication of a published unified intake against native SQLite.

An intake report's booleans and caller-rehashed DAG are declarations, not a
native dependency proof. Rebuild with the unchanged reviewed intake, using
explicit pinned native review/device/occupancy inputs. Temporary output paths
are irrelevant; complete exported contents and input source receipts must match.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from . import unified_kernel_dag_intake_v1 as native_intake
from .kernel_sequence_audit_v2 import verify
from .native_collect import receipt

SCHEMA='gsdeploy.unified_kernel_dag_intake_reauthentication.v1'


def need(value,reason):
    if not value:raise ValueError(reason)


def read(ref):
    verify(ref);return json.loads(Path(ref['path']).read_text())


def authenticate(entry,native_review_ref):
    declared=read(entry['intake_report'])
    need(declared.get('schema')==native_intake.SCHEMA
         and declared.get('validation_labels_read') is False
         and declared.get('status')=='COMPLETE_NATIVE_MAPPING_PATH_DIAGNOSTICS_NO_FIT',
         'Explicit training-only intake role required before native reconstruction')
    need(declared.get('review')==native_review_ref,'Exact native review binding required')
    need('device_attributes' in entry and 'occupancy_audit' in entry,
         'Explicit pinned device attributes and exact-source occupancy audit required')
    for key in ('device_attributes','occupancy_audit'):
        verify(entry[key]);need(entry[key] in declared['sources'],'Declared intake lacks exact '+key+' source receipt')
    verify(native_review_ref)
    for ref in declared['sources']:verify(ref)
    need(set(declared['artifacts'])=={'catalog.json','capture.json','paths.json'},
         'Complete native intake export cohort required')
    catalog=read(declared['artifacts']['catalog.json']);capture=read(declared['artifacts']['capture.json'])
    need(catalog.get('training_only') is True and catalog.get('validation_labels_read') is False
         and capture.get('split')=='training' and capture.get('validation_labels_read') is False,
         'Explicit catalog/capture training roles required before native reconstruction')
    with tempfile.TemporaryDirectory(prefix='kernel-dag-native-reauth-') as temporary:
        native_intake.intake(native_review_ref['path'],entry['device_attributes']['path'],
            Path(temporary)/'rebuilt',occupancy_audit=entry['occupancy_audit']['path'])
        # Compare the actual published JSON on both sides. The in-memory return
        # contains tuples (stream keys) that JSON correctly serializes as arrays.
        rebuilt=read(receipt(Path(temporary)/'rebuilt/report.json'))
        # The only output-location-dependent report field is the artifact map.
        # Every input source and all structural/native diagnostic fields match.
        need({k:v for k,v in rebuilt.items() if k!='artifacts'}==
             {k:v for k,v in declared.items() if k!='artifacts'},
             'Fresh native intake report/input sources differ from declared intake')
        new_catalog=read(rebuilt['artifacts']['catalog.json'])
        new_capture=read(rebuilt['artifacts']['capture.json'])
        need(new_catalog==catalog,'Fresh SQLite-reconstructed complete native DAG catalog differs')
        need(new_capture==capture,'Fresh SQLite-reconstructed complete native capture differs')
        need(read(rebuilt['artifacts']['paths.json'])==read(declared['artifacts']['paths.json']),
             'Fresh native active-set paths differ')
        # Native evidence fields inside these objects point to pinned original
        # sources, never to temporary exports. Return freshly rebuilt objects.
        return dict(report=declared,catalog=new_catalog,capture=new_capture,
            reauthentication=dict(schema=SCHEMA,native_sqlite_reparsed=True,
                complete_catalog_capture_paths_equal=True,input_sources_equal=True,
                temporary_export_paths_ignored_only=True,GPU_started=False,fit_started=False))
