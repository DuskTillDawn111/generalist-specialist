"""CPU-only immutable metadata/roster check. No GPU query or subprocess."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
ORIGINAL=ROOT/'artifacts/static-study-gpu-20261022/execution-packet-v2.json'
PROTOCOL='missing_standalone_full_resident_baseline_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def validate_packet(packet):
    if (packet.get('schema')!='gsdeploy.missing_state_baseline_packet.v1' or packet.get('protocol')!=PROTOCOL
        or packet.get('diagnostic_only') is not True or packet.get('U') is not None
        or packet.get('GPU_started') is not False or packet.get('completion_claim') is not False
        or packet.get('current_device_verified') is not False):raise ValueError('Diagnostic/preparation contract differs')
    if (packet.get('gpu_uuid')!='GPU-8a262fe0-598c-46b7-c953-41d9da84a47e' or packet.get('device')!='0'
        or packet.get('gpu_memory_mib')!=49140 or packet.get('safe_factor')!=1.110854):raise ValueError('Target/resource contract differs')
    bindings=packet.get('bindings',{})
    if not bindings or digest(bindings)!=packet.get('bindings_sha256'):raise ValueError('Binding-map SHA differs')
    for filename,expected in bindings.items():
        if not Path(filename).is_absolute() or sha(filename)!=expected:raise ValueError('Binding changed: '+filename)
    for key in ('system_config','samples','workers','admission'):
        if str(Path(packet[key]).resolve()) not in bindings:raise ValueError('Unbound input: '+key)
    source=Path(packet['frozen_source']).resolve()
    for directory in ('gsdeploy','scripts'):
        files=list((source/directory).rglob('*.py'))
        if not files or any(str(p.resolve()) not in bindings for p in files):raise ValueError('Frozen sources incompletely bound')
    for name in ('prepare.py','check.py','run.py'):
        if str(Path(__file__).with_name(name).resolve()) not in bindings:raise ValueError('Tool source unbound: '+name)
    if str(ORIGINAL) not in bindings:raise ValueError('Original domain source unbound')
    original=json.loads(ORIGINAL.read_text());config=json.loads(Path(packet['system_config']).read_text())
    before=copy.deepcopy(original['config']);after=copy.deepcopy(config)
    # Identity binding is the only catalog change; no service/quality/resource
    # replacement, domain pruning or fallback service for UNKNOWN paths.
    for value in (before,after):
        value.pop('random_seed',None);value.pop('horizon_seconds',None)
        value['catalog'].pop('measurement_context',None)
    if before!=after:raise ValueError('Original seven-model mathematical domain/config changed')
    if config['horizon_seconds']!=600 or config['random_seed']!=0:raise ValueError('Identity-only binding limits differ')
    if config['catalog']['measurement_context']['gpu_uuid']!=packet['gpu_uuid']:raise ValueError('Config target differs')
    if json.loads(Path(packet['workers']).read_text())!=original['workers']:raise ValueError('Original worker registry changed')
    preparation=Path(__file__).with_name('prepare.py')
    spec=importlib.util.spec_from_file_location('_seven_baseline_prepare_check',preparation)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if source!=module.SOURCE:raise ValueError('Frozen source root differs')
    expected_jobs,expected_pools=module.make_jobs(config,packet['samples'])
    if packet['pools']!=expected_pools or packet['jobs']!=expected_jobs:raise ValueError('Context/roster/limit identity differs')
    declaration_path=Path(packet['system_config']).parent/'discovery-protocol.json'
    if str(declaration_path) not in bindings:raise ValueError('Declaration unbound')
    declaration=json.loads(declaration_path.read_text())
    if not all(declaration.get(k) is True for k in ('prepared_only','no_cross_context_pruning','no_auto_retry','no_auto_next_context')):
        raise ValueError('Stop/no-pruning declaration differs')
    if declaration.get('U') is not None or declaration.get('GPU_started') is not False:raise ValueError('Preparation claims execution/utility')
    return dict(valid=True,contexts=5,path_states=9,global_primary_requests=sum(p['global_primary_count'] for j in expected_jobs for p in j['phases']),
        priming_requests=sum(len(j['priming_requests']) for j in expected_jobs),protocol=PROTOCOL,
        original_seven_model_domain_preserved=True,current_device_verified=False,GPU_started=False,U=None)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--packet',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(validate_packet(json.loads(args.packet.read_text()))))


if __name__=='__main__':main()
