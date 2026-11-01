"""Freeze installed CPU-audited environments and native wheel provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1048576):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',required=True)
    p.add_argument('--audit-root',required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    assets=Path(a.assets).resolve();base=Path(a.audit_root).resolve()
    envfile=root/'configs/compact_environments.json';plan=json.loads(envfile.read_text())
    expected={'common':'4.48.2','standard':'4.51.3','qwen':'4.52.0.dev0','minicpm':'4.44.2','deepseek':'4.46.3'}
    installed={}
    for name,version in expected.items():
        path=base/('environment-'+name+'.json');data=json.loads(path.read_text())
        assert data['status']=='PASS_CPU_IMPORTS_AND_DEPENDENCIES',data
        assert data['installed_packages']['transformers']==version,data['installed_packages']['transformers']
        assert data['torch_cuda_build']=='12.4' and data['cxx11_abi'] is False
        assert not data['cuda_initialized'] and data['weights_loaded']==0
        lock=root/'configs/environments'/(name+'.lock')
        installed['compact-'+name]={'python':data['python'],'cpu_audit':str(path),
            'python_version':data['python_version'],'lock_hashes':{str(lock.relative_to(root)):sha(lock)},
            'gpu_kernels_tested':False}
    wheels={}
    for name,pattern in {'qwen-transformers':'transformers-4.52.0.dev0-*.whl',
                         'phi-flash':'flash_attn-2.7.4.post1+*.whl',
                         'deepseek-flash':'flash_attn-2.7.3+*.whl',
                         'minicpm-gptq':'auto_gptq-0.8.0.dev0+cu124-*.whl'}.items():
        paths=list((assets/'envs/wheels').glob(pattern));assert len(paths)==1,(name,paths)
        path=paths[0]
        with zipfile.ZipFile(path) as archive:assert archive.testzip() is None
        wheels[name]={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
    native={'schema':'gsdeploy.native_dependencies.v1','state':'CPU_ABI_IMPORTED_GPU_KERNELS_UNVALIDATED',
        'python_abi':'cp310','platform':'linux_x86_64','torch':'2.6.0+cu124','cuda':'12.4','cxx11_abi':False,
        'wheels':wheels,'source_pins':{
            'transformers':'https://github.com/huggingface/transformers/tree/cb39f7dd5ba874ee1859b47283b08cd3a6ab5a0d',
            'auto-gptq':'https://github.com/RanchiZhao/AutoGPTQ/tree/a9c8109ef450793e3d890c76a36f22177fcbbe28'},
        'source_archives':{str(p):sha(p) for p in [base/'qwen-source.zip',base/'gptq-source.zip']},
        'gptq_build':{'architecture':'SM89','compiler':'Clang14 + NVCC12.4.131 + CCCL12.4.127.post1',
            'jobs':1,'modules':['autogptq_cuda_64','autogptq_cuda_256','exllama_kernels'],
            'exllamav2':False,'marlin':False,'qigen':False,'kernel_math_patch':False,
            'header_and_dtype_dispatch_patch_sha256':sha(root/'configs/environments/autogptq-memory.patch'),
            'build_script_sha256':sha(root/'scripts/build_compact_gptq.py'),
            'source_patch_receipt':str(base/'gptq-build-patch.json'),
            'source_patch_receipt_sha256':sha(base/'gptq-build-patch.json')},
        'environment_layers':{'compact-common':['phi-flash'],'compact-standard':[],
            'compact-qwen':[],'compact-minicpm':['minicpm-gptq'],'compact-deepseek':['deepseek-flash']}}
    nativepath=root/'configs/environments/native-lock.json';nativepath.write_text(json.dumps(native,indent=2)+'\n')
    for entry in installed.values():entry['lock_hashes']['configs/environments/native-lock.json']=sha(nativepath)
    plan.update(state='CPU_INSTALLED_LOCKED_GPU_UNVALIDATED',environments_installed=True,
        installed_environments=installed,gpu_started=False,models_loaded=0)
    for entry in plan['models'].values():
        entry.update(state='CPU_DEPENDENCIES_AND_PREPROCESSING_IMPLEMENTED_GPU_UNVALIDATED',
            blocking_checks=['Native GPU loading and quantization/attention kernels',
                             'Complete outputs/EOS, measured quality/memory/latency/board energy',
                             'Feasible replicas/colocation and strict pool admission'])
        entry.pop('source_extra_packages',None)
        quant=entry['quantization'];quant.pop('additional_dependency',None);quant['status']='GPU_CONFIGURATION_UNVALIDATED'
    plan['notes']=['Complete transitive hash locks, pinned source/wheel layers and CPU import receipts.',
        'No candidate checkpoints loaded, no GPU initialization or kernel measurement.',
        'The native GPTQ source has a recorded headers/equivalent dtype-dispatch build patch; selected kernels are unchanged.',
        'Old experimental environments were not upgraded.']
    envfile.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    studyfile=root/'configs/study.json';study=json.loads(studyfile.read_text())
    study['task_adapter_protocol'].update(sha256=sha(root/'configs/compact_protocol.json'),environment_plan_sha256=sha(envfile))
    studyfile.write_text(json.dumps(study,ensure_ascii=False,indent=2)+'\n')
    print('ENVIRONMENTS_SEALED_CPU_ONLY',len(installed))


if __name__=='__main__':main()
