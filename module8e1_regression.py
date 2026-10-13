"""All old CPU tests in their existing environments, plus v2 tests."""
import json,subprocess,sys
from pathlib import Path
from deployment_search.runtime_cache import save_json, file_hash
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/module8e1_method_revision'
for name in ['regression_results.json','regression_execution.log','regression_progress.log']:
    p=OUT/name;q=OUT/('initial_'+name)
    if p.exists() and not q.exists():p.rename(q)
files=sorted(subprocess.check_output(['rg','--files','-g','test*.py','-g','tests.py','-g','!envs/**','-g','!models/**','-g','!datasets/**','-g','!results/**','-g','!method_v2/**','-g','!**/__pycache__/**'],cwd=ROOT,text=True).splitlines())
modules=[p[:-3].replace('/','.') for p in files]
needs_sklearn=['performance_model.generalization.test_gate','performance_model.local_calibration.test_local_calibration']
code='''import unittest,json,sys
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(m) for m in sys.argv[2:])
with open(sys.argv[1]+'.log','w') as stream:r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
d={'status':'PASS' if r.wasSuccessful() else 'FAIL','tests_run':r.testsRun,'failures':[(str(t),e) for t,e in r.failures],'errors':[(str(t),e) for t,e in r.errors],'skips':[(str(t),s) for t,s in r.skipped]}
open(sys.argv[1]+'.json','w').write(json.dumps(d,indent=2)+'\\n')
sys.exit(0 if r.wasSuccessful() else 1)
'''
parts=[]
for name,exe,mods in [('data_env',sys.executable,[m for m in modules if m not in needs_sklearn]+['method_v2.tests']),('gsdeploy','/root/miniconda3/envs/gsdeploy/bin/python',needs_sklearn)]:
    prefix=OUT/('regression_'+name)
    with (OUT/('regression_'+name+'_stdout.log')).open('w') as stream:p=subprocess.run([exe,'-c',code,str(prefix),*mods],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    result=json.loads(Path(str(prefix)+'.json').read_text());parts.append({'environment':exe,'modules':mods,'exit_code':p.returncode,**result})
result={'status':'PASS' if all(p['status']=='PASS' for p in parts) else 'FAIL','old_module_count':len(modules),'new_test_module':'method_v2.tests','tests_run':sum(p['tests_run'] for p in parts),'parts':parts,'CPU_unit_tests_only':True,'GPU_runs':0,'no_package_installation_or_environment_modification':True,'initial_attempt_errors':'Two historical sklearn test modules required existing gsdeploy environment; initial error evidence retained','v2_production_enumeration':0,'legacy_small_oracles':'Only unchanged bounded old unit-test fixtures'}
save_json(OUT/'regression_results.json',result);print(json.dumps({'status':result['status'],'tests_run':result['tests_run'],'old_modules':len(modules)},ensure_ascii=False),flush=True)
if result['status']!='PASS':raise SystemExit(1)
