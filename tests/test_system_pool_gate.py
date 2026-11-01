import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from gsdeploy.profiling.compact_experiments import verify_system_pool_admission
from gsdeploy.profiling.prepare import digest

class SystemPoolGateTests(unittest.TestCase):
 def test_reduced_system_manifest_checks_parent_full_memory_and_independent_costs(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);parent_path=root/'parent.json'
   workers={'g':{'spec':'pinned-worker'}}
   parent={'workers':workers,'sources':{},'code_hashes':{},'memory_validation':True,'jobs':[{'phase':'memory'}]}
   parent_path.write_text(json.dumps(parent))
   manifest={'workers':workers,'jobs':[{'phase':'colocation'}],'admission_evidence':{'manifest':str(parent_path),'manifest_sha256':digest(parent_path),'results':str(root/'results')}}
   with patch('gsdeploy.profiling.compact_experiments.verify_memory_screen') as memory, \
        patch('gsdeploy.profiling.compact_experiments.calibration_quality_screen',return_value={'passed':True}) as calibration, \
        patch('gsdeploy.profiling.compact_experiments.assess',return_value={'all_requirements_passed':True}) as assess:
    verify_system_pool_admission(manifest,root/'window')
    memory.assert_called_once_with(parent_path,parent,root/'results')
    calibration.assert_called_once_with(parent_path,parent,root/'results')
    assess.assert_called_once()
    assess.return_value={'all_requirements_passed':False}
    with self.assertRaisesRegex(ValueError,'Independent quality'):verify_system_pool_admission(manifest,root/'window')
    assess.return_value={'all_requirements_passed':True}
    calibration.return_value={'passed':False}
    with self.assertRaisesRegex(ValueError,'calibration quality'):verify_system_pool_admission(manifest,root/'window')
   changed=copy.deepcopy(manifest);changed['workers']['g']['spec']='different'
   with self.assertRaisesRegex(ValueError,'System workers differ'):verify_system_pool_admission(changed,root/'window')
   parent_path.write_text('{}')
   with self.assertRaisesRegex(ValueError,'Changed system pool'):verify_system_pool_admission(manifest,root/'window')
 def test_missing_pool_evidence_is_rejected_before_gpu(self):
  with self.assertRaisesRegex(ValueError,'complete pool evidence'):
   verify_system_pool_admission({'workers':{}},Path('never-created'))
