from copy import deepcopy
import unittest

from tools.queue_aware_20261023.protocol import digest
from tools.queue_aware_20261023.workflow import select_validation_jobs,resources_valid,EXPECTED_GPU


class WorkflowTests(unittest.TestCase):
    def fixtures(self):
        jobs=[];rank=[]
        for strategy,score in [('queue_quality',-.5),('queue_latency',-.501),('queue_specialist_only',-.6),('no_split_swrr',-.7)]:
            contract=dict(deployment={'text':2,'generalist_2':1,'ocr':1},strategy=strategy)
            candidate=dict(deployment=contract['deployment'],strategy=strategy,policy_contract_sha256=digest(contract))
            identity=dict(seed=113,policy_contract_sha256=digest(contract),measurement_claim='development')
            jobs.append(dict(id=digest(identity),label=f'{strategy}-seed113',strategy=strategy,seed=113,identity=identity,policy_contract=contract))
            rank.append(dict(candidate_id=digest(candidate),candidate=candidate,complete_eligible_all_declared_seeds=True,mean_U=score))
        return dict(jobs=jobs,seeds=[113,114]),dict(ranking=rank)

    def test_freezes_strong_control_and_all_reserved_seeds_without_reselection(self):
        packet,report=self.fixtures();before=deepcopy(packet)
        validation=select_validation_jobs(packet,report,[81,82,71,72,73])
        self.assertEqual(len(validation['jobs']),10)
        self.assertEqual([j['strategy'] for j in validation['jobs'][:4]],
            ['queue_quality','queue_specialist_only','queue_specialist_only','queue_quality'])
        self.assertTrue(validation['selection']['no_reselection_on_validation'])
        self.assertEqual(validation['jobs'][0]['identity']['measurement_claim'],'INDEPENDENT_FIXED_SELECTION_NATIVE_VALIDATION')
        self.assertEqual(validation['jobs'][1]['identity']['development_rank'],3)
        self.assertEqual(validation['jobs'][-1]['phase'],'final_confirmation')
        self.assertEqual(packet,before)
        self.assertEqual(len({j['id'] for j in validation['jobs']}),10)

    def test_partial_development_never_selected(self):
        packet,report=self.fixtures();report['ranking'][0]['complete_eligible_all_declared_seeds']=False
        validation=select_validation_jobs(packet,report,[81])
        self.assertEqual(validation['jobs'][0]['strategy'],'queue_latency')
        for row in report['ranking']:row['complete_eligible_all_declared_seeds']=False
        with self.assertRaisesRegex(ValueError,'Fewer'):
            select_validation_jobs(packet,report,[81])

    def test_resource_failure_stops_instead_of_using_diagnostic_utility(self):
        raw=dict(status='completed',preflight=dict(gpu=dict(uuid=EXPECTED_GPU)),
            memory_evidence=dict(measured=True,resource_valid=True),energy_evidence=dict(coverage_valid=True))
        resources_valid(raw)
        raw['energy_evidence']['coverage_valid']=False
        with self.assertRaisesRegex(RuntimeError,'Energy'):
            resources_valid(raw)


if __name__=='__main__':unittest.main()
