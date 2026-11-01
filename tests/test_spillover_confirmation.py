"""Contract and native-entry CPU fixtures; no GPU or real measured claim."""
import copy
from dataclasses import replace
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.config import SystemConfig
from gsdeploy.quality import quality_digest
from gsdeploy.static_study import _file
from gsdeploy.spillover_confirmation import (CONTRACT_SCHEMA,ROLE_NAMES,prepare,validate,verify_run,run,report,seal,Runtime,_collection_receipt)
from tests.test_calibrated_replay_v2 import fitted_replay_fixture
from tests.test_confirmation import prepared

ROOT=Path(__file__).resolve().parents[1]


def fixture(root,runtime_root=ROOT):
    old,paths=prepared(root,config=fitted_replay_fixture())
    raw=json.loads(paths[1].read_text());raw["random_seed"]=51
    config=SystemConfig(**raw)
    contract=dict(schema=CONTRACT_SCHEMA,study_id="CPU_FIXTURE_NOT_MEASURED",workload=config.workload,
        horizon_seconds=config.horizon_seconds,warmup_seconds=30,seed_roles=dict(development=[51],validation=[61,62],confirmation=[71,72,73]),
        training_forbidden=True,purpose="spillover_final_not_training",required_roles=list(ROLE_NAMES),
        winner_label="free_search_winner",target_task=next(iter(config.workload)))
    plans=[old["candidates"][0]["plan"],old["candidates"][-1]["plan"]]
    candidates=[dict(label=ROLE_NAMES[0],plan=plans[0],status="OPTIMAL_FIXTURE",optimality_certified=True),
        dict(label=ROLE_NAMES[1],plan=None,status="INFEASIBLE",optimality_certified=True),
        dict(label=ROLE_NAMES[2],plan=None,status="INFEASIBLE",optimality_certified=True),
        dict(label=ROLE_NAMES[3],plan=plans[1],status="OPTIMAL_FIXTURE",optimality_certified=True),
        dict(label=ROLE_NAMES[4],plan=plans[1],status="OPTIMAL_FIXTURE",optimality_certified=True)]
    bundle=dict(schema="gsdeploy.spillover_confirmation_candidates.v1",config_sha256=quality_digest(config.as_dict()),
        study_contract_sha256=quality_digest(contract),candidates=candidates)
    files=[root/n for n in ("new-config.json","candidates.json","contract.json")]
    for p,v in zip(files,(config.as_dict(),bundle,contract)):p.write_text(json.dumps(v))
    packet=prepare(paths[0],*files[:2],files[2],runtime_root)
    return packet,paths[0],files


def native(packet,job):
    config=replace(SystemConfig(**packet["config"]),random_seed=job["seed"]).as_dict()
    config["catalog"].update(performance_mode="task_replica_proxy",memory_mode="standalone_proxy")
    config["catalog"]["measurement_context"]["runtime_checkpoint_sha256"]=job["identity"]["runtime_checkpoint_sha256"]
    identity=copy.deepcopy(job["identity"])
    return dict(status="completed",backend="native",experiment_job_id=job["id"],experiment_phase="confirmation",
        confirmation_packet_sha256=packet["confirmation_sha256"],executed_measurement_identity=identity,
        executed_measurement_id=quality_digest(identity),config=config,plan=job["plan"],
        priming_policy=identity["warmup_policy"],priming_size_metric=identity["warmup_size_metric"],
        preflight=dict(ready=True,gpu_checked=True,device="0",gpu=dict(uuid="CPU fixture GPU",memory_mib=32)),
        transport_config=dict(random_seed=job["seed"],measurement_seconds=identity["horizon_seconds"],
            arrival_process=identity["arrival_process"],warmup_seconds=identity["warmup_seconds"],drain_timeout_seconds=identity["drain_seconds"]),
        measurement_window=dict(duration_seconds=identity["horizon_seconds"],trace_sha256=identity["trace_sha256"]))


def write_native(path,packet,job):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(native(packet,job)))
    (path.parent/"spillover-collection-receipt.json").write_text(json.dumps({**_collection_receipt(packet),"run":_file(path)}))


def utility():
    return dict(eligible=True,energy_verified=True,U=-.25,**{t+"_norm":.25 for t in ("Q","T","L","P","M","E")})


class SpilloverConfirmationTests(unittest.TestCase):
    def gate(self,config,contract):
        return dict(version_sha256="CPU_FIXTURE_TRANSFER_VALIDATED",seed_roles=contract["seed_roles"])

    def test_prepare_deduplicates_custom_seed_roles_and_seals_purpose(self):
        with tempfile.TemporaryDirectory() as tmp,patch("gsdeploy.spillover_confirmation._gate",side_effect=self.gate):
            packet,_,_=fixture(Path(tmp))
            self.assertEqual(len(packet["candidates"]),2)
            self.assertEqual(len(packet["jobs"]),6)
            self.assertEqual({j["seed"] for j in packet["jobs"]},{71,72,73})
            self.assertTrue(validate(packet)["valid"])
            self.assertTrue(all(j["phase"]=="confirmation" and j["identity"]["confirmation_purpose"]=="spillover_final_not_training" for j in packet["jobs"]))
            changed=copy.deepcopy(packet);changed["jobs"][0]["purpose"]="calibration";changed["confirmation_sha256"]=seal(changed)
            self.assertFalse(validate(changed)["valid"])
            changed=copy.deepcopy(packet);changed["execution_checkpoint_contract"]["runtime_checkpoint_sha256"][next(iter(changed["config"]["catalog"]["models"]))]="different";changed["confirmation_sha256"]=seal(changed)
            self.assertFalse(validate(changed)["valid"])

    def test_real_transfer_gate_rejects_unvalidated_and_overlapping_seeds(self):
        from gsdeploy.replay_transfer import prepare_transfer
        from gsdeploy.spillover_confirmation import _gate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/"basis.json";source.write_text(json.dumps(fitted_replay_fixture().as_dict()))
            raw=prepare_transfer(source,{t:.5 for t in fitted_replay_fixture().workload})
            with self.assertRaisesRegex(ValueError,"TRANSFER_NOT_VALIDATED"):
                _gate(SystemConfig(**raw),{})
            from gsdeploy.spillover_confirmation import _gate as gate
            config=fitted_replay_fixture()
            spec=dict(seed_roles=dict(development=[51],validation=[61,62],confirmation=[51,72,73]))
            with patch("gsdeploy.replay_transfer.validated_transfer_spec",return_value=spec),patch("gsdeploy.replay_transfer.check_scoring"):
                with self.assertRaisesRegex(ValueError,"disjoint"):
                    gate(config,dict(schema=CONTRACT_SCHEMA,study_id="fixture",seed_roles=spec["seed_roles"]))

    def test_native_identity_transport_scoring_and_missing_utility_guards(self):
        with tempfile.TemporaryDirectory() as tmp,patch("gsdeploy.spillover_confirmation._gate",side_effect=self.gate):
            root=Path(tmp);packet,_,_=fixture(root);job=packet["jobs"][0];raw=native(packet,job);path=root/"run.json"
            write_native(path,packet,job)
            with patch("gsdeploy.evaluation.evaluate_run",return_value={"utility":utility()}):
                self.assertTrue(verify_run(packet,job,path)[1]["eligible"])
                for kind in ("purpose","seed","config","phase"):
                    changed=copy.deepcopy(raw)
                    if kind=="purpose":
                        changed["executed_measurement_identity"]["confirmation_purpose"]="development"
                        changed["executed_measurement_id"]=quality_digest(changed["executed_measurement_identity"])
                    if kind=="seed":changed["transport_config"]["random_seed"]=51
                    if kind=="config":changed["config"]["weights"]["Q"]+=.1
                    if kind=="phase":changed["experiment_phase"]="calibration"
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):verify_run(packet,job,path)
            path.write_text(json.dumps(raw))
            with patch("gsdeploy.evaluation.evaluate_run",return_value={"utility":dict(eligible=False,U=None,energy_verified=True)}):
                with self.assertRaisesRegex(ValueError,"never substitute zero"):verify_run(packet,job,path)

    def test_frozen_executor_receives_only_declared_collection_delta(self):
        with tempfile.TemporaryDirectory() as tmp,patch("gsdeploy.spillover_confirmation._gate",side_effect=self.gate):
            root=Path(tmp);packet,_,_=fixture(root);job=packet["jobs"][0];before=copy.deepcopy(packet);observed=[]
            class DummyRuntime:
                def __init__(self,*args):pass
                def __enter__(self):
                    class Study:
                        @staticmethod
                        def execute_job(collection,key,output,execute=False):
                            from gsdeploy.static_study import validate_packet
                            self_assert=validate_packet(collection)
                            if not self_assert["valid"]:raise ValueError(self_assert)
                            observed.append(copy.deepcopy(collection));Path(output).mkdir()
                            Path(output,"run.json").write_text(json.dumps(native(packet,job)))
                            return native(packet,job)
                    return Study
                def __exit__(self,*args):pass
            # validate uses a real frozen _job; replace Runtime only for the
            # execute call after the independent validation already passed.
            real=Runtime
            calls=[0]
            def select(root):
                calls[0]+=1
                return DummyRuntime(root) if calls[0]==2 else real(root)
            with patch("gsdeploy.spillover_confirmation.Runtime",side_effect=select),patch("gsdeploy.evaluation.evaluate_run",return_value={"utility":utility()}):
                run(packet,job["id"],root/"run",execute=True)
            self.assertEqual(packet,before)
            self.assertEqual(observed[0]["jobs"][0]["identity"],job["identity"])
            self.assertEqual(observed[0]["confirmation_sha256"],packet["confirmation_sha256"])
            self.assertEqual(observed[0]["config"]["catalog"]["performance_mode"],"task_replica_proxy")

    def test_real_snapshot_execute_entry_preserves_confirmation_identity(self):
        snapshot=ROOT/"artifacts/static-study-gpu-20261022/execution-source-v5"
        if not snapshot.is_dir():
            snapshot=ROOT
        with tempfile.TemporaryDirectory() as tmp,patch("gsdeploy.spillover_confirmation._gate",side_effect=self.gate):
            root=Path(tmp);packet,_,_=fixture(root,runtime_root=snapshot);job=packet["jobs"][0];output=root/"actual-entry"
            runtime=Runtime(snapshot)
            study=runtime.study
            # Actual unmodified frozen execute_job runs. Only checkpoint I/O,
            # GPU query and transport inference are explicit CPU fixtures.
            bound=replace(SystemConfig(**packet["config"]),random_seed=job["seed"]).as_dict()
            bound["catalog"].update(performance_mode="task_replica_proxy",memory_mode="standalone_proxy")
            bound["catalog"]["measurement_context"]["runtime_checkpoint_sha256"]=job["identity"]["runtime_checkpoint_sha256"]
            def transport(config,plan,*args,priming_policy=None,**options):
                value=native(packet,job);value["config"]=config.as_dict()
                for key in ("experiment_job_id","experiment_phase","confirmation_packet_sha256",
                            "executed_measurement_identity","executed_measurement_id"):
                    value.pop(key)
                return value
            runner=__import__("importlib").import_module(runtime.prefix+".runtime.runner")
            class Borrow:
                def __init__(self,*args):pass
                def __enter__(self):return study
                def __exit__(self,*args):pass
            try:
                with patch("gsdeploy.spillover_confirmation.Runtime",Borrow), \
                        patch.object(study,"bind_native_job",return_value=(bound,root/"manifest.json",packet["workers"])), \
                        patch("subprocess.check_output",return_value="CPU fixture GPU"), \
                        patch.object(runner,"run",side_effect=transport) as mocked, \
                        patch("gsdeploy.evaluation.evaluate_run",return_value={"utility":utility()}):
                    mocked.__signature__=inspect.signature(transport)
                    actual,_=run(packet,job["id"],output,execute=True)
                self.assertEqual(actual["experiment_phase"],"confirmation")
                self.assertEqual(actual["executed_measurement_identity"],job["identity"])
                self.assertEqual(actual["confirmation_packet_sha256"],packet["confirmation_sha256"])
                self.assertTrue((output/"spillover-collection-receipt.json").is_file())
                self.assertEqual(actual["collection_contract"]["collection_performance_mode"],"task_replica_proxy")
            finally:
                runtime.close()

    def test_partial_and_complete_reports_do_not_invent_missing_pairs(self):
        with tempfile.TemporaryDirectory() as tmp,patch("gsdeploy.spillover_confirmation._gate",side_effect=self.gate):
            root=Path(tmp);packet,_,_=fixture(root);packet_path=root/"packet.json";packet_path.write_text(json.dumps(packet));paths=[]
            for job in packet["jobs"]:
                p=root/job["id"]/"run.json";write_native(p,packet,job);paths.append(p)
            with patch("gsdeploy.evaluation.evaluate_run",return_value={"utility":utility()}):
                partial=report(packet_path,paths[:1],root/"partial.json")
                self.assertEqual(partial["status"],"INCOMPLETE_CONFIRMATION")
                self.assertNotIn("student_interval",partial["paired"]["comparisons"][0])
                full=report(packet_path,paths,root/"full.json")
                self.assertEqual(full["status"],"ALL_DECLARED_JOBS_VERIFIED")
                self.assertTrue(full["paired"]["all_planned_blocks_complete"])
                with self.assertRaisesRegex(ValueError,"duplicate"):report(packet_path,paths+[paths[0]],root/"bad.json")


if __name__=="__main__":unittest.main()
