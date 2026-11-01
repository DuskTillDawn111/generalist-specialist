import copy
import unittest

from gsdeploy.forecasting.simulator import FullHorizonModel
from tools.predictor_validation_20261027.background_statistical import BackgroundStatisticalProvider,fit_traces


def request(rid,alias,family,task,duration,pool="input"):
    return {"request_id":rid,"alias":alias,"family":family,"task":task,"pool_id":pool,
        "handler_start_ns":0,"handler_end_ns":duration,
        "segments":[{"phase":"prefill","start_ns":0,"end_ns":duration//2},
                    {"phase":"decode","start_ns":duration//2,"end_ns":duration}]}


def fixture():
    traces = []
    def add(rows):
        traces.append({"id":str(len(traces)),"split":"training","execution_context_sha256":"v7-fixture","requests":rows})
    for repeat in range(2):
        add([request("s","s","text","text",100_000_000)])
        for task in ("image","asr"):
            add([request("g","g","generalist_2",task,100_000_000)])
        add([request("s","s","text","text",150_000_000),request("g","g","generalist_2","image",150_000_000)])
    return fit_traces(traces)


class BackgroundStatisticalTests(unittest.TestCase):
    def setUp(self):
        self.doc = fixture();self.provider = BackgroundStatisticalProvider(self.doc,"v7-fixture")
        self.aliases = {"s":{"family":"text","supported_tasks":["text"]},
                        "g":{"family":"generalist_2","supported_tasks":["image","asr"]}}

    def active(self,alias,task,phase="prefill"):
        profile = self.provider.lookup[(self.aliases[alias]["family"],task,"input")]
        return {"alias":alias,"task":task,"pool_id":"input","profile_id":profile["id"],"segment_id":phase}

    def test_seen_image_pair_known_unseen_asr_same_familycounts_unknown(self):
        s,g = self.active("s","text"),self.active("g","image")
        value = self.provider.service(s,{"s":s,"g":g},self.aliases,"mean")
        self.assertTrue(value["known"]);self.assertFalse(value["competitor_task_pooling"])
        other = self.active("g","asr")
        unknown = self.provider.service(s,{"s":s,"g":other},self.aliases,"mean")
        self.assertFalse(unknown["known"])
        self.assertEqual(unknown["reason"],"UNSEEN_TARGET_FULL_FAMILY_TASK_MIXTURE")

    def test_unseen_phase_interpolation_stays_within_seen_task_mixture(self):
        s,g = self.active("s","text"),self.active("g","image","decode")
        value = self.provider.service(s,{"s":s,"g":g},self.aliases,"mean")
        self.assertTrue(value["known"]);self.assertTrue(value["interpolated"])
        self.assertFalse(value["physically_identified"])

    def test_task_aware_full_event_forecast(self):
        model = FullHorizonModel(self.aliases,self.provider)
        queues = {"s":[{"request_id":"s","task":"text","pool_id":"input","arrival_ns":0}],
                  "g":[{"request_id":"g","task":"image","pool_id":"input","arrival_ns":0}]}
        forecast = model.forecast(queues,0,1_000_000_000,require_energy=False)
        self.assertEqual(set(forecast["finishes"]),{"s","g"})
        self.assertFalse(forecast["energy_known"])

    def test_actual_coefficients_have_distinct_task_context_ids(self):
        mixtures = self.doc["training_task_mixtures"]
        self.assertTrue(any(any(m["task"]=="image" for m in mix) for mix in mixtures))
        self.assertTrue(any(any(m["task"]=="asr" for m in mix) for mix in mixtures))
        targets = [p["target"] for p in self.doc["base_model"]["handler_models"]]
        self.assertEqual(len(targets),3)
        self.assertEqual({t[1] for t in targets},{"text","image","asr"})

    def test_no_frozen_source_modifications_or_cross_identity_transfer(self):
        with self.assertRaisesRegex(ValueError,"identity"):
            BackgroundStatisticalProvider(self.doc,"old-v6")
        mutated = copy.deepcopy(self.doc);mutated["base_model"]["handler_models"][0]["progress"] = {}
        with self.assertRaisesRegex(ValueError,"Changed"):
            BackgroundStatisticalProvider(mutated,"v7-fixture")


if __name__=="__main__":unittest.main()
