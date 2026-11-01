import copy
import unittest

import numpy as np

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.simulator import FullHorizonModel
from tools.predictor_validation_20261027.statistical_profiles import (
    StatisticalProvider, bounded_ridge, digest, fit_traces, full_count,
)


def row(rid,alias,family,start,end,pool="input"):
    midpoint = (start+end)//2
    return {"request_id":rid,"alias":alias,"family":family,"task":"text","pool_id":pool,
            "handler_start_ns":start,"handler_end_ns":end,
            "segments":[{"phase":"prefill","start_ns":start,"end_ns":midpoint},
                        {"phase":"decode","start_ns":midpoint,"end_ns":end}]}


def fixture():
    traces = []
    for repeat in range(2):
        for family,alias in (("text","s"),("generalist_2","g")):
            traces.append({"id":f"solo:{family}:{repeat}","split":"training",
                "execution_context_sha256":"ctx","requests":[row("r",alias,family,0,100_000_000)]})
        traces.append({"id":f"pair:{repeat}","split":"training","execution_context_sha256":"ctx",
            "requests":[row("s","s","text",0,200_000_000),row("g","g","generalist_2",0,200_000_000)]})
    return fit_traces(traces)


class StatisticalProfilesTests(unittest.TestCase):
    def setUp(self):
        self.doc = fixture()
        self.provider = StatisticalProvider(self.doc,"ctx")
        self.aliases = {"s":{"family":"text","supported_tasks":["text"]},
                        "g":{"family":"generalist_2","supported_tasks":["text"]}}

    def active(self,alias,phase="prefill"):
        profile = self.provider.lookup[(self.aliases[alias]["family"],"text","input")]
        return {"alias":alias,"task":"text","pool_id":"input","profile_id":profile["id"],"segment_id":phase}

    def test_rank_deficiency_is_statistical_prediction_not_physical(self):
        speed,diagnostic = bounded_ridge([[1.,1.],[1.,1.]],[1.,1.])
        self.assertEqual(diagnostic["matrix_rank"],1)
        self.assertTrue(np.all((speed>=.05)&(speed<=8)))
        self.assertFalse(diagnostic["physically_identified"])

    def test_pair_real_context_and_unseen_multiphase_fallback(self):
        s,g = self.active("s"),self.active("g")
        known = self.provider.service(s,{"s":s,"g":g},self.aliases,"mean")
        self.assertTrue(known["known"])
        self.assertFalse(known["interpolated"])
        g["segment_id"] = "decode"
        fallback = self.provider.service(s,{"s":s,"g":g},self.aliases,"mean")
        self.assertTrue(fallback["known"])
        self.assertTrue(fallback["interpolated"])
        fk,_ = full_count([{"family":"text"},{"family":"generalist_2"}])
        expected = self.provider.handler[("text","text","input")]["progress"][fk]
        self.assertAlmostEqual(fallback["progress"],expected)
        self.assertFalse(fallback["independently_validated"])

    def test_no_pairwise_extrapolation_to_unseen_threeway(self):
        aliases = {**self.aliases,"s2":{"family":"text","supported_tasks":["text"]}}
        s,g = self.active("s"),self.active("g")
        third = {**s,"alias":"s2"}
        value = self.provider.service(s,{"s":s,"s2":third,"g":g},aliases,"mean")
        self.assertFalse(value["known"])
        self.assertEqual(value["reason"],"UNSEEN_TARGET_FULL_FAMILY_COUNT")

    def test_unknown_input_and_uncalibrated_scenario(self):
        with self.assertRaises(UnknownEvidence):
            self.provider.plan({"alias":"s","task":"text","pool_id":"new"},self.aliases,"mean")
        with self.assertRaises(UnknownEvidence):
            self.provider.plan({"alias":"s","task":"text","pool_id":"input"},self.aliases,"p90")
        self.assertIsNone(self.provider.power({},self.aliases,"mean")["watts"])
        self.assertIsNone(self.provider.memory({},self.aliases,"mean")["peak_memory_mib"])

    def test_fingerprint_and_identity(self):
        mutated = copy.deepcopy(self.doc);mutated["profiles"][0]["segments"][0]["duration_ms"] *= 2
        with self.assertRaises(ValueError):StatisticalProvider(mutated,"ctx")
        with self.assertRaises(ValueError):StatisticalProvider(self.doc,"other")

    def test_validation_rows_do_not_change_fitted_coefficients(self):
        traces = [{"id":"v","split":"validation","execution_context_sha256":"ctx","requests":[row("v","s","text",0,99_000_000_000)]}]
        # Fixtures expose the same deterministic training design directly.
        train = []
        for rep in range(2):
            for family,alias in (("text","s"),("generalist_2","g")):
                train.append({"id":f"solo:{family}:{rep}","split":"training","execution_context_sha256":"ctx","requests":[row("r",alias,family,0,100_000_000)]})
            train.append({"id":f"pair:{rep}","split":"training","execution_context_sha256":"ctx","requests":[row("s","s","text",0,200_000_000),row("g","g","generalist_2",0,200_000_000)]})
        mixed = fit_traces(train+traces)
        self.assertEqual(mixed["handler_models"],self.doc["handler_models"])
        self.assertEqual(mixed["semantic_models"],self.doc["semantic_models"])
        self.assertEqual(mixed["coverage"]["validation_batches_excluded"],1)

    def test_full_event_simulation_duck_api(self):
        model = FullHorizonModel(self.aliases,self.provider)
        forecast = model.forecast({"s":[{"request_id":"s","task":"text","pool_id":"input","arrival_ns":0}],
                                   "g":[{"request_id":"g","task":"text","pool_id":"input","arrival_ns":0}]},
                                  0,1_000_000_000,require_energy=False)
        self.assertEqual(len(forecast["finishes"]),2)
        self.assertEqual(forecast["incomplete_ids"],[])
        self.assertFalse(forecast["energy_known"])
        self.assertFalse(forecast["memory_known"])


if __name__=="__main__":
    unittest.main()
