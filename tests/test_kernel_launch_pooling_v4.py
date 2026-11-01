"""CPU fixtures for the separate geometry schema, never capture evidence."""
import copy
import unittest

from tools.predictor_validation_20261027.kernel_launch_pooling_v4 import geometry,pooling_group,fit_group,predict_group


def signature(grid):
    return dict(name='fixture_typed_kernel<float>',grid=grid,block=[128,1,1],
        registers_per_thread=16,static_shared_bytes=0,dynamic_shared_bytes=0,stream_role=0)


def rows():
    return [dict(run_id=run,features=geometry(signature([grid,1,1])),duration_ns=100+10*grid)
        for run in ('cpu_fixture_run_1','cpu_fixture_run_2') for grid in (10,20)]


class LaunchPoolingTests(unittest.TestCase):
    def test_grid_work_is_explicit_geometry_and_rejects_zero(self):
        self.assertEqual(geometry(signature([2,3,4])),dict(grid_x=2,grid_y=3,grid_z=4,grid_blocks=24))
        with self.assertRaises(ValueError):geometry(signature([0,3,4]))

    def test_precision_and_static_resources_separate_pooling_domains(self):
        a=pooling_group({},'fixture','prefill',1,dict(dtype='float'),signature([10,1,1]))
        b=pooling_group({},'fixture','prefill',1,dict(dtype='float'),signature([20,1,1]))
        self.assertEqual(a,b)
        c=copy.deepcopy(b);c['launch_settings']['registers_per_thread']=32
        self.assertNotEqual(a,c)
        self.assertNotEqual(a,pooling_group({},'fixture','decode',1,dict(dtype='float'),signature([10,1,1])))

    def test_minimum_three_observations_and_two_real_runs_not_lowered(self):
        self.assertFalse(fit_group(rows()[:2])['fit_accepted'])
        one_run=copy.deepcopy(rows())
        for r in one_run:r['run_id']='one_cpu_fixture'
        self.assertFalse(fit_group(one_run)['fit_accepted'])

    def test_fit_predicts_intermediate_geometry_without_target_fallback(self):
        r=predict_group(fit_group(rows()),geometry(signature([15,1,1])))
        self.assertTrue(r['known']);self.assertAlmostEqual(r['duration_hypothesis_ns'],250)
        self.assertFalse(r['measured_target_fallback_used'])

    def test_outside_geometry_hull_stays_unqualified_numeric_hypothesis(self):
        r=predict_group(fit_group(rows()),geometry(signature([30,1,1])))
        self.assertFalse(r['known']);self.assertAlmostEqual(r['duration_hypothesis_ns'],400)
        self.assertIn('OUTSIDE_TRAINING_GEOMETRY_CONVEX_HULL',r['reasons'])

    def test_grid_axis_configuration_cannot_be_replaced_by_equal_volume(self):
        r=predict_group(fit_group(rows()),geometry(signature([1,15,1])))
        self.assertFalse(r['known']);self.assertIn('OUTSIDE_TRAINING_GEOMETRY_AFFINE_SUBSPACE',r['reasons'])

    def test_bad_training_residual_is_retained_with_executable_hypothesis(self):
        data=rows();data[-1]['duration_ns']=2000
        model=fit_group(data);self.assertFalse(model['fit_accepted'])
        r=predict_group(model,geometry(signature([15,1,1])))
        self.assertIsNotNone(r['duration_hypothesis_ns']);self.assertFalse(r['known'])
        self.assertIn('TRAINING_RESIDUAL_EXCEEDS_25_PERCENT',r['reasons'])


if __name__=='__main__':unittest.main()
