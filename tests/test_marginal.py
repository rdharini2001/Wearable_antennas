"""Validation of geometry-only probabilistic inference and exported evidence."""
import json
import unittest
import joblib
import numpy as np
import pandas as pd
from src.discovery import ROOT, observations, geometry_groups, _split_indices
from src.marginal import crps_sample, conformal_quantile, fit_marginal, run

class MarginalForecastTests(unittest.TestCase):
    def test_crps_collapses_to_absolute_error_for_point_mass(self):
        self.assertAlmostEqual(crps_sample(2,[0]),2)
        self.assertAlmostEqual(crps_sample(1,[0,2]),.5)
        self.assertTrue(crps_sample(-.5,[0,2])>=0)

    def test_conformal_quantile(self):
        q=conformal_quantile(np.arange(1,20),alpha=.1)
        self.assertEqual(q,18)
        self.assertEqual(conformal_quantile(np.arange(20),alpha=.1),18)

    def test_disjoint_training_calibration(self):
        frame=observations();group=geometry_groups(frame)
        tr,te=_split_indices(frame,group,'frequency_interpolation',2023)
        obj,setup=fit_marginal(frame,group,tr)
        self.assertEqual(setup['training_distribution_rows']+setup['calibration_rows'],len(tr))
        self.assertEqual(setup['conditional_designs'],8)
        self.assertEqual(len(obj.samples),8)
        self.assertTrue(all(len(x)>5 for x in obj.samples))
        self.assertTrue(np.isfinite(obj.predict(frame[list(frame.columns[1:5])].to_numpy()[:10])).all())

    def test_oracle_floor_and_artifact_replay(self):
        evidence=json.loads((ROOT/'results'/'marginal'/'metrics.json').read_text())
        self.assertGreater(evidence['pointwise_mean']['mse'],evidence['test_oracle_floor']['mse_minimum']-1e-10)
        self.assertGreater(evidence['pointwise_median']['mae'],evidence['test_oracle_floor']['mae_minimum']-1e-10)
        self.assertGreaterEqual(evidence['probabilistic']['interval_coverage'],0)
        self.assertLessEqual(evidence['probabilistic']['interval_coverage'],1)
        forecast=pd.read_csv(ROOT/'results'/'marginal'/'forecasts.csv')
        predictor=joblib.load(ROOT/'results'/'marginal'/'frequency_blind_distribution.joblib')
        frame=observations()
        idx=forecast.row.to_numpy(int)
        x=frame.iloc[idx][['PatchLength','PatchWidth','SlotLength','SlotWidth']].to_numpy(float)
        np.testing.assert_allclose(predictor.predict(x),forecast.mean_db.to_numpy(),atol=1e-12)
        np.testing.assert_allclose(predictor.interval(x),forecast[['interval_lower_db','interval_upper_db']].to_numpy(),atol=1e-12)

    def test_website_has_study_sources(self):
        site=(ROOT/'docs'/'index.html').read_text()
        for phrase in ('frequency-marginal','quantile-chart','distribution','SUPPLEMENTARY_METHODS.pdf'):
            self.assertIn(phrase,site)
        self.assertTrue((ROOT/'results'/'marginal'/'metrics.json').exists())
        score=json.loads((ROOT/'results'/'enhanced'/'scorecard.json').read_text())
        feasibility=score['frequency_blind_pointwise_feasibility']
        self.assertLessEqual(feasibility['overall_pointwise_threshold_ceiling'],len(pd.read_csv(ROOT/'data'/'manuscript_results.csv'))*4)
        self.assertTrue(all(item['metric'] in ('mse','rmse','mae','r2') for item in feasibility['unattainable_thresholds_on_fixed_split']))
        self.assertEqual(len(pd.read_csv(ROOT/'results'/'marginal'/'split_stability.csv')),4)

if __name__=='__main__':unittest.main()
