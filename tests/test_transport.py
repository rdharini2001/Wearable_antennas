"""Tests of frequency-marginal response modeling and metric provenance."""
import json
import unittest
import joblib
import numpy as np
import pandas as pd

from src.discovery import ROOT, observations, geometry_groups, _split_indices
from src.enhanced import tune_curve
from src.transport import ConditionalSweepDistribution, transport_evaluate, comparison_with_paper

class TransportModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame=observations()
        cls.group=geometry_groups(cls.frame)
        cls.train,cls.test=_split_indices(cls.frame,cls.group,'frequency_interpolation',2023)
        cls.method,_=tune_curve(cls.frame,cls.group,cls.train,2023)
        cls.model=ConditionalSweepDistribution.fit(cls.frame,cls.train,cls.method)

    def test_frequency_not_required_for_forecast(self):
        geom=self.frame.iloc[self.test[10]][['PatchLength','PatchWidth','SlotLength','SlotWidth']].to_numpy(float)
        q=self.model.quantile(geom,np.linspace(.01,.99,31))
        self.assertEqual(q.shape,(31,))
        self.assertTrue(np.isfinite(q).all())
        self.assertTrue(np.all(np.diff(q)>=-1e-8))
        with self.assertRaises(ValueError):self.model.quantile(np.r_[geom,2.4],[.5])

    def test_fit_is_invariant_to_heldout_label_changes(self):
        altered=self.frame.copy()
        altered.loc[self.test,'Strength']=altered.loc[self.test,'Strength'].to_numpy()+999
        predictor=ConditionalSweepDistribution.fit(altered,self.train,self.method)
        for geom in self.model.geometries:
            np.testing.assert_array_equal(self.model.quantile(geom,[.1,.5,.9]),predictor.quantile(geom,[.1,.5,.9]))

    def test_transport_has_order_invariant_score(self):
        first,_=transport_evaluate(self.frame,self.train,self.test,self.model)
        altered=self.frame.copy()
        for g in np.unique(self.group[self.test]):
            select=self.test[self.group[self.test]==g]
            altered.loc[select,'Strength']=altered.loc[select,'Strength'].to_numpy()[::-1]
        second,_=transport_evaluate(altered,self.train,self.test,self.model)
        for k in ('w2_squared_db2','w2_db','w1_db','distribution_explained_variance'):
            self.assertAlmostEqual(first[k],second[k],places=11)

    def test_saved_distribution_replays_export(self):
        saved=joblib.load(ROOT/'results'/'transport'/'conditional_sweep_distribution.joblib')
        matches=pd.read_csv(ROOT/'results'/'transport'/'rank_couplings.csv')
        for geom,chunk in matches.groupby('geometry'):
            x=np.asarray([float(z) for z in geom.split('|')])
            np.testing.assert_allclose(saved.fixed_quantile_grid(x,len(chunk)),
                                       chunk.sort_values('quantile_rank').predicted_quantile_db.to_numpy(),atol=1e-9)

    def test_paper_report_is_post_training(self):
        output=comparison_with_paper()
        self.assertEqual(output['total_thresholds'],96)
        self.assertEqual(output['numeric_thresholds_met'],96)
        self.assertEqual(output['pointwise_thresholds_met'],48)
        self.assertEqual(output['distributional_thresholds_met'],48)
        self.assertFalse(output['exact_paper_reproduction'])
        self.assertFalse(output['same_definition_for_all_96'])
        scores=pd.read_csv(ROOT/'results'/'transport'/'v2_96_threshold_comparison.csv')
        self.assertEqual(len(scores),96)
        self.assertTrue(scores[scores.frequency_mode=='with'].same_metric_definition.all())
        self.assertFalse(scores[scores.frequency_mode=='without'].same_metric_definition.any())
        self.assertEqual(scores[scores.frequency_mode=='without'].v2_metric.nunique(),4)

    def test_stability_and_site_artifacts(self):
        seeds=pd.read_csv(ROOT/'results'/'transport'/'seed_stability.csv')
        self.assertEqual(len(seeds),4)
        self.assertGreater(seeds.w2_db.max(),0)
        site=(ROOT/'docs'/'index.html').read_text()
        for phrase in ('WEAVE V2','transportChart','v2W2','v2BlindWins','v2_96_threshold_comparison.csv'):
            self.assertIn(phrase,site)
        self.assertIn('not exact paper replication',site)
        self.assertTrue((ROOT/'docs'/'downloads'/'v2_96_threshold_comparison.csv').exists())

if __name__=='__main__':unittest.main()
