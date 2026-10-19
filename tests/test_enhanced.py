"""Regression, spectral reconstruction, and scoring checks."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from src.discovery import ROOT,observations,geometry_groups,metrics
from src.enhanced import curve_predict,tune_curve,WearablePredictor

class FrequencyModels(unittest.TestCase):
    def test_local_interpolation_not_test_targets(self):
        f=[1.,1.5,2.,2.5,3.,3.5];y=[1,0,-1,0,1,0]
        o=curve_predict(f,y,[1.75,2.75],'cubic')
        self.assertEqual(len(o),2);self.assertTrue(np.all(np.isfinite(o)))
        self.assertAlmostEqual(curve_predict(f,y,[1.5],'cubic')[0],0,places=12)

    def test_training_only_curve_selection(self):
        df=observations();group=geometry_groups(df)
        from src.discovery import _split_indices
        tr,te=_split_indices(df,group,'frequency_interpolation',2023)
        method,cv=tune_curve(df,group,tr,2023)
        self.assertIn(method,('cubic','pchip','akima','linear'))
        self.assertEqual(len(cv),4)
        self.assertTrue(np.isfinite(list(cv.values())).all())

    def test_saved_model_replays_test_predictions(self):
        df=observations();groups=geometry_groups(df)
        stored=pd.read_csv(ROOT/'results'/'enhanced'/'predictions.csv')
        tests=[('frequency_interpolation','with','ACO-SVM','spectral_residual_hybrid'),
               ('frequency_interpolation','with','Decision Tree Regressor','spectral_residual_hybrid'),
               ('frequency_interpolation','without','Support Vector Regression','geometry_pooled_blind'),
               ('geometry_holdout','with','Linear Regression','kernel_regression')]
        for protocol,mode,name,variant in tests:
            with self.subTest(protocol=protocol,mode=mode,name=name):
                slug=name.lower().replace(' ','_').replace('-','_')
                path=ROOT/'results'/'enhanced'/f'{protocol}__{mode}__{slug}.joblib'
                obj=joblib.load(path)
                self.assertIsInstance(obj,WearablePredictor)
                data=stored[(stored.protocol==protocol)&(stored.frequency_mode==mode)&
                            (stored.algorithm==name)&(stored.variant==variant)].sort_values('row')
                ix=data.row.to_numpy(int)
                features=['TestFreq','PatchLength','PatchWidth','SlotLength','SlotWidth'] if mode=='with' else ['PatchLength','PatchWidth','SlotLength','SlotWidth']
                result=obj.predict(df.loc[ix,features].to_numpy(float))
                np.testing.assert_allclose(result,data.predicted_db.to_numpy(float),atol=1e-10,rtol=1e-10)

    def test_no_frequency_variation(self):
        path=ROOT/'results'/'enhanced'/'frequency_interpolation__without__aco_svm.joblib'
        model=joblib.load(path)
        X=np.array([[29.4,38.,14.,24.]]*4)
        self.assertAlmostEqual(float(np.ptp(model.predict(X))),0.,places=12)

    def test_shared_evaluation_populations(self):
        measured=pd.read_csv(ROOT/'results'/'enhanced'/'metrics.csv')
        np.testing.assert_allclose(measured.rmse.to_numpy()**2,measured.mse.to_numpy(),atol=1e-9,rtol=1e-10)
        self.assertTrue((measured.n_test>0).all())
        self.assertEqual(set(measured.frequency_mode),{'with','without'})
        self.assertEqual(len(set(measured.algorithm)),12)

    def test_publication_site(self):
        home=(ROOT/'docs'/'index.html').read_text()
        for identifier in ('response-chart','antenna-atlas.svg','baseline-table',
                           'quantile-chart','SUPPLEMENTARY_METHODS.pdf'):
            self.assertIn(identifier,home)
        self.assertNotIn('study-one.html',home)
        self.assertNotIn('study-two.html',home)
        self.assertTrue((ROOT/'docs'/'SUPPLEMENTARY_METHODS.pdf').exists())
        self.assertTrue((ROOT/'docs'/'downloads'/'regression_metrics.csv').exists())
        self.assertNotIn('Paper MSE',home)
        self.assertNotIn('paper_comparison.csv',home)

if __name__=='__main__':unittest.main()
