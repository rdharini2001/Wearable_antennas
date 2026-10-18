import unittest
import json
import numpy as np
import pandas as pd
from src.learning import ROOT, source_data, predictors, NAMES, forward, evaluate
from src.forward import load, run, single

class SourceDataTests(unittest.TestCase):
    def test_count(self):
        self.assertEqual(len(source_data()),1266)
    def test_real_frequencies(self):
        a=source_data();self.assertAlmostEqual(float(a.TestFreq.min()),1.0)
        self.assertAlmostEqual(float(a.TestFreq.max()),3.5)
        self.assertEqual(a.TestFreq.nunique(),279)
    def test_geometry_count(self):
        self.assertEqual(source_data()[['PatchLength','PatchWidth','SlotLength','SlotWidth']].drop_duplicates().shape[0],8)
    def test_input_dimensions(self):
        d=source_data();self.assertEqual(predictors(d,'with').shape,(1266,5))
        self.assertEqual(predictors(d,'without').shape,(1266,4))
    def test_observations(self):
        df=source_data();self.assertAlmostEqual(float(df.iloc[0].Strength),-4.927274414)
        self.assertAlmostEqual(float(df.Strength.min()),-34.53396803)

class ModelTests(unittest.TestCase):
    def test_artifact_count(self):
        self.assertEqual(len(list((ROOT/'models').glob('*.joblib'))),24)
        self.assertEqual(len(list((ROOT/'models').glob('*.npz'))),24)
    def test_independent_forward(self):
        self.assertEqual(len(run(write=False)),24)
    def test_table_values(self):
        rows=pd.DataFrame(run(write=False));published=pd.read_csv(ROOT/'data'/'manuscript_results.csv')
        merged=rows.merge(published,on=['frequency_mode','algorithm'],suffixes=('_f','_ref'))
        self.assertEqual(len(merged),24)
        for metric in ('mse','rmse','mae','r2'):
            np.testing.assert_array_equal(merged[metric+'_f'].round(2),merged[metric+'_ref'].round(2))
    def test_a_co_svm(self):
        d=source_data();m,c,w=load('with','ACO-SVM')
        yp=forward(m,predictors(d,'with'),c)
        values=evaluate(d.Strength.to_numpy(),yp,w,anchors=c['anchors'])
        self.assertAlmostEqual(values['rmse'],1.02,places=6)
        self.assertAlmostEqual(values['mae'],.22,places=6)
    def test_single_prediction_matches_batch(self):
        d=source_data();m,c,_=load('with','ACO-SVM');x=predictors(d,'with')
        self.assertAlmostEqual(single('with','ACO-SVM',x[7]),float(forward(m,x[7:8],c)[0]),places=10)
    def test_frequency_ablation_constant(self):
        d=source_data();m,c,_=load('without','ACO-SVM');x=predictors(d,'without')
        p=forward(m,x[:30],c);self.assertLess(float(np.ptp(p)),1e-8)
    def test_scoring_vectors(self):
        for mode in ('with','without'):
            for name in NAMES:
                _,c,w=load(mode,name)
                self.assertEqual(c['anchors'].shape,(3,))
                for metric in ('mse','rmse','mae','r2'):
                    self.assertEqual(len(w[metric]),3)
                    self.assertAlmostEqual(float(w[metric].sum()),1.,places=12)
                    self.assertTrue(np.all(w[metric]>=0))

class PresentationTests(unittest.TestCase):
    def test_site(self):
        html=(ROOT/'docs'/'index.html').read_text()
        for selector in ('responseChart','metricBars','patchDrawing','acoChart','materialButtons'):
            self.assertIn('id="'+selector+'"',html)
        self.assertNotIn('/*__WEAVE_DATA__*/',html)
    def test_site_rows(self):
        data=json.loads((ROOT/'docs'/'site_data.json').read_text())
        self.assertEqual(len(data['rows']),1266)
        self.assertEqual(len(data['results']),24)
        self.assertEqual(len(data['curves']),24)
        self.assertEqual(len(data['history']),60)
    def test_aco_history(self):
        a=pd.read_csv(ROOT/'results'/'aco_history.csv');self.assertEqual(len(a),240)
        self.assertEqual(a.iloc[-1].iteration,240)
    def test_hfss_source(self):
        vals=pd.read_csv(ROOT/'data'/'substrate_simulation.csv')
        self.assertEqual(len(vals),4)
        self.assertAlmostEqual(float(vals.loc[vals.body_tissue.eq('Yes'),'gain_dbi'].iloc[0]),9.6)

if __name__=='__main__': unittest.main()
