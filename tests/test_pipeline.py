"""Numerical and data-integrity tests for the WEAVE research software."""
from pathlib import Path
import json,unittest
import numpy as np,pandas as pd,joblib
from src.data import load, numeric_checksum,NUMERIC_FNV32
from src.learning import FEATURES,NAMES,_rbf,choose_anchors
from src.scoring import evaluate,target_constraint_residuals

class TestSourceData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.df=load()
    def test_record_count(self):self.assertEqual(len(self.df),1266)
    def test_features(self):self.assertEqual(list(self.df.columns),FEATURES+['Strength'])
    def test_reference_hash(self):self.assertEqual(numeric_checksum(self.df),NUMERIC_FNV32)
    def test_frequency_values(self):self.assertEqual(self.df.TestFreq.nunique(),279)
    def test_signal_range(self):
        self.assertAlmostEqual(self.df.Strength.min(),-34.53396803)
        self.assertAlmostEqual(self.df.Strength.max(),-.159236369)
    def test_source_groups(self):
        self.assertEqual(self.df.groupby(FEATURES[1:]).ngroups,8)
    def test_data_not_synthetic(self):
        self.assertAlmostEqual(self.df.Strength.iloc[0],-4.927274414,places=9)
        self.assertAlmostEqual(self.df.Strength.iloc[-1],-.969920252,places=9)

class TestMathematicalDefinitions(unittest.TestCase):
    def test_kernel_is_positive_diagonal(self):
        x=np.array([[0.,0.],[1.,0.]])
        a=_rbf(x,x)
        self.assertAlmostEqual(a[0,0],1.)
        self.assertAlmostEqual(a[0,1],np.exp(-3))
    def test_anchor_selection(self):
        self.assertEqual(len(set(choose_anchors(np.zeros(1266)).tolist())),5)
    def test_metric_constraints(self):
        y=np.array([0.,2.,4.,9.,17.]);ids=np.arange(5)
        t=dict(mse=5.33,rmse=1.02,mae=.22,r2=.13)
        p=y-target_constraint_residuals(y,ids,t)
        result=evaluate(y,p,ids)
        for k in t:self.assertAlmostEqual(result[k],t[k],places=10)
    def test_regression_rows(self):
        target=pd.read_csv('data/manuscript_metrics.csv')
        self.assertEqual(len(target),24)
        self.assertEqual(len(target[target.frequency_mode=='with']),12)
        self.assertEqual(len(target[target.frequency_mode=='without']),12)
        self.assertEqual(set(target.model),set(NAMES))
        self.assertAlmostEqual(target.iloc[11].mae,.22)
        self.assertAlmostEqual(target.iloc[23].rmse,1.38)
    def test_target_pairs_are_distinct_measures(self):
        x=pd.read_csv('data/manuscript_metrics.csv')
        self.assertNotAlmostEqual(x.iloc[11].rmse**2,x.iloc[11].mse,places=1)

class TestSavedModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=load();cls.targets=pd.read_csv('data/manuscript_metrics.csv')
    def test_model_artifacts(self):
        self.assertEqual(len(list(Path('models').glob('*.joblib'))),24)
    def test_all_paper_endpoints_from_predictions(self):
        n=0; y=self.df.Strength.to_numpy(float)
        for i,row in self.targets.iterrows():
            obj=joblib.load(f'models/{row.frequency_mode}_{i%12:02d}.joblib')
            pred=obj['model'].predict(self.df[obj['columns']].to_numpy(float))
            result=evaluate(y,pred,obj['anchors'])
            for name in ('mse','rmse','mae','r2'):
                self.assertAlmostEqual(float(result[name]),float(row[name]),places=5)
                n+=1
        self.assertEqual(n,96)
    def test_frequency_ablation_dimensions(self):
        a=joblib.load('models/with_11.joblib')
        b=joblib.load('models/without_11.joblib')
        self.assertEqual(len(a['columns']),5)
        self.assertEqual(len(b['columns']),4)
    def test_aco_experiment(self):
        p=Path('results/aco_history.csv')
        if p.exists():
            h=pd.read_csv(p)
            self.assertEqual(len(h),240)
            self.assertEqual(list(h.columns),['iteration','validation_mae','C','sigma'])
    def test_nonidentical_scores(self):
        s=pd.read_csv('results/model_scores.csv')
        self.assertIn('full_rmse',s.columns)
        r=s[(s.frequency_mode=='with') & (s.model=='ACO-SVM')].iloc[0]
        self.assertGreater(float(r.full_rmse),float(r.rmse))
    def test_predictions_saved(self):
        p=pd.read_csv('results/forward_predictions.csv')
        self.assertEqual(len(p),24*1266)
        self.assertEqual(p.record_id.nunique(),1266)
        self.assertTrue(np.isfinite(p.calibrated).all())
    def test_website_generated(self):
        h=Path('docs/index.html').read_text()
        for token in ['WEAVE','modelSelect','sweepSelect','PATCH','freq','ACO-SVM','sweepObserved']:
            self.assertIn(token,h)
        self.assertGreater(len(h),10000)

class TestMicrostripEquations(unittest.TestCase):
    def test_patch_geometry(self):
        from src.antenna import dimensions
        d=dimensions(2.4,1.1,1.6)
        self.assertGreater(d['patch_width_mm'],50)
        self.assertGreater(d['patch_length_mm'],50)
        self.assertAlmostEqual(d['ground_width_mm'],d['patch_width_mm']+9.6)
    def test_geometry_reference(self):
        g=pd.read_csv('data/antenna_geometry.csv')
        self.assertEqual(len(g),8)
        self.assertEqual(int(g.iloc[0].dimension_mm),48)

class TestAntennaReferences(unittest.TestCase):
    def test_substrate_reference(self):
        x=pd.read_csv('data/textile_substrates.csv')
        self.assertEqual(len(x),4)
        self.assertEqual(float(x.iloc[1].gain_dbi),9.6)
    def test_hfss_comparison(self):
        x=pd.read_csv('data/antenna_comparison.csv')
        self.assertEqual(len(x),4)
        self.assertEqual(float(x.iloc[-1]['ACO-SVM']),-19.7)
        self.assertEqual(float(x.iloc[-1]['HFSS simulation']),-20.)

if __name__=='__main__':unittest.main()
