import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.discovery import (SearchConfig, geometry_groups, metrics, choose_svr, discover,
                           epsilon_from_data, observations)
from src.audit import audit, compatible_mse_rmse, full_population_lower_bound
from src.physics import patch_dimensions


def sample():
    rows = []
    for g in range(8):
        for j in range(12):
            frequency = 1.5 + 2*j/11
            value = np.sin(frequency*3 + 0.12*g) + 0.1*g
            rows.append([frequency, 25+g, 30+g, g/2, 3+g, value])
    return pd.DataFrame(rows, columns=['TestFreq','PatchLength','PatchWidth','SlotLength','SlotWidth','Strength'])


class ScientificProtocol(unittest.TestCase):
    def test_grouped_sweeps(self):
        frame=sample()
        groups=geometry_groups(frame)
        self.assertEqual(len(set(groups)),8)
        for i in range(8): self.assertEqual(len(set(groups[i*12:(i+1)*12])),1)

    def test_metrics_consistent(self):
        s=metrics([1,2,3], [2,2,5])
        self.assertAlmostEqual(s['rmse']**2,s['mse'],places=12)
        self.assertAlmostEqual(s['mae'],1.)

    def test_discovery_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'source.csv';sample().to_csv(path,index=False)
            config=SearchConfig(iterations=3,ants=3,evaluations=11,folds=2)
            result=discover(path,Path(td)/'out',config=config)
            self.assertEqual(len(result['metrics']),96)
            for protocol, indices in result['partitions'].items():
                self.assertEqual(len(set(indices['test']).intersection(indices['train'])),0)
            self.assertEqual(len(result['calibration']),2)
            self.assertTrue(all(v['C']>0 and v['sigma']>0
                  for protocol in result['calibration'].values() for v in protocol.values()))
            for row in result['metrics']:
                self.assertTrue(np.isclose(row['rmse']**2,row['mse'],rtol=1e-12,atol=1e-9))

    def test_patch_dimensions(self):
        result=patch_dimensions(2.4, 1.1, 1.6)
        self.assertGreater(result['patch_width_mm'],result['patch_length_mm'])
        self.assertAlmostEqual(result['ground_width_mm']-result['patch_width_mm'],9.6,places=7)
        self.assertAlmostEqual(result['ground_length_mm']-result['patch_length_mm'],9.6,places=7)
        self.assertGreater(result['fringing_extension_mm'],0)
        with self.assertRaises(ValueError):patch_dimensions(-1,1.1,1.6)

    def test_audit_detects_impossible_pair(self):
        self.assertFalse(compatible_mse_rmse(5.33, 1.02))
        self.assertTrue(compatible_mse_rmse(8.52, 2.92))

    def test_frequency_absent_lower_bound(self):
        frame=sample()
        actual=full_population_lower_bound(frame)
        self.assertGreaterEqual(actual['mse'],0)
        groups=geometry_groups(frame)
        means=np.array([np.mean(frame.Strength.to_numpy()[groups==g]) for g in groups])
        self.assertAlmostEqual(actual['mse'],np.mean((frame.Strength.to_numpy()-means)**2))

if __name__ == '__main__': unittest.main()
